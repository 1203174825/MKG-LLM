"""
Stage 2: Test script using the same forward-pass inference as train_stage2.py.

Uses batched forward pass (no autoregressive generation) for fast evaluation,
exactly matching the validate() method in train_stage2.py.
"""
import os
import re
import json
import pickle
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.models.stage2 import KGAlignmentLayer, ModifiedEmbedding
from src.utils.metrics import compute_cls_metrics, compute_reg_metrics
from src.utils.config import CONFIG

SIGNAL_TOKEN_ID = 151925
N_KG_TOKENS = 5
DELAY_MEAN = 5.9
DELAY_STD = 23.6
PROMPT_SYSTEM = (
    "As an expert in flight delay prediction with extensive knowledge in "
    "aviation operations, weather impacts, and air traffic management, "
    "you can assess flight delay status. Based on the flight information "
    "and knowledge graph features, answer my questions."
)


def load_kg_cache(cache_dir, years, months):
    """Load KG features from cache directory structure (same as train_stage2)."""
    if not os.path.exists(cache_dir):
        return None

    metadata_file = os.path.join(cache_dir, 'metadata.json')
    if not os.path.exists(metadata_file):
        return None

    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    print(f"\nLoading KG features from {cache_dir}...")
    print(f"  Total cached samples: {metadata.get('total_samples', 'unknown')}")

    all_feats = {}
    for year in years:
        year_str = f"{year:04d}"
        year_dir = os.path.join(cache_dir, year_str)
        if not os.path.exists(year_dir):
            continue

        for month in months:
            month_str = f"{month:02d}"
            month_dir = os.path.join(year_dir, month_str)
            if not os.path.exists(month_dir):
                continue

            import calendar
            _, days_in_month = calendar.monthrange(year, month)
            for day in range(1, days_in_month + 1):
                day_str = f"{day:02d}"
                day_file = os.path.join(month_dir, f"{day_str}.pkl")
                if not os.path.exists(day_file):
                    continue

                with open(day_file, 'rb') as f:
                    day_data = pickle.load(f)

                day_feats = day_data['features']
                for idx, feat in day_feats.items():
                    all_feats[(year, month, day, idx)] = torch.tensor(
                        feat, dtype=torch.float32
                    )

    print(f"  Loaded {len(all_feats)} KG features")
    return all_feats


def test_stage2(stage1_ckpt: str = None, stage2_dir: str = None, batch_size: int = 4):
    """Run Stage 2 evaluation using batched forward pass (same as training validate)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from src.train.train_stage2 import Stage2Dataset, collect_samples_for_days, Stage1Model
    from src.data.kg_builder import DailyKGBuilder

    if stage1_ckpt is None:
        stage1_ckpt = os.path.join(CONFIG.paths.output_dir, 'stage1_best.pt')
    if stage2_dir is None:
        stage2_dir = os.path.join(CONFIG.paths.output_dir, 'stage2_best_val_auc')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nLoading Stage 2 model from {stage2_dir}')

    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG.paths.llm_weight_dir, trust_remote_code=True
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    qwen_model = AutoModelForCausalLM.from_pretrained(
        CONFIG.paths.llm_weight_dir,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    lora_path = os.path.join(stage2_dir, 'lora_adapter')
    if os.path.exists(lora_path):
        qwen_model = PeftModel.from_pretrained(qwen_model, lora_path)
        print(f'Loaded LoRA adapter from {lora_path}')

    aligner = KGAlignmentLayer(n_tokens=N_KG_TOKENS)
    aligner_path = os.path.join(stage2_dir, 'aligner.pt')
    if os.path.exists(aligner_path):
        aligner.load_state_dict(torch.load(aligner_path, map_location=device))
        print(f'Loaded alignment layer from {aligner_path}')

    original_embedding = qwen_model.get_input_embeddings()
    modified_embedding = ModifiedEmbedding(original_embedding, aligner)
    qwen_model.set_input_embeddings(modified_embedding)

    # Recreate exact same components as in LLMTrainer.__init__
    llm_pool = torch.nn.Sequential(
        torch.nn.Linear(1536, 384),
        torch.nn.GELU(),
        torch.nn.Dropout(0.1),
    )
    kg_pool = torch.nn.Linear(384, 384)
    fusion_gate = torch.nn.Module()
    fusion_gate.alpha = torch.nn.Parameter(torch.tensor(0.5))
    cls_head = torch.nn.Linear(384, 1)
    reg_head = torch.nn.Linear(384, 1)

    # Load saved weights
    for name, module in [
        ('llm_pool', llm_pool),
        ('kg_pool', kg_pool),
        ('fusion_gate', fusion_gate),
        ('cls_head', cls_head),
        ('reg_head', reg_head),
    ]:
        path = os.path.join(stage2_dir, f'{name}.pt')
        if os.path.exists(path):
            module.load_state_dict(torch.load(path, map_location=device))
            print(f'Loaded {name} from {path}')

    # Move all to device and eval mode
    qwen_model.eval().to(device)
    aligner.eval().to(device)
    llm_pool.eval().to(device)
    kg_pool.eval().to(device)
    cls_head.eval().to(device)
    reg_head.eval().to(device)

    # Load pre-computed KG features
    print('\nLoading pre-computed KG features from Stage 1 cache...')
    kg_cache_dir = os.path.join(CONFIG.paths.output_dir, 'src', 'kg_features_cache')
    precomputed_kg_feats = load_kg_cache(
        kg_cache_dir, CONFIG.data.test_years, CONFIG.data.test_months
    )
    if precomputed_kg_feats is None:
        print(f"\n⚠️ No precomputed KG features found at {kg_cache_dir}")
        return

    # Build test dataset (same pattern as train_stage2.py)
    print('\nCollecting test samples...')
    kg_builder = DailyKGBuilder()

    normalizer_path = os.path.join(CONFIG.paths.output_dir, 'normalizer.pkl')
    if os.path.exists(normalizer_path):
        with open(normalizer_path, 'rb') as f:
            normalizer_stats = pickle.load(f)
        kg_builder.feat_sum = normalizer_stats['feat_sum']
        kg_builder.feat_sq_sum = normalizer_stats['feat_sq_sum']
        kg_builder.feat_count = normalizer_stats['feat_count']
        print(f'Loaded normalizer stats from {normalizer_path}')

    # Select test days (2025, 15th day of each month)
    import calendar
    test_days = []
    for month in range(1, 13):
        _, days_in_month = calendar.monthrange(2025, month)
        day = min(15, days_in_month)
        test_days.append((2025, month, day))

    test_samples = collect_samples_for_days(test_days, max_samples_per_day=None)
    print(f'Collected {len(test_samples)} test samples')

    # Load Stage 1 model (needed by Stage2Dataset)
    stage1_model = Stage1Model()
    if os.path.exists(stage1_ckpt):
        state = torch.load(stage1_ckpt, map_location='cpu')
        model_state = state.get('model_state', state.get('model', state))
        stage1_model.load_state_dict(model_state)
        print(f'Loaded Stage 1 model from {stage1_ckpt}')
    stage1_model.eval().to(device)

    # Create dataset
    test_dataset = Stage2Dataset(
        test_samples, kg_builder, stage1_model,
        precomputed_kg_feats=precomputed_kg_feats,
    )
    print(f'  Test dataset: {len(test_dataset)} samples')

    # Build dataloader
    dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=_collate_fn,
        num_workers=0,
    )

    # Inference loop (same as validate() in train_stage2.py)
    all_cls_preds = []
    all_cls_labels = []
    all_reg_preds = []
    all_reg_labels = []

    print(f'\nRunning Stage 2 inference (batch_size={batch_size})...')
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            label_mask = batch['label_mask'].to(device)
            kg_feat = batch['kg_feat'].to(device)
            delays = batch['delay'].to(device)
            is_delayed = batch['is_delayed'].to(device)

            inputs_embeds = modified_embedding(input_ids, kg_feat=kg_feat)
            outputs = qwen_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            all_hidden = outputs.hidden_states[-1].float()
            instr_mask = (1 - label_mask).float().unsqueeze(-1)
            masked_hidden = all_hidden * instr_mask
            instr_count = instr_mask.sum(dim=1).clamp(min=1)
            llm_pooled = llm_pool(masked_hidden.sum(dim=1) / instr_count)

            kg_pooled = kg_pool(kg_feat)

            alpha = torch.sigmoid(fusion_gate.alpha)
            fused = alpha * llm_pooled + (1 - alpha) * kg_pooled

            cls_logits = cls_head(fused).squeeze(-1)
            reg_pred_norm = reg_head(fused).squeeze(-1)

            reg_pred = reg_pred_norm * DELAY_STD + DELAY_MEAN

            all_cls_preds.append(torch.sigmoid(cls_logits).cpu())
            all_cls_labels.append(is_delayed.cpu())
            all_reg_preds.append(reg_pred.cpu())
            all_reg_labels.append(delays.cpu())

    # Compute metrics
    all_cls_preds = torch.cat(all_cls_preds, dim=0)
    all_cls_labels = torch.cat(all_cls_labels, dim=0)
    all_reg_preds = torch.cat(all_reg_preds, dim=0)
    all_reg_labels = torch.cat(all_reg_labels, dim=0)

    cls_metrics = compute_cls_metrics(all_cls_preds, all_cls_labels)
    reg_metrics = compute_reg_metrics(all_reg_preds, all_reg_labels)

    # Find best classification threshold
    from sklearn.metrics import f1_score as sk_f1
    cls_probs = all_cls_preds.numpy().flatten()
    cls_labels = all_cls_labels.numpy().flatten()

    best_thresh = 0.5
    best_f1 = 0.0
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (cls_probs >= t).astype(int)
        f1_t = sk_f1(cls_labels, preds, zero_division=0)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thresh = t

    alpha_val = torch.sigmoid(fusion_gate.alpha).item()

    print('\n' + '=' * 70)
    print('  STAGE 2 LLM TEST RESULTS (Batched Forward Pass)')
    print('=' * 70)
    print(f"  Classification: AUC={cls_metrics['auc']:.4f}, "
          f"F1={cls_metrics['f1']:.4f} (@0.5), Best F1={best_f1:.4f} (@{best_thresh:.2f}), "
          f"Accuracy={cls_metrics['accuracy']:.4f}")
    print(f"  Classification: Precision={cls_metrics['precision']:.4f}, "
          f"Recall={cls_metrics['recall']:.4f}")
    print(f"  Regression: MAE={reg_metrics['mae']:.2f}min, "
          f"RMSE={reg_metrics['rmse']:.2f}min, R²={reg_metrics['r2']:.4f}")
    print(f"  Fusion: alpha={alpha_val:.4f} (LLM={alpha_val:.2f}, KG={1-alpha_val:.2f})")
    print(f"{'='*70}")

    results_data = []
    for i in range(len(all_cls_preds)):
        results_data.append({
            'true_delay': all_reg_labels[i].item(),
            'pred_delay': all_reg_preds[i].item(),
            'true_cls': int(all_cls_labels[i].item()),
            'pred_cls': int((cls_probs[i] >= best_thresh).astype(int)),
            'pred_prob': cls_probs[i],
        })

    results_df = pd.DataFrame(results_data)
    save_path = os.path.join(CONFIG.paths.output_dir, 'stage2_test_results.csv')
    results_df.to_csv(save_path, index=False)
    print(f'\nResults saved to {save_path}')

    return {
        'cls_metrics': cls_metrics,
        'reg_metrics': reg_metrics,
        'best_f1': best_f1,
        'best_threshold': best_thresh,
        'results_df': results_df,
    }


def _collate_fn(batch):
    """Collate function matching train_stage2's _collate_fn."""
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    label_masks = [item['label_mask'] for item in batch]
    kg_feats = [item['kg_feat'] for item in batch]
    delays = [item['delay'] for item in batch]
    is_delayed = [item['is_delayed'] for item in batch]

    # Pad sequences
    max_len = max(len(x) for x in input_ids)
    attention_mask = torch.stack([
        torch.cat([torch.ones(len(x), dtype=torch.long),
                   torch.zeros(max_len - len(x), dtype=torch.long)])
        for x in input_ids
    ])
    input_ids = torch.stack([
        torch.cat([x, torch.full((max_len - len(x),), 151643, dtype=torch.long)])
        for x in input_ids
    ])
    labels = torch.stack([
        torch.cat([x, torch.full((max_len - len(x),), -100, dtype=torch.long)])
        for x in labels
    ])
    label_mask = torch.stack([
        torch.cat([x, torch.ones(max_len - len(x), dtype=torch.long)])
        for x in label_masks
    ])
    kg_feat = torch.stack([
        torch.tensor(f, dtype=torch.float) if isinstance(f, np.ndarray) else f
        for f in kg_feats
    ])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'label_mask': label_mask,
        'kg_feat': kg_feat,
        'delay': torch.tensor(delays, dtype=torch.float),
        'is_delayed': torch.tensor(is_delayed, dtype=torch.long),
    }


if __name__ == '__main__':
    import sys
    import os
    # Add project root directory to Python path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.utils.config import CONFIG
    # Set output directory to Output/src
    CONFIG.paths.output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Output', 'src')
    test_stage2()
