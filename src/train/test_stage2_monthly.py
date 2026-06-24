"""
Stage 2: Test script - Output metrics for the 15th day of each month in 2025
"""
import os
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
                # Cache keys are simple integer indices (0, 1, 2, ...)
                # Build lookup: (year, month, day, idx) -> feature
                for idx, feat in day_feats.items():
                    all_feats[(year, month, day, idx)] = torch.tensor(
                        feat, dtype=torch.float32
                    )

    print(f"  Loaded {len(all_feats)} KG features")
    return all_feats


def test_stage2_monthly(stage1_ckpt: str = None, stage2_dir: str = None, batch_size: int = 4):
    """Run Stage 2 evaluation with monthly breakdown."""
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

    # Recreate components
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

    # Build test dataset
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

    # Load Stage 1 model
    stage1_model = Stage1Model()
    if os.path.exists(stage1_ckpt):
        state = torch.load(stage1_ckpt, map_location='cpu')
        model_state = state.get('model_state', state.get('model', state))
        stage1_model.load_state_dict(model_state)
        print(f'Loaded Stage 1 model from {stage1_ckpt}')
    stage1_model.eval().to(device)

    # Create dataset for all samples
    test_samples = collect_samples_for_days(test_days, max_samples_per_day=None)
    print(f'Collected {len(test_samples)} test samples')

    # collect_samples_for_days already sets sample['month'] as integer
    # Build month lookup from test_samples for later grouping
    sample_months = [s['month'] for s in test_samples]

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
    
    # Build month lookup from test_samples
    sample_months = [s['month'] for s in test_samples]

    # Collect all predictions with month info
    all_months = []
    all_cls_preds = []
    all_cls_labels = []
    all_reg_preds = []
    all_reg_labels = []

    print(f'\nRunning Stage 2 inference (batch_size={batch_size})...')
    batch_start = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Testing")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            label_mask = batch['label_mask'].to(device)
            kg_feat = batch['kg_feat'].to(device)
            delays = batch['delay'].to(device)
            is_delayed = batch['is_delayed'].to(device)
            
            # Get months for this batch
            batch_size_actual = input_ids.size(0)
            batch_months = sample_months[batch_start:batch_start + batch_size_actual]
            all_months.extend(batch_months)
            batch_start += batch_size_actual

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

    # Concatenate all predictions
    all_cls_preds = torch.cat(all_cls_preds, dim=0)
    all_cls_labels = torch.cat(all_cls_labels, dim=0)
    all_reg_preds = torch.cat(all_reg_preds, dim=0)
    all_reg_labels = torch.cat(all_reg_labels, dim=0)

    # Find best threshold
    from sklearn.metrics import f1_score as sk_f1, precision_score, recall_score, accuracy_score
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

    # Overall metrics
    print('\n' + '=' * 100)
    print('  STAGE 2 TEST RESULTS - 2025 Monthly (15th of each month)')
    print('=' * 100)

    # Monthly breakdown
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    print(f"\n{'Month':<8} {'Samples':<8} {'AUC':<8} {'Acc':<8} {'Pre':<8} {'Rec':<8} {'F1':<8} {'MAE':<8} {'RMSE':<8} {'R²':<8}")
    print('-' * 100)

    monthly_results = []
    for month in range(1, 13):
        mask = np.array(all_months) == month
        if mask.sum() == 0:
            continue

        month_probs = cls_probs[mask]
        month_labels = cls_labels[mask]
        month_reg_preds = all_reg_preds.numpy()[mask]
        month_reg_labels = all_reg_labels.numpy()[mask]

        # Classification metrics
        month_cls_preds = (month_probs >= best_thresh).astype(int)
        month_auc = compute_cls_metrics(
            torch.tensor(month_probs),
            torch.tensor(month_labels)
        )['auc']
        month_acc = accuracy_score(month_labels, month_cls_preds)
        month_pre = precision_score(month_labels, month_cls_preds, zero_division=0)
        month_rec = recall_score(month_labels, month_cls_preds, zero_division=0)
        month_f1 = sk_f1(month_labels, month_cls_preds, zero_division=0)

        # Regression metrics
        month_mae = np.mean(np.abs(month_reg_preds - month_reg_labels))
        month_rmse = np.sqrt(np.mean((month_reg_preds - month_reg_labels) ** 2))
        ss_res = np.sum((month_reg_labels - month_reg_preds) ** 2)
        ss_tot = np.sum((month_reg_labels - np.mean(month_reg_labels)) ** 2)
        month_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        print(f"{month_names[month-1]:<8} {mask.sum():<8} {month_auc:<8.4f} {month_acc:<8.4f} {month_pre:<8.4f} {month_rec:<8.4f} {month_f1:<8.4f} {month_mae:<8.2f} {month_rmse:<8.2f} {month_r2:<8.4f}")

        monthly_results.append({
            'month': month,
            'month_name': month_names[month-1],
            'samples': int(mask.sum()),
            'auc': month_auc,
            'accuracy': month_acc,
            'precision': month_pre,
            'recall': month_rec,
            'f1': month_f1,
            'mae': month_mae,
            'rmse': month_rmse,
            'r2': month_r2
        })

    print('-' * 100)

    # Overall metrics
    overall_cls_preds = (cls_probs >= best_thresh).astype(int)
    overall_auc = compute_cls_metrics(all_cls_preds, all_cls_labels)['auc']
    overall_acc = accuracy_score(cls_labels, overall_cls_preds)
    overall_pre = precision_score(cls_labels, overall_cls_preds, zero_division=0)
    overall_rec = recall_score(cls_labels, overall_cls_preds, zero_division=0)
    overall_f1 = sk_f1(cls_labels, overall_cls_preds, zero_division=0)
    overall_mae = np.mean(np.abs(all_reg_preds.numpy() - all_reg_labels.numpy()))
    overall_rmse = np.sqrt(np.mean((all_reg_preds.numpy() - all_reg_labels.numpy()) ** 2))
    ss_res = np.sum((all_reg_labels.numpy() - all_reg_preds.numpy()) ** 2)
    ss_tot = np.sum((all_reg_labels.numpy() - np.mean(all_reg_labels.numpy())) ** 2)
    overall_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    print(f"{'OVERALL':<8} {len(cls_labels):<8} {overall_auc:<8.4f} {overall_acc:<8.4f} {overall_pre:<8.4f} {overall_rec:<8.4f} {overall_f1:<8.4f} {overall_mae:<8.2f} {overall_rmse:<8.2f} {overall_r2:<8.4f}")
    print('=' * 100)

    alpha_val = torch.sigmoid(fusion_gate.alpha).item()
    print(f"\nFusion: alpha={alpha_val:.4f} (LLM={alpha_val:.2f}, KG={1-alpha_val:.2f})")
    print(f"Best threshold: {best_thresh:.2f}")

    # Save results
    results_df = pd.DataFrame(monthly_results)
    save_path = os.path.join(CONFIG.paths.output_dir, 'stage2_monthly_results.csv')
    results_df.to_csv(save_path, index=False)
    print(f'\nMonthly results saved to {save_path}')

    return monthly_results


def _collate_fn(batch):
    """Collate function for Stage2Dataset."""
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
    test_stage2_monthly(batch_size=8)
