import os
import sys
import torch
import numpy as np
import pandas as pd
from src.utils.config import CONFIG
from src.data.kg_builder import DailyKGBuilder
from src.data.aeolus_dataset import TabularDataset
from src.models.stage1 import Stage1Model
from src.models.stage2 import KGAlignmentLayer
from src.train.train_stage2 import Stage2Trainer, Stage2Dataset, load_kg_cache, collect_samples_for_days
from accelerate import Accelerator


def trace_single_flight(trainer, dataset, sample_idx: int, verbose: bool = True):
    """Trace complete inference for a single flight."""
    sample = dataset.samples[sample_idx]
    if verbose:
        print(f"\n{'='*80}")
        print(f"FLIGHT TRACE: {sample['date']}")
        print(f"{'='*80}")
        flight = sample['flight_data']
        print(f"  Flight: {flight.get('OP_CARRIER_FL_NUM', 'N/A')}")
        print(f"  Route: {flight.get('ORIGIN_INDEX', '?')} -> {flight.get('DEST_INDEX', '?')}")
        print(f"  Scheduled Departure: {int(flight.get('CRS_DEP_TIME_MIN', 480))//60:02d}:{int(flight.get('CRS_DEP_TIME_MIN', 480))%60:02d}")
        print(f"  Actual Delay: {sample['delay']:.1f} min")
        print(f"  Is Delayed (>=15min): {bool(sample['is_delayed'])}")
    
    # Step 1: Get KG features
    kg_feat_raw = dataset[sample_idx]['kg_feat']
    if isinstance(kg_feat_raw, np.ndarray):
        kg_feat_raw = torch.from_numpy(kg_feat_raw)
    kg_feat = kg_feat_raw.unsqueeze(0)
    if verbose:
        print(f"\n  [Step 1] KG Features (from Stage 1 encoder):")
        print(f"    Shape: {kg_feat.shape}")
        print(f"    Range: [{kg_feat.min():.4f}, {kg_feat.max():.4f}]")
        print(f"    Mean: {kg_feat.mean():.4f}, Std: {kg_feat.std():.4f}")
        print(f"    Sample values (first 10): {kg_feat[0, :10].tolist()}")
    
    # Step 2: KG-LLM Alignment
    trainer.aligner.eval()
    with torch.no_grad():
        aligned = trainer.aligner(kg_feat.to(trainer.device))
    if verbose:
        print(f"\n  [Step 2] KG-LLM Alignment:")
        print(f"    Input shape: {kg_feat.shape}")
        print(f"    Output shape: {aligned.shape}")
        print(f"    Aligned range: [{aligned.min():.4f}, {aligned.max():.4f}]")
    
    # Step 3: LLM Forward (with KG features injected)
    trainer.qwen_model.eval()
    trainer.modified_embedding.eval()
    dataset_item = dataset[sample_idx]
    input_ids = dataset_item['input_ids'].unsqueeze(0).to(trainer.device)
    label_mask = dataset_item['label_mask'].unsqueeze(0).to(trainer.device)
    with torch.no_grad():
        inputs_embeds = trainer.modified_embedding(input_ids, kg_feat=kg_feat.to(trainer.device))
        outputs = trainer.qwen_model(
            inputs_embeds=inputs_embeds,
            labels=input_ids,
            output_hidden_states=True,
        )
    if verbose:
        print(f"\n  [Step 3] LLM Forward Pass:")
        print(f"    Input IDs shape: {input_ids.shape}")
        print(f"    Inputs embeds shape: {inputs_embeds.shape}")
        print(f"    Hidden states (last layer) shape: {outputs.hidden_states[-1].shape}")
        print(f"    LM Loss: {outputs.loss.item():.4f}")
    
    # Step 4: Cross-Attention Fusion + Per-Dimension Gate (improved)
    all_hidden = outputs.hidden_states[-1].float()
    kg_tokens = trainer.aligner(kg_feat.to(trainer.device))  # (1, 5, 1536) — aligned KG embeddings
    batch_size = kg_tokens.shape[0]
    with torch.no_grad():
        kg_pooled = trainer.kg_pool(kg_tokens.reshape(batch_size, -1))
        
        # Pool instruction-only hidden states (P0 fix: exclude response tokens)
        instr_mask = (1 - label_mask).float().unsqueeze(-1)
        masked_hidden = all_hidden * instr_mask
        instr_count = instr_mask.sum(dim=1).clamp(min=1)
        llm_pooled = trainer.llm_pool(masked_hidden.sum(dim=1) / instr_count)
        

        llm_q = trainer.llm_query(llm_pooled)
        kg_k = trainer.kg_key(kg_tokens)
        kg_v = trainer.kg_value(kg_tokens)
        attn_scores = torch.bmm(llm_q.unsqueeze(1), kg_k.transpose(1, 2))
        attn_weights = torch.softmax(attn_scores / (512 ** 0.5), dim=-1)
        kg_attended = torch.bmm(attn_weights, kg_v).squeeze(1)
        

        gate_input = torch.cat([kg_pooled, llm_pooled], dim=-1)
        gate_vec = trainer.gate_layer(gate_input)
        gate_mean = gate_vec.mean().item()
        

        fused = gate_vec * llm_pooled + (1 - gate_vec) * kg_attended
    if verbose:
        print(f"\n  [Step 4] Cross-Attention Fusion + Per-Dimension Gate:")
        print(f"    KG pooled range: [{kg_pooled.min():.4f}, {kg_pooled.max():.4f}]")
        print(f"    LLM pooled range: [{llm_pooled.min():.4f}, {llm_pooled.max():.4f}]")
        print(f"    Cross-Attention weights (5 KG tokens): {attn_weights.squeeze(0).tolist()}")
        print(f"    Per-Dimension Gate mean: {gate_mean:.4f} (<0.5 = more KG, >0.5 = more LLM)")
        print(f"    Gate range: [{gate_vec.min():.4f}, {gate_vec.max():.4f}]")
        print(f"    Fused range: [{fused.min():.4f}, {fused.max():.4f}]")
    
    # Step 5: Dual-Task Predictions
    with torch.no_grad():
        projected = trainer.task_projector(fused)
        cls_logits = trainer.cls_head(projected).squeeze(-1)
        reg_pred_norm = trainer.reg_head(projected).squeeze(-1)
        reg_pred = reg_pred_norm * 23.6 + 5.9
        cls_prob = torch.sigmoid(cls_logits)
        is_delayed_pred = bool(cls_prob.item() >= 0.5)
    if verbose:
        print(f"\n  [Step 5] Dual-Task Predictions:")
        print(f"    Classification Logit: {cls_logits.item():.4f}")
        print(f"    Classification Probability: {cls_prob.item():.4f}")
        print(f"    Predicted Delayed (>=0.5): {is_delayed_pred}")
        print(f"    Regression (normalized): {reg_pred_norm.item():.4f}")
        print(f"    Predicted Delay: {reg_pred.item():.1f} min")
        print(f"    Actual Delay: {sample['delay']:.1f} min")
        print(f"    Actual Delayed: {bool(sample['is_delayed'])}")
        print(f"    Prediction CORRECT: {is_delayed_pred == bool(sample['is_delayed'])}")
    
    return {
        'kg_feat': kg_feat,
        'aligned': aligned,
        'gate_vec': gate_vec,
        'attn_weights': attn_weights,
        'kg_pooled': kg_pooled,
        'llm_pooled': llm_pooled,
        'fused': fused,
        'cls_prob': cls_prob,
        'reg_pred': reg_pred,
        'is_delayed_pred': is_delayed_pred,
        'actual_delay': sample['delay'],
        'actual_is_delayed': bool(sample['is_delayed']),
    }


def analyze_chain_leakage(kg_builder, tabular_df, sample_idx):
    """Analyze how much delay information leaks through the chain graph."""
    print(f"\n  [Chain Graph Analysis] for flight index {sample_idx}:")
    
    # Find the tail number
    row = tabular_df.iloc[sample_idx]
    tail_num = str(row.get('TAIL_NUM', ''))
    print(f"    Aircraft: {tail_num}")
    print(f"    Flight Number: {row.get('OP_CARRIER_FL_NUM', 'N/A')}")
    
    # Find preceding flights
    tabular_tmp = tabular_df.reset_index(drop=True).copy()
    tabular_tmp['_dep'] = pd.to_numeric(tabular_tmp['CRS_DEP_TIME_MIN'], errors='coerce').fillna(0)
    tabular_tmp['_arr'] = pd.to_numeric(tabular_tmp['CRS_ARR_TIME_MIN'], errors='coerce').fillna(0)
    
    grp = tabular_tmp[tabular_tmp['TAIL_NUM'] == tail_num].sort_values('_dep')
    if len(grp) == 0:
        print("    No flights found for this aircraft")
        return
    
    current_idx_in_grp = grp[grp.index == sample_idx]
    if len(current_idx_in_grp) == 0:
        print("    This flight not found in aircraft group")
        return
    
    current_pos = list(grp.index).index(sample_idx)
    
    if current_pos == 0:
        print("    This is the FIRST flight for this aircraft today - no preceding flight")
        print("    No chain-based label leakage possible")
    else:
        prev_row = grp.iloc[current_pos - 1]
        prev_dep_delay = prev_row.get('DEP_DELAY', 0)
        prev_arr_delay = prev_row.get('ARR_DELAY', 0)
        prev_dep_delay = np.nan_to_num(prev_dep_delay, nan=0.0)
        prev_arr_delay = np.nan_to_num(prev_arr_delay, nan=0.0)
        
        print(f"\n    PRECEDING FLIGHT:")
        print(f"      Flight: {prev_row.get('OP_CARRIER_FL_NUM', 'N/A')}")
        print(f"      Route: {prev_row.get('ORIGIN_INDEX', '?')} -> {prev_row.get('DEST_INDEX', '?')}")
        print(f"      Actual DEP_DELAY: {prev_dep_delay:.1f} min")
        print(f"      Actual ARR_DELAY: {prev_arr_delay:.1f} min")
        print(f"      Time Gap to Current: {prev_row.get('_arr', 0)} -> {row.get('_dep', 0)} = {row.get('_dep', 0) - prev_row.get('_arr', 0):.0f} min")
        
        # Check how this flows into chain features
        dep_norm = max(min(prev_dep_delay / 120.0, 1.0), -1.0)
        arr_norm = max(min(prev_arr_delay / 120.0, 1.0), -1.0)
        
        print(f"\n    CHAIN EDGE FEATURES (normalized):")
        print(f"      dep_norm: {dep_norm:.4f} (actual: {prev_dep_delay:.1f})")
        print(f"      arr_norm: {arr_norm:.4f} (actual: {prev_arr_delay:.1f})")
        
        print(f"\n    ⚠️ LABEL LEAKAGE RISK: HIGH")
        print(f"    The current flight's KG features contain normalized actual delay values")
        print(f"    from its preceding flight. If the preceding flight was delayed, this")
        print(f"    information strongly correlates with the current flight being delayed.")


def main():
    """Main tracing function."""
    # Initialize
    accelerator = Accelerator()
    device = accelerator.device
    
    # Load Stage 1 checkpoint
    stage1_ckpt = os.path.join(CONFIG.paths.output_dir, 'stage1_best.pt')
    if not os.path.exists(stage1_ckpt):
        print(f"ERROR: Stage 1 checkpoint not found at {stage1_ckpt}")
        return
    
    print("Loading Stage 2 trainer...")
    trainer = Stage2Trainer(stage1_ckpt=stage1_ckpt, accelerator=accelerator)
    
    # Initialize KG builder
    kg_builder = DailyKGBuilder()
    
    # Select test day
    test_days = [(2017, 1, 15)]  # 2017/01/15
    print(f"\nCollecting test samples for {test_days[0][0]}/{test_days[0][1]:02d}/{test_days[0][2]:02d}...")
    test_samples = collect_samples_for_days(test_days, max_samples_per_day=1000)
    print(f"Collected {len(test_samples)} test samples")
    
    # Load KG features
    train_cache_dir = os.path.join(CONFIG.paths.output_dir, 'kg_features_cache')
    test_kg_feats = load_kg_cache(train_cache_dir, [2017], list(range(1, 13)))
    
    stage1_model = trainer.get_stage1_model()
    test_dataset = Stage2Dataset(test_samples, kg_builder, stage1_model, 
                                 precomputed_kg_feats=test_kg_feats)
    
    # Find representative flights
    delayed_indices = [i for i, s in enumerate(test_samples) if s['is_delayed']]
    on_time_indices = [i for i, s in enumerate(test_samples) if not s['is_delayed']]
    
    # Select samples
    sample_indices = []
    if len(on_time_indices) > 0:
        sample_indices.append(on_time_indices[0])  # First on-time flight
    if len(delayed_indices) > 0:
        sample_indices.append(delayed_indices[0])  # First delayed flight
    if len(delayed_indices) > 5:
        sample_indices.append(delayed_indices[5])  # Another delayed flight
    
    print(f"\nTracing {len(sample_indices)} representative flights...")
    
    for idx in sample_indices:
        result = trace_single_flight(trainer, test_dataset, idx)
        
        # Analyze chain leakage
        tabular_dir = CONFIG.paths.tabular_dir
        year, month, day = test_samples[idx]['year'], test_samples[idx]['month'], test_samples[idx]['day']
        fname = f"flight_with_weather_{year % 100:02d}{month:02d}{day:02d}.csv"
        fpath = os.path.join(tabular_dir, str(year), f"{month:02d}", fname)
        if os.path.exists(fpath):
            tabular_df = pd.read_csv(fpath)
            analyze_chain_leakage(kg_builder, tabular_df, test_samples[idx]['idx'])
        else:
            print(f"\n  [Chain Graph Analysis] SKIP: tabular data not found")
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nKey Findings:")
    print(f"  1. Check gate values: if gate < 0.3, KG features dominate prediction")
    print(f"  2. Check chain graph: preceding flight delays leak into current flight features")
    print(f"  3. Compare KG pooled vs LLM pooled: which contributes more to final prediction?")
    print(f"  4. Verify normalization consistency between training and inference")


if __name__ == "__main__":
    main()
