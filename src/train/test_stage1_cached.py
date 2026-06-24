"""
Test Stage 1 using pre-cached KG features (two-step approach).

Step 1: KG features are already cached in output/kg_features_cache/
Step 2: Load cached KG features → Stage 1 dual-task heads → predict

This is functionally equivalent to --stage 1 --eval, but separates
KG feature extraction from prediction to demonstrate the decoupled architecture.

Usage:
    python -m src.train.test_stage1_cached
"""
import os
import pickle
import torch
import numpy as np
import pandas as pd
import calendar
from tqdm import tqdm

from src.models.stage1 import Stage1Model
from src.utils.config import CONFIG
from src.utils.metrics import compute_cls_metrics, compute_reg_metrics


def load_cached_kg_features(year: int, month: int, day: int, cache_dir: str) -> dict:
    """Load pre-cached KG features for a specific day."""
    month_str = f"{month:02d}"
    day_str = f"{day:02d}"
    year_str = f"{year:04d}"
    
    cache_file = os.path.join(cache_dir, year_str, month_str, f"{day_str}.pkl")
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"KG features not found: {cache_file}")
    
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
    
    return data


def build_targets_from_df(tabular_df, device):
    """Build classification and regression targets from tabular data."""
    delays_raw = tabular_df[CONFIG.data.target_col].values.astype(float)
    labels = torch.tensor(
        (delays_raw >= CONFIG.data.delay_threshold).astype(float),
        dtype=torch.float32
    )
    delays_reg = torch.tensor(delays_raw, dtype=torch.float32)
    
    return labels.to(device), delays_reg.to(device), delays_raw


def load_tabular_data(year: int, month: int, day: int) -> pd.DataFrame:
    """Load tabular flight data for a specific day."""
    month_str = f"{month:02d}"
    day_str = f"{day:02d}"
    year_str = f"{year:04d}"
    
    fname = f"flight_with_weather_{year % 100:02d}{month_str}{day_str}.csv"
    fpath = os.path.join(CONFIG.paths.tabular_dir, year_str, month_str, fname)
    
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Tabular data not found: {fpath}")
    
    df = pd.read_csv(fpath)
    df = df.drop(
        columns=[c for c in CONFIG.data.forbidden_cols if c in df.columns],
        errors='ignore'
    )
    return df


def test_stage1_cached():
    """Test Stage 1 model using pre-cached KG features."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "=" * 70)
    print("STAGE 1 TEST: Two-Step Approach (Cached KG Features → Dual-Task Heads)")
    print("=" * 70)
    
    # Load Stage 1 model
    ckpt_path = os.path.join(CONFIG.paths.output_dir, "stage1_best.pt")
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Stage 1 checkpoint not found at {ckpt_path}")
        return
    
    model = Stage1Model()
    state = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state['model_state'] if 'model_state' in state else state)
    model.eval().to(device)
    print(f"✓ Stage 1 model loaded from: {ckpt_path}")
    
    # Load best threshold from normalizer
    best_threshold = 0.5
    normalizer_path = os.path.join(CONFIG.paths.output_dir, "normalizer.pkl")
    if os.path.exists(normalizer_path):
        import pickle as pkl
        with open(normalizer_path, 'rb') as f:
            stats = pkl.load(f)
        best_threshold = stats.get('best_threshold', 0.5)
        print(f"✓ Loaded best_threshold={best_threshold:.2f} from {normalizer_path}")
    
    # Cache directory
    cache_dir = os.path.join(CONFIG.paths.output_dir, 'kg_features_cache')
    print(f"✓ KG feature cache directory: {cache_dir}")
    
    # Select test days (2025, 15th of each month)
    test_days = []
    for month in range(1, 13):
        _, max_day = calendar.monthrange(2025, month)
        actual_day = min(15, max_day)
        test_days.append((month, actual_day))
    
    print(f"\nSelected {len(test_days)} test days (2025, 15th of each month):")
    for month, day in test_days:
        print(f"  2025/{month:02d}/{day:02d}")
    print()
    
    # Collect predictions
    all_cls_preds = []
    all_cls_labels = []
    all_reg_preds = []
    all_reg_labels = []
    
    with torch.no_grad():
        for month, day in tqdm(test_days, desc="Testing with cached KG features"):
            try:
                # Step 1: Load cached KG features
                cached_data = load_cached_kg_features(2025, month, day, cache_dir)
                kg_features_dict = cached_data['features']
                
                # Step 2: Load tabular data (for labels and flight info)
                tabular_df = load_tabular_data(2025, month, day)
                n_flights = len(tabular_df)
                
                # Build targets
                cls_labels, reg_labels, reg_labels_raw = build_targets_from_df(tabular_df, device)
                
                # Step 3: Convert KG features to tensor
                kg_feats_list = []
                for idx in range(n_flights):
                    if idx in kg_features_dict:
                        kg_feats_list.append(kg_features_dict[idx])
                    else:
                        # Fallback: zero features if not in cache
                        kg_feats_list.append(np.zeros(CONFIG.kg.gat_out_dim, dtype=np.float32))
                
                kg_feats = torch.tensor(
                    np.array(kg_feats_list),
                    dtype=torch.float32,
                    device=device
                )
                
                # Step 4: Pass KG features through feature_gate THEN dual-task heads
                # (skip KG encoder, use cached features directly)
                # IMPORTANT: Must apply feature_gate before pred_heads (same as forward)
                e_f_gated = model.feature_gate(kg_feats)
                cls_logits, reg_pred = model.pred_heads(e_f_gated)
                cls_logits = cls_logits.squeeze(-1)
                reg_pred = reg_pred.squeeze(-1)
                
                # Collect predictions
                cls_preds = torch.sigmoid(cls_logits).cpu().flatten()
                all_cls_preds.append(cls_preds)
                all_cls_labels.append(cls_labels[:n_flights].cpu().flatten())
                
                reg_preds_raw = reg_pred.cpu().numpy().flatten()  # Output raw minutes (no longer normalized)
                all_reg_preds.append(reg_preds_raw)
                all_reg_labels.append(reg_labels_raw[:n_flights].flatten())
                
                print(f"  ✓ 2025/{month:02d}/{day:02d}: {n_flights} flights processed")
                
            except FileNotFoundError as e:
                print(f"  ✗ 2025/{month:02d}/{day:02d}: SKIPPED ({e})")
                continue
            except Exception as e:
                print(f"  ✗ 2025/{month:02d}/{day:02d}: ERROR ({e})")
                import traceback
                traceback.print_exc()
                continue
    
    # Concatenate all predictions
    if not all_cls_preds:
        print("\nERROR: No predictions collected!")
        return
    
    all_cls_preds = torch.cat(all_cls_preds, dim=0)
    all_cls_labels = torch.cat(all_cls_labels, dim=0)
    all_reg_preds = np.concatenate(all_reg_preds, axis=0)
    all_reg_labels = np.concatenate(all_reg_labels, axis=0)
    
    # Print results
    print("\n" + "=" * 70)
    print("TEST RESULTS (Two-Step: Cached KG Features → Dual-Task Heads)")
    print("=" * 70)
    
    # Compute metrics at default threshold 0.5 and optimal threshold
    cls_metrics_05 = compute_cls_metrics(all_cls_preds, all_cls_labels, threshold=0.5)
    print(f"\nClassification (threshold=0.50):")
    print(f"  AUC:       {cls_metrics_05['auc']:.4f}")
    print(f"  AP:        {cls_metrics_05.get('ap', 0.0):.4f}")
    print(f"  Accuracy:  {cls_metrics_05['accuracy']:.4f}")
    print(f"  Precision: {cls_metrics_05['precision']:.4f}")
    print(f"  Recall:    {cls_metrics_05['recall']:.4f}")
    print(f"  F1-Score:  {cls_metrics_05['f1']:.4f}")

    cls_metrics_opt = compute_cls_metrics(all_cls_preds, all_cls_labels, threshold=best_threshold)
    print(f"\nClassification (threshold={best_threshold:.2f}, optimal from training):")
    print(f"  Accuracy:  {cls_metrics_opt['accuracy']:.4f}")
    print(f"  Precision: {cls_metrics_opt['precision']:.4f}")
    print(f"  Recall:    {cls_metrics_opt['recall']:.4f}")
    print(f"  F1-Score:  {cls_metrics_opt['f1']:.4f}")
    
    # Confusion matrix at threshold 0.5
    preds_05 = (all_cls_preds.numpy() >= 0.5).astype(int)
    labels_int = all_cls_labels.numpy().astype(int)
    tp = int(np.sum((preds_05 == 1) & (labels_int == 1)))
    fp = int(np.sum((preds_05 == 1) & (labels_int == 0)))
    fn = int(np.sum((preds_05 == 0) & (labels_int == 1)))
    tn = int(np.sum((preds_05 == 0) & (labels_int == 0)))
    print(f"\nConfusion Matrix (threshold=0.50):")
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    
    # Regression metrics
    reg_metrics = compute_reg_metrics(
        torch.tensor(all_reg_preds, dtype=torch.float32),
        torch.tensor(all_reg_labels, dtype=torch.float32),
    )
    print(f"\nRegression:")
    print(f"  MAE:       {reg_metrics['mae']:.4f} minutes")
    print(f"  RMSE:      {reg_metrics['rmse']:.4f} minutes")
    print(f"  R²:        {reg_metrics['r2']:.4f}")
    
    print(f"\nTotal samples evaluated: {len(all_cls_labels)}")
    print("=" * 70)
    
    return all_cls_preds, all_cls_labels, all_reg_preds, all_reg_labels


if __name__ == '__main__':
    test_stage1_cached()
