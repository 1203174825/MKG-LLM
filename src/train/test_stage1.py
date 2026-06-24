"""
Stage 1 evaluation script (standalone, v5).

Architecture:
  - Main GAT: Flight KG (8 relation types)
  - Chain GAT: Same-aircraft chain (preceded_by)
  - Network GAT: Airport + Flight heterogeneous graph (3 types, unified)
"""
import os
import torch
import numpy as np
import dgl
import calendar
from tqdm import tqdm
from accelerate import Accelerator

from src.models.stage1 import Stage1Model
from src.data.aeolus_dataset import AeolusDataLoader
from src.data.kg_builder import DailyKGBuilder
from src.utils.metrics import compute_cls_metrics, compute_reg_metrics
from src.utils.config import CONFIG


def build_targets(tabular_df, device, use_raw_reg=True):
    """Build classification and regression targets.

    Following official experiments:
    - Regression uses raw DEP_DELAY values (no normalization)
    - Classification uses 15-minute threshold
    """
    delays_raw = tabular_df[CONFIG.data.target_col].values.astype(float)
    labels = torch.tensor((delays_raw >= CONFIG.data.delay_threshold).astype(float), dtype=torch.float32)

    if use_raw_reg:
        delays_reg = torch.tensor(delays_raw, dtype=torch.float32)
    else:
        reg_max = CONFIG.stage1.reg_target_max
        delays_reg = torch.clamp(torch.tensor(delays_raw, dtype=torch.float32) / reg_max, 0.0, 1.0)

    return labels.to(device), delays_reg.to(device), delays_raw


def _select_test_days(year, months, target_day=15):
    """Select one day per month (15th or middle day)."""
    selected = []
    for month in months:
        _, max_day = calendar.monthrange(year, month)
        actual_day = min(target_day, max_day)
        selected.append((month, actual_day))
    return selected


def test_stage1():
    """Evaluate Stage 1 model on 2017 monthly test set (1 day/month = 12 days, v5)."""
    accelerator = Accelerator(mixed_precision="no")
    device = accelerator.device
    os.makedirs(CONFIG.paths.output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("EVALUATION on Test Set (v5: Network GAT with Airport+Flight hetero graph)")
    print("=" * 60)

    model = Stage1Model().to(device)
    ckpt = os.path.join(CONFIG.paths.output_dir, "stage1_best.pt")
    if not os.path.exists(ckpt):
        print(f"Error: Checkpoint not found at {ckpt}")
        return
    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"Loaded checkpoint from: {ckpt}")

    test_days = _select_test_days(2017, list(range(1, 13)), target_day=15)
    print(f"\nSelected {len(test_days)} test days:")
    for month, day in test_days:
        print(f"  2017/{month:02d}/{day:02d}")
    print()

    # Load normalization statistics
    kg_builder = DailyKGBuilder()
    import pickle
    normalizer_path = os.path.join(CONFIG.paths.output_dir, "normalizer.pkl")
    if os.path.exists(normalizer_path):
        with open(normalizer_path, 'rb') as f:
            normalizer_stats = pickle.load(f)
        kg_builder.feat_sum = normalizer_stats['feat_sum']
        kg_builder.feat_sq_sum = normalizer_stats['feat_sq_sum']
        kg_builder.feat_count = normalizer_stats['feat_count']
        print(f"Loaded normalizer stats from {normalizer_path}")
    else:
        print(f"WARNING: No normalizer stats found at {normalizer_path}, using raw features")

    # Collect all predictions and labels
    all_cls_preds = []
    all_cls_labels = []
    all_reg_preds_raw = []
    all_reg_labels_raw = []

    with torch.no_grad():
        model.eval()
        for month, target_day in tqdm(test_days, desc="Testing"):
            day_loader = AeolusDataLoader(year=2017, months=[month])
            for day_data in day_loader.iter_days():
                if day_data.day == target_day:
                    if day_data.tabular.empty:
                        continue

                    # Build KG graph (g_network is Airport+Flight heterogeneous graph)
                    g, time_enc, n_flights, g_chain, g_network, airport_flight_map = kg_builder.build(
                        year=day_data.year, month=day_data.month, day=day_data.day,
                        tabular_df=day_data.tabular, chain_data=day_data.chain,
                        network_graph=day_data.network,
                    )
                    g = g.to(device)
                    time_enc = time_enc.to(device)

                    feat = g.ndata["feat"].to(device)
                    target_nids = torch.arange(n_flights, device=device)

                    # Network heterogeneous graph setup
                    network_feat = None
                    network_edge_feat = None
                    airport_map = None
                    if g_network is not None:
                        g_network = g_network.to(device)
                        network_feat = g_network.ndata['feat'].to(device)
                        if g_network.num_edges() > 0:
                            network_edge_feat = g_network.edata.get('feat')
                        airport_map = {
                            'flight_node_offset': airport_flight_map.get('flight_node_offset', n_flights),
                            'origin_ap_ids': airport_flight_map['origin_ap_ids'].to(device),
                            'dest_ap_ids': airport_flight_map['dest_ap_ids'].to(device),
                        }

                    # Use unified forward() interface
                    with torch.no_grad():
                        result = model(
                            blocks=None,
                            g_main=g,
                            feat=feat,
                            etypes_list=None,
                            time_enc=time_enc,
                            target_idx=target_nids,
                            edge_feat_list=None,
                            g_chain=g_chain.to(device) if g_chain is not None else None,
                            chain_feat=feat[:n_flights],
                            g_network=g_network,
                            network_feat=network_feat,
                            network_edge_feat=network_edge_feat,
                            airport_flight_map=airport_map,
                            flight_nids=target_nids,
                        )
                        cls_logits = result["cls_logits"]
                        reg_pred = result["reg_pred"]

                    cls_labels, reg_labels, reg_labels_raw = build_targets(
                        day_data.tabular, device
                    )

                    cls_preds = torch.sigmoid(cls_logits).cpu().flatten()
                    all_cls_preds.append(cls_preds)
                    all_cls_labels.append(cls_labels[:n_flights].cpu().flatten())

                    reg_preds_raw = reg_pred.cpu().numpy().flatten()
                    all_reg_preds_raw.append(reg_preds_raw)
                    all_reg_labels_raw.append(reg_labels_raw[:n_flights].flatten())

    all_cls_preds = torch.cat(all_cls_preds, dim=0)
    all_cls_labels = torch.cat(all_cls_labels, dim=0)
    all_reg_preds = np.concatenate(all_reg_preds_raw, axis=0)
    all_reg_labels = np.concatenate(all_reg_labels_raw, axis=0)

    # Classification metrics
    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)

    cls_metrics = compute_cls_metrics(all_cls_preds, all_cls_labels)
    print(f"\nClassification (threshold=0.5):")
    print(f"  AUC:       {cls_metrics['auc']:.4f}")
    print(f"  AP:        {cls_metrics.get('ap', 0.0):.4f}")
    print(f"  Accuracy:  {cls_metrics['accuracy']:.4f}")
    print(f"  Precision: {cls_metrics['precision']:.4f}")
    print(f"  Recall:    {cls_metrics['recall']:.4f}")
    print(f"  F1-Score:  {cls_metrics['f1']:.4f}")

    # Confusion matrix at threshold 0.5
    preds_05 = (all_cls_preds.numpy() >= 0.5).astype(int)
    labels_int = all_cls_labels.numpy().astype(int)
    tp = int(np.sum((preds_05 == 1) & (labels_int == 1)))
    fp = int(np.sum((preds_05 == 1) & (labels_int == 0)))
    fn = int(np.sum((preds_05 == 0) & (labels_int == 1)))
    tn = int(np.sum((preds_05 == 0) & (labels_int == 0)))
    print(f"\nConfusion Matrix:")
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    # Optimal threshold
    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(labels_int, all_cls_preds.numpy())
        youden = tpr - fpr
        best_idx = np.argmax(youden)
        best_threshold = thresholds[best_idx]
        preds_opt = (all_cls_preds.numpy() >= best_threshold).astype(int)
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        print(f"\nClassification (optimal threshold={best_threshold:.4f}, Youden Index):")
        print(f"  Accuracy:  {accuracy_score(labels_int, preds_opt):.4f}")
        print(f"  Precision: {precision_score(labels_int, preds_opt, zero_division=0):.4f}")
        print(f"  Recall:    {recall_score(labels_int, preds_opt, zero_division=0):.4f}")
        print(f"  F1-Score:  {f1_score(labels_int, preds_opt, zero_division=0):.4f}")
        tp2 = int(np.sum((preds_opt == 1) & (labels_int == 1)))
        fp2 = int(np.sum((preds_opt == 1) & (labels_int == 0)))
        fn2 = int(np.sum((preds_opt == 0) & (labels_int == 1)))
        tn2 = int(np.sum((preds_opt == 0) & (labels_int == 0)))
        print(f"\nConfusion Matrix:")
        print(f"  TP={tp2}, FP={fp2}, FN={fn2}, TN={tn2}")
    except:
        pass

    # Regression metrics (in minutes)
    reg_metrics = compute_reg_metrics(
        torch.tensor(all_reg_preds, dtype=torch.float32),
        torch.tensor(all_reg_labels, dtype=torch.float32),
    )
    print(f"\nRegression:")
    print(f"  MAE:       {reg_metrics['mae']:.4f} minutes")
    print(f"  RMSE:      {reg_metrics['rmse']:.4f} minutes")
    print(f"  R²:        {reg_metrics['r2']:.4f}")

    print(f"\nTotal samples evaluated: {len(all_cls_labels)}")
    print("=" * 50)
    return model


if __name__ == "__main__":
    import sys
    import os
    # Add project root directory to Python path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    test_stage1()
