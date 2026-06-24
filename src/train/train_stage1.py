"""
Stage 1 training loop (v7 — Shared GAT + Adapter-Delta + Learnable Fusion).

Architecture:
  - Main KG (25d): Flight graph (8 relation types) — ops + weather + geo
  - Chain KG (25d): Same-aircraft chain (preceded_by) — chain propagation
  - Network KG (25d): Airport + Flight heterogeneous graph (3 types) — congestion
  - Shared GAT with 3 Adapter-Delta modules (25→8→25) providing modality-specific projections
  - Learnable Fusion (3 scalar weights, softmax-normalized) + FeatureGate + PredictionHeads

Trained on 2024 daily KG snapshots (6 days/month), validated on 2024 quarterly (4 days).
"""
import os, json, time, gc, pickle
from datetime import datetime
import random
import numpy as np
import torch
import dgl
from tqdm import tqdm
from accelerate import Accelerator

from src.models.stage1 import Stage1Model
from src.data.aeolus_dataset import AeolusDataLoader
from src.data.kg_builder import DailyKGBuilder
from src.data.dgl_sampler import (
    create_dataloader, extract_flight_node_ids, extract_edge_types
)
from src.utils.metrics import compute_cls_metrics, compute_reg_metrics
from src.utils.config import CONFIG


def set_deterministic(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_targets(tabular_df, device, use_raw_reg=False):
    """Build classification and regression targets from tabular data.

    Following official experiments:
    - Regression uses raw DEP_DELAY values (no normalization)
    - Classification uses 15-minute threshold
    """
    delays_raw = tabular_df[CONFIG.data.target_col].values.astype(float)
    labels = torch.tensor((delays_raw >= CONFIG.data.delay_threshold).astype(float), dtype=torch.float32)

    if use_raw_reg:
        # Option A: Directly predict raw values (following official Regressor experiment)
        delays_reg = torch.tensor(delays_raw, dtype=torch.float32)
    else:
        # Option B: Normalize to [0,1] (old approach)
        reg_max = CONFIG.stage1.reg_target_max
        delays_reg = torch.clamp(torch.tensor(delays_raw, dtype=torch.float32) / reg_max, 0.0, 1.0)

    return labels.to(device), delays_reg.to(device), delays_raw


def denormalize_reg(pred_norm, reg_max):
    """Convert normalized regression predictions back to minutes."""
    return pred_norm * reg_max


def build_tabular_features(tabular_df, kg_builder, device):
    """Build 25-dim feature tensor from tabular data matching kg_builder's flight node feat.

    Returns (feat_tensor, n_flights) where feat_tensor shape = (N, 25).
    """
    import numpy as np
    n_flights = len(tabular_df)
    cont_cols = CONFIG.data.cont_cols
    n_cont = len(cont_cols)
    
    flight_feat = np.zeros((n_flights, n_cont), dtype=np.float32)
    for i, col in enumerate(cont_cols):
        if col in tabular_df.columns:
            vals = tabular_df[col].values.astype(np.float32)
            vals = np.nan_to_num(vals, nan=0.0)
            flight_feat[:, i] = vals
    
    # Normalize using the same global stats (same as _build_flight_node_feat)
    if kg_builder.feat_sum is not None and kg_builder.feat_count > 0:
        mean = kg_builder.feat_sum / kg_builder.feat_count
        var = (kg_builder.feat_sq_sum / kg_builder.feat_count) - (mean ** 2)
        var = np.maximum(var, 1e-8)
        std = np.sqrt(var)
        flight_feat = (flight_feat - mean) / std
    
    # Pad 5 zeros for airport feature slots (matching g.ndata['feat'][flight_id] structure)
    combined = np.concatenate([flight_feat, np.zeros((n_flights, 5), dtype=np.float32)], axis=1)
    return torch.tensor(combined, dtype=torch.float32, device=device), n_flights


def subsample_flights(tabular_df, year, month, day, n_samples=2000):
    """Unified sampling function: use fixed random seed to ensure consistent sampling between Stage 1 and Stage 2.

    Args:
        tabular_df: Raw data DataFrame
        year: Year
        month: Month
        day: Day
        n_samples: Number of samples (default 2000)

    Returns:
        Sampled DataFrame
    """
    if len(tabular_df) <= n_samples:
        return tabular_df
    
    # Use fixed seed: 42 + year*10000 + month*100 + day
    # Each date has a unique seed, and Stage 1 and Stage 2 use the same seed
    seed = 42 + year * 10000 + month * 100 + day
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(tabular_df), n_samples, replace=False)
    return tabular_df.iloc[indices].reset_index(drop=True)


def train_mlp_epoch(model, optimizer, accelerator, features,
                    cls_target, reg_target, device, batch_size=128):
    """MLP-only training loop: no DGL, no graphs."""
    model.train()
    n = features.shape[0]
    indices = torch.randperm(n, device=device)
    total_loss = 0.0
    n_batches = 0
    
    for i in range(0, n, batch_size):
        batch_idx = indices[i:i+batch_size]
        feat_b = features[batch_idx]
        cls_b = cls_target[batch_idx]
        reg_b = reg_target[batch_idx]
        
        cls_logits, reg_pred = model.forward_tabular(feat_b)
        loss, details = model.compute_loss(cls_logits, reg_pred, cls_b, reg_b)
        
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += details['loss']
        n_batches += 1
        
        if n_batches % 10 == 0:
            print(f"  batch {n_batches}: loss={details['loss']:.4f}, alpha=[{details['alpha_cls']:.3f}, {details['alpha_reg']:.3f}]")
    
    return total_loss / max(n_batches, 1)


def train_epoch(model, optimizer, accelerator, dataloader,
                all_labels, all_delays, time_enc, device,
                node_features=None,
                g_chain=None, chain_feat=None,
                g_network=None, network_feat=None, network_edge_feat=None,
                airport_flight_map=None):
    """Single epoch training loop (v5)."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        blocks = [b.to(device) for b in blocks]

        if node_features is not None:
            src_nids = blocks[0].srcdata[dgl.NID].to(device)
            feat = node_features[src_nids]
        else:
            feat = blocks[0].srcdata.get("feat",
                torch.zeros(input_nodes.shape[0], 64))
            feat = feat.to(device)

        # Move graphs to device
        g_chain_b = g_chain.to(device) if g_chain is not None else None
        g_network_b = g_network.to(device) if g_network is not None else None
        network_feat_b = network_feat.to(device) if network_feat is not None else None
        network_edge_feat_b = network_edge_feat.to(device) if network_edge_feat is not None else None

        airport_map_b = None
        if airport_flight_map is not None:
            airport_map_b = {
                'flight_node_offset': airport_flight_map.get('flight_node_offset', 0),
                'origin_ap_ids': airport_flight_map['origin_ap_ids'].to(device),
                'dest_ap_ids': airport_flight_map['dest_ap_ids'].to(device),
            }

        etypes_list = []
        edge_feat_list = []
        for blk in blocks:
            if dgl.ETYPE in blk.edata:
                etypes_list.append(blk.edata[dgl.ETYPE])
            else:
                etypes_list.append(torch.zeros(blk.num_edges(), dtype=torch.long, device=device))
            if "feat" in blk.edata:
                edge_feat_list.append(blk.edata["feat"].to(device))
            else:
                edge_feat_list.append(None)

        device_nodes = output_nodes.to(device)
        max_idx = max(all_labels.shape[0] - 1, 0)
        valid_mask = device_nodes <= max_idx
        if not valid_mask.all():
            tqdm.write(f"  batch {step}: skipped {int((~valid_mask).sum())} nodes with invalid IDs")
            valid_device_nodes = device_nodes[valid_mask]
            if len(valid_device_nodes) == 0:
                continue
        else:
            valid_device_nodes = device_nodes

        cls_target = all_labels[valid_device_nodes]
        reg_target = all_delays[valid_device_nodes]
        target_idx = torch.arange(len(valid_device_nodes), device=device)

        outputs = model(
            blocks=blocks, feat=feat,
            etypes_list=etypes_list,
            time_enc=time_enc.to(device),
            target_idx=target_idx,
            edge_feat_list=edge_feat_list,
            g_chain=g_chain_b, chain_feat=chain_feat,
            g_network=g_network_b,
            network_feat=network_feat_b,
            network_edge_feat=network_edge_feat_b,
            airport_flight_map=airport_map_b,
            flight_nids=valid_device_nodes,
        )

        loss, log_dict = model.compute_loss(
            cls_logits=outputs["cls_logits"],
            reg_pred=outputs["reg_pred"],
            cls_target=cls_target,
            reg_target=reg_target,
            batch_idx=step,
        )

        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(
                model.parameters(), CONFIG.stage1.gradient_clip
            )
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        n_batches += 1

        if step % 10 == 0:
            tqdm.write(
                f"  batch {step}: loss={loss.item():.4f}, "
                f"alpha=[{log_dict['alpha_cls']:.3f}, {log_dict['alpha_reg']:.3f}]"
            )

    return total_loss / max(n_batches, 1)


def validate(model, accelerator, device, kg_builder=None):
    """Validate by calling model.forward() directly (no manual unwrapping).
    
    Returns (weighted_loss, task_loss, best_val_f1, best_threshold) where:
      - weighted_loss = uncertainty-weighted loss (used for checkpoint selection)
      - task_loss = bce + huber (raw task loss, always >= 0, for reference only)
      - best_val_f1 = best F1 score on validation set
      - best_threshold = threshold that achieves best_val_f1
    """
    raw_model = accelerator.unwrap_model(model)
    raw_model.eval()
    total_weighted_loss = 0.0
    total_task_loss = 0.0
    n_days = 0
    all_cls_logits = []
    all_cls_labels = []

    use_mlp_only = getattr(raw_model, 'use_mlp_only', False)

    if kg_builder is None:
        kg_builder = DailyKGBuilder()

    with torch.no_grad():
        # Validation: 10th day of the 2nd month of each quarter in 2024 (Feb 10, May 10, Aug 10, Nov 10)
        for month in [2, 5, 8, 11]:
            day_loader = AeolusDataLoader(year=2024, months=[month])
            found = False
            for day_data in day_loader.iter_days():
                if day_data.day == 10:
                    if day_data.tabular.empty:
                        continue

                    # No sampling, use all flights

                    if use_mlp_only:
                        feat_tensor, n_flights = build_tabular_features(
                            day_data.tabular, kg_builder, device)
                        cls_labels, reg_target, _ = build_targets(day_data.tabular, device, use_raw_reg=True)
                        cls_logits, reg_pred = raw_model.forward_tabular(feat_tensor)
                        _, details = raw_model.compute_loss(cls_logits, reg_pred, cls_labels, reg_target)
                        total_weighted_loss += details['loss']
                        total_task_loss += details['bce'] + details['huber']
                        n_days += 1
                        all_cls_logits.append(torch.sigmoid(cls_logits).cpu().numpy().flatten())
                        all_cls_labels.append(cls_labels.cpu().numpy().flatten())
                        found = True
                        break

                    g, time_enc, n_flights, g_chain, g_network, airport_flight_map = kg_builder.build(
                        year=day_data.year, month=day_data.month, day=day_data.day,
                        tabular_df=day_data.tabular, chain_data=day_data.chain,
                        network_graph=day_data.network,
                    )
                    g = g.to(device)
                    time_enc = time_enc.to(device)

                    feat = g.ndata["feat"].to(device)
                    target_nids = torch.arange(n_flights, device=device)

                    g_chain_b = g_chain.to(device) if g_chain is not None else None
                    chain_feat_b = g_chain.ndata["feat"].to(device) if g_chain is not None else None

                    g_network_b = None
                    network_feat_b = None
                    network_edge_feat_b = None
                    airport_map_b = None
                    if g_network is not None:
                        g_network_b = g_network.to(device)
                        network_feat_b = g_network.ndata['feat'].to(device)
                        network_edge_feat_b = g_network.edata['feat'].to(device) if 'feat' in g_network.edata else None
                        airport_map_b = {
                            'flight_node_offset': airport_flight_map.get('flight_node_offset', n_flights),
                            'origin_ap_ids': airport_flight_map['origin_ap_ids'].to(device),
                            'dest_ap_ids': airport_flight_map['dest_ap_ids'].to(device),
                        }

                    outputs = raw_model(
                        blocks=None, feat=feat,
                        etypes_list=[], time_enc=time_enc,
                        target_idx=target_nids,
                        edge_feat_list=[],
                        g_main=g, g_chain=g_chain_b,
                        chain_feat=chain_feat_b,
                        g_network=g_network_b,
                        network_feat=network_feat_b,
                        network_edge_feat=network_edge_feat_b,
                        airport_flight_map=airport_map_b,
                        flight_nids=target_nids,
                    )

                    cls_labels, reg_target, _ = build_targets(day_data.tabular, device, use_raw_reg=True)
                    _, details = raw_model.compute_loss(
                        outputs["cls_logits"], outputs["reg_pred"],
                        cls_labels[:n_flights], reg_target[:n_flights],
                    )
                    total_weighted_loss += details['loss']
                    total_task_loss += details['bce'] + details['huber']
                    n_days += 1
                    all_cls_logits.append(torch.sigmoid(outputs["cls_logits"]).cpu().numpy().flatten()[:n_flights])
                    all_cls_labels.append(cls_labels[:n_flights].cpu().numpy().flatten())
                    found = True
                    break

            if not found:
                print(f"  Warning: 2025/{month}/10 not available")

    if n_days == 0:
        return 0.0, 0.0, 0.0, 0.5

    # Compute optimal classification threshold
    import numpy as np
    from sklearn.metrics import f1_score
    all_logits = np.concatenate(all_cls_logits, axis=0)
    all_labels = np.concatenate(all_cls_labels, axis=0)
    
    best_val_f1 = 0.0
    best_threshold = 0.5
    for thresh in np.arange(0.05, 0.96, 0.01):
        preds = (all_logits >= thresh).astype(int)
        f1 = f1_score(all_labels, preds, zero_division=0)
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_threshold = thresh

    return total_weighted_loss / n_days, total_task_loss / n_days, best_val_f1, best_threshold


def _select_training_days(data_loader, days_per_month=5):
    """Select multiple days per month from the data loader (uniformly spread)."""
    import calendar
    selected = []
    monthly_days = {}  # {month: [day_data]}

    # Collect all days grouped by month
    for day_data in data_loader.iter_days():
        month = day_data.month
        if month not in monthly_days:
            monthly_days[month] = []
        monthly_days[month].append(day_data)

    # Select days_per_month from each month (uniformly spread)
    for month in sorted(monthly_days.keys()):
        days = monthly_days[month]
        if len(days) <= days_per_month:
            selected.extend(days)
        else:
            # Uniformly select days
            step = len(days) // days_per_month
            for i in range(days_per_month):
                idx = min(i * step + step // 2, len(days) - 1)
                selected.append(days[idx])

    return selected


def train_stage1(model_kwargs=None):
    """Main training entry point for Stage 1.
    
    Trains the 3-modality fusion model on 2024 (72 days training, 4 days validation),
    tests on 2025 (12 days), then caches KG features for Stage 2.
    """
    set_deterministic(CONFIG.train.seed)
    accelerator = Accelerator(
        mixed_precision="fp16" if CONFIG.train.fp16 else "no",
    )
    device = accelerator.device
    os.makedirs(CONFIG.paths.output_dir, exist_ok=True)
    print(f"\nDevice: {device}")

    model_kwargs = model_kwargs or {}
    use_mean_fusion = model_kwargs.get('use_mean_fusion', False)
    skip_modalities = model_kwargs.get('skip_modalities', [])
    feature_mask_indices = model_kwargs.get('feature_mask_indices', None)
    use_mlp_only = model_kwargs.get('use_mlp_only', False)

    print("\n" + "=" * 60)
    print(f"Training data: 2024 (15th day/month x 12 months = 12 days, ALL flights)")
    print(f"Validation data: 2024 (Feb/May/Aug/Nov 10th = 4 days, ALL flights)")
    print(f"Test data: 2025 (15th day/month x 12 months = 12 days, ALL flights)")
    print(f"Epochs: {CONFIG.stage1.epochs}")
    print(f"LR: {CONFIG.stage1.lr}, Batch: {CONFIG.stage1.batch_size}")
    print(f"FP16: {CONFIG.train.fp16}, Reg weight: {CONFIG.stage1.reg_loss_weight}, Reg max: {CONFIG.stage1.reg_target_max}")
    if use_mlp_only:
        print(f"Mode: MLP-ONLY (no knowledge graph)")
    if feature_mask_indices:
        print(f"Feature mask indices: {feature_mask_indices}")
    print("=" * 60 + "\n")

    model = Stage1Model(
        skip_modalities=skip_modalities,
        use_mean_fusion=use_mean_fusion,
        feature_mask_indices=feature_mask_indices,
        use_mlp_only=use_mlp_only,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG.stage1.lr,
        weight_decay=CONFIG.stage1.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6,
    )

    model, optimizer = accelerator.prepare(model, optimizer)

    best_val_loss = float("inf")
    patience = 0

    log_path = os.path.join(CONFIG.paths.output_dir, "stage1_log.txt")
    log_lines = []
    log_lines.append("epoch,train_loss,val_loss,val_task_loss,lr")
    print(f"Logging metrics to {log_path}")

    data_loader = AeolusDataLoader(year=2024, months=CONFIG.data.train_months)
    kg_builder = DailyKGBuilder()

    # 15th day of each month in 2024 (training set: 12 days, use all flights)
    train_days_list = [15]
    training_days = []
    for day_data in data_loader.iter_days():
        if day_data.day in train_days_list:
            training_days.append(day_data)
    print(f"Selected {len(training_days)} training days:")
    for d in training_days:
        print(f"  {d.year}/{d.month:02d}/{d.day:02d}")
    print()

    # Fit global normalization statistics (using all training days)
    print("Fitting feature normalizer on training data...")
    kg_builder.fit_normalizer([d.tabular for d in training_days if not d.tabular.empty])
    print()

    for epoch in range(CONFIG.stage1.epochs):
        model.train()
        epoch_loss = 0.0
        n_days = 0

        for day_data in tqdm(training_days,
                             desc=f"Epoch {epoch+1}/{CONFIG.stage1.epochs}"):
            if day_data.tabular.empty:
                continue

            if use_mlp_only:
                # MLP-only: build features directly from tabular, no graph
                feat_tensor, n_flights = build_tabular_features(day_data.tabular, kg_builder, device)
                all_labels, all_delays, _ = build_targets(day_data.tabular, device, use_raw_reg=True)
                
                day_loss = train_mlp_epoch(
                    model, optimizer, accelerator,
                    feat_tensor, all_labels, all_delays, device,
                    batch_size=CONFIG.stage1.batch_size,
                )
                epoch_loss += day_loss
                n_days += 1
                continue

            g, time_enc, n_flights, g_chain, g_network, airport_flight_map = kg_builder.build(
                year=day_data.year, month=day_data.month, day=day_data.day,
                tabular_df=day_data.tabular, chain_data=day_data.chain,
                network_graph=day_data.network,
            )
            g = g.to(device)
            g_chain = g_chain.to(device) if g_chain is not None else None
            chain_feat = g_chain.ndata["feat"].to(device) if g_chain is not None else None

            # Network heterogeneous graph setup
            network_feat = None
            network_edge_feat = None
            airport_map = None
            if g_network is not None:
                g_network = g_network.to(device)
                network_feat = g_network.ndata['feat'].to(device)
                if g_network.num_edges() > 0:
                    network_edge_feat = g_network.edata['feat'].to(device)
                airport_map = {
                    'flight_node_offset': airport_flight_map.get('flight_node_offset', n_flights),
                    'origin_ap_ids': airport_flight_map['origin_ap_ids'],
                    'dest_ap_ids': airport_flight_map['dest_ap_ids'],
                }

            target_nids = torch.arange(n_flights)
            batch_loader = create_dataloader(
                g, target_nids, batch_size=CONFIG.stage1.batch_size,
            )

            all_labels, all_delays, _ = build_targets(day_data.tabular, device, use_raw_reg=True)

            day_loss = train_epoch(
                model, optimizer, accelerator,
                batch_loader, all_labels, all_delays, time_enc, device,
                node_features=g.ndata["feat"],
                g_chain=g_chain,
                chain_feat=chain_feat,
                g_network=g_network,
                network_feat=network_feat,
                network_edge_feat=network_edge_feat,
                airport_flight_map=airport_map,
            )
            epoch_loss += day_loss
            n_days += 1

        avg_loss = epoch_loss / max(n_days, 1)
        print(f"\nEpoch {epoch+1} avg loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")

        val_loss, val_task_loss, val_f1, val_thresh = validate(model, accelerator, device, kg_builder=kg_builder)
        print(f"Validation loss: {val_loss:.4f} (task_loss: {val_task_loss:.4f}, F1={val_f1:.4f}@{val_thresh:.2f})")

        lr_now = scheduler.get_last_lr()[0]
        log_lines.append(f"{epoch+1},{avg_loss:.4f},{val_loss:.4f},{val_task_loss:.4f},{val_f1:.4f},{val_thresh:.2f},{lr_now:.2e}")
        with open(log_path, 'w') as f:
            f.write("\n".join(log_lines))

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_threshold = val_thresh
            patience = 0
            ckpt = os.path.join(CONFIG.paths.output_dir, "stage1_best.pt")
            accelerator.save(accelerator.unwrap_model(model).state_dict(), ckpt)
            print(f"  Checkpoint saved to {ckpt} (val_loss: {val_loss:.4f}, task_loss: {val_task_loss:.4f}, best_thresh={val_thresh:.2f})")

            # Save model configuration (for ablation recovery)
            config_path = os.path.join(CONFIG.paths.output_dir, "model_config.json")
            with open(config_path, 'w') as f:
                json.dump(model_kwargs, f)

            # Save normalization statistics and optimal threshold
            import pickle
            normalizer_path = os.path.join(CONFIG.paths.output_dir, "normalizer.pkl")
            with open(normalizer_path, 'wb') as f:
                pickle.dump({
                    'feat_sum': kg_builder.feat_sum,
                    'feat_sq_sum': kg_builder.feat_sq_sum,
                    'feat_count': kg_builder.feat_count,
                    'best_threshold': best_threshold,
                }, f)
            print(f"  Normalizer stats + best_threshold={best_threshold:.2f} saved to {normalizer_path}")
        else:
            patience += 1
            if patience >= CONFIG.stage1.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print("Stage 1 training complete.")
    
    # Extract and cache KG features for Stage 2 (only specific required dates)
    print("\n" + "=" * 60)
    print("EXTRACTING AND CACHING KG FEATURES FOR STAGE 2 (Required days only)")
    print("=" * 60)
    
    from src.train.extract_kg_features import extract_and_cache_features
    import calendar
    
    # 2024: 15th day of each month (training set: 12 days)
    train_days_list = [15]
    train_days_by_month = {}
    for month in range(1, 13):
        train_days_by_month[month] = [d for d in train_days_list if d <= calendar.monthrange(2024, month)[1]]
    
    # 2024: Feb 10, May 10, Aug 10, Nov 10 (validation set: 4 days)
    val_days_by_month = {2: [10], 5: [10], 8: [10], 11: [10]}
    
    # 2025: 15th day of each month (test set: 12 days)
    test_days_by_month = {}
    for month in range(1, 13):
        test_days_by_month[month] = [min(15, calendar.monthrange(2025, month)[1])]
    
    # Extract 2024 training days
    print(f"\nExtracting 2024 training days (12 days)...")
    for month in range(1, 13):
        extract_and_cache_features(
            stage1_ckpt=os.path.join(CONFIG.paths.output_dir, 'stage1_best.pt'),
            years=[2024],
            months=[month],
            days=train_days_by_month[month],
            output_path=None,
            model_kwargs=model_kwargs,
        )
    
    # Extract 2024 validation days
    print(f"\nExtracting 2024 validation days (4 days)...")
    for month in [2, 5, 8, 11]:
        extract_and_cache_features(
            stage1_ckpt=os.path.join(CONFIG.paths.output_dir, 'stage1_best.pt'),
            years=[2024],
            months=[month],
            days=val_days_by_month[month],
            output_path=None,
            model_kwargs=model_kwargs,
        )
    
    # Extract 2025 test days
    print(f"\nExtracting 2025 test days (12 days)...")
    for month in range(1, 13):
        extract_and_cache_features(
            stage1_ckpt=os.path.join(CONFIG.paths.output_dir, 'stage1_best.pt'),
            years=[2025],
            months=[month],
            days=test_days_by_month[month],
            output_path=None,
            model_kwargs=model_kwargs,
        )
    
    print(f"\n✅ Stage 1 complete with KG features cached (28 days total)!")
    
    return model


if __name__ == "__main__":
    from src.utils.config import CONFIG
    train_stage1()
