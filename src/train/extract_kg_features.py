"""
Usage:
    python -m src.train.extract_kg_features
"""
import os
import pickle
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import calendar

from src.models.stage1 import Stage1Model
from src.data.kg_builder import DailyKGBuilder
from src.utils.config import CONFIG


def extract_and_cache_features(
    stage1_ckpt: str = None,
    years: list = None,
    months: list = None,
    output_path: str = None,
    days: list = None,
    model_kwargs: dict = None,
):

    if stage1_ckpt is None:
        stage1_ckpt = os.path.join(CONFIG.paths.output_dir, 'stage1_best.pt')
    if years is None:
        years = CONFIG.data.train_years
    if months is None:
        months = CONFIG.data.train_months
    if output_path is None:
        output_path = os.path.join(CONFIG.paths.output_dir, 'kg_features_cache.pkl')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nLoading Stage 1 checkpoint from: {stage1_ckpt}')
    
    # Load Stage 1 model with correct architecture
    if model_kwargs is None:
        model_kwargs = {}
    model = Stage1Model(**model_kwargs)
    state = torch.load(stage1_ckpt, map_location='cpu')
    model.load_state_dict(state['model_state'] if 'model_state' in state else state, strict=False)
    model.eval().to(device)
    print(f'Stage 1 model loaded to {device}')
    
    # Initialize KG builder
    kg_builder = DailyKGBuilder()
    
    # Load normalizer stats from training (P0-3 critical fix)
    import pickle
    normalizer_path = os.path.join(CONFIG.paths.output_dir, "normalizer.pkl")
    if os.path.exists(normalizer_path):
        with open(normalizer_path, 'rb') as f:
            normalizer_stats = pickle.load(f)
        kg_builder.feat_sum = normalizer_stats['feat_sum']
        kg_builder.feat_sq_sum = normalizer_stats['feat_sq_sum']
        kg_builder.feat_count = normalizer_stats['feat_count']
        print(f'Loaded normalizer stats from {normalizer_path}')
    else:
        print(f'WARNING: No normalizer stats found at {normalizer_path}')
    
    # Create base cache directory
    base_cache_dir = os.path.join(CONFIG.paths.output_dir, 'kg_features_cache')
    os.makedirs(base_cache_dir, exist_ok=True)
    
    tabular_dir = CONFIG.paths.tabular_dir
    saved_count = 0
    total_samples = 0
    total_size = 0
    
    # Process each year and month
    for year in years:
        for month in months:
            _, days_in_month = calendar.monthrange(year, month)
            day_range = days if days else range(1, days_in_month + 1)
            for day in day_range:
                day_str = f"{day:02d}"
                month_str = f"{month:02d}"
                year_str = f"{year:04d}"
                
                # Check if CSV exists
                fname = f"flight_with_weather_{year % 100:02d}{month_str}{day_str}.csv"
                fpath = os.path.join(tabular_dir, year_str, month_str, fname)
                if not os.path.exists(fpath):
                    continue
                
                # Check if already cached
                day_dir = os.path.join(base_cache_dir, year_str, month_str)
                day_file = os.path.join(day_dir, f"{day_str}.pkl")
                if os.path.exists(day_file):
                    print(f"  Skip {year}/{month_str}/{day_str} (already cached)")
                    continue
                
                try:
                    # Load tabular data for this day
                    tabular_df = pd.read_csv(fpath)
                    tabular_df = tabular_df.drop(
                        columns=[c for c in CONFIG.data.forbidden_cols if c in tabular_df.columns], 
                        errors='ignore'
                    )
                    n_flights = len(tabular_df)
                    
                    # Build KG graph
                    g, time_enc, n_nodes, g_chain, g_network, airport_flight_map = \
                        kg_builder.build(year, month, day, tabular_df)
                    
                    # Move to device
                    g = g.to(device)
                    g_chain = g_chain.to(device) if g_chain is not None else None
                    g_network = g_network.to(device) if g_network is not None else None
                    time_enc = time_enc.to(device)
                    
                    # Get node features
                    feat = g.ndata.get('feat', torch.zeros(g.num_nodes(), 25, device=device))
                    
                    # Prepare chain features (Step 7: 25-dim from g_chain.ndata, shared GAT)
                    chain_feat = None
                    if g_chain is not None and g_chain.num_nodes() > 0:
                        chain_feat = g_chain.ndata.get('feat', torch.zeros(g_chain.num_nodes(), 25, device=device))
                    
                    # Prepare network features
                    network_feat = None
                    network_edge_feat = None
                    if g_network is not None and g_network.num_nodes() > 0:
                        network_feat = g_network.ndata.get('feat', torch.zeros(g_network.num_nodes(), 25, device=device))
                        network_edge_feat = g_network.edata.get('feat')
                        if network_edge_feat is not None:
                            network_edge_feat = network_edge_feat.to(device)
                    
                    # Prepare airport_flight_map
                    airport_map_b = None
                    if airport_flight_map is not None:
                        airport_map_b = {
                            'flight_node_offset': airport_flight_map.get('flight_node_offset', 0),
                            'origin_ap_ids': airport_flight_map['origin_ap_ids'].to(device),
                            'dest_ap_ids': airport_flight_map['dest_ap_ids'].to(device),
                        }
                    
                    # Get all flight node IDs
                    all_flight_ids = torch.arange(n_nodes, dtype=torch.long, device=device)
                    
                    # Extract features for all flights in this day
                    with torch.no_grad():
                        out = model(
                            blocks=None,
                            feat=feat,
                            etypes_list=None,
                            time_enc=time_enc,
                            target_idx=all_flight_ids,
                            g_main=g,
                            g_chain=g_chain, chain_feat=chain_feat,
                            g_network=g_network, network_feat=network_feat,
                            network_edge_feat=network_edge_feat,
                            airport_flight_map=airport_map_b,
                            flight_nids=all_flight_ids,
                        )
                        # out['e_f'] is (n_nodes, gat_out_dim)
                        # Convert to numpy immediately to avoid torch serialization overhead
                        kg_feats = out['e_f'].detach().cpu().numpy()
                    
                    # Create directory
                    os.makedirs(day_dir, exist_ok=True)
                    
                    # Save to file: day_dir/DD.pkl
                    # IMPORTANT: Use .copy() to create independent arrays
                    day_feats = {}
                    for idx in range(n_flights):
                        if idx < len(kg_feats):
                            day_feats[idx] = kg_feats[idx].copy()
                    
                    day_cache_data = {
                        'features': day_feats,
                        'metadata': {
                            'stage1_ckpt': stage1_ckpt,
                            'date': f"{year_str}/{month_str}/{day_str}",
                            'n_samples': len(day_feats),
                            'feat_dim': CONFIG.kg.gat_out_dim,
                        }
                    }
                    
                    with open(day_file, 'wb') as f:
                        pickle.dump(day_cache_data, f)
                    
                    file_size = os.path.getsize(day_file)
                    total_size += file_size
                    saved_count += 1
                    total_samples += len(day_feats)
                    
                    # Clear GPU memory
                    del kg_feats, g, time_enc, feat
                    if g_chain is not None:
                        del g_chain
                    if g_network is not None:
                        del g_network
                    torch.cuda.empty_cache()
                    
                    print(f"  ✅ {year}/{month_str}/{day_str}: {n_flights} flights saved ({file_size/1024:.0f} KB)")
                
                except Exception as e:
                    print(f'\n  ❌ Error processing {year}/{month_str}/{day_str}: {e}')
                    import traceback
                    traceback.print_exc()
                    continue
    
    # Print summary
    print(f'\n✅ Cached {total_samples} KG features successfully!')
    print(f'   Total files: {saved_count}')
    print(f'   Total size: {total_size / 1024 / 1024:.1f} MB')
    print(f'   Cache directory: {base_cache_dir}')
    
    # Save metadata file for easy lookup
    metadata_file = os.path.join(base_cache_dir, 'metadata.json')
    metadata = {
        'stage1_ckpt': stage1_ckpt,
        'years': years,
        'months': months,
        'total_samples': total_samples,
        'total_files': saved_count,
        'total_size_mb': round(total_size / 1024 / 1024, 1),
        'feat_dim': CONFIG.kg.gat_out_dim,
        'cache_dir': base_cache_dir,
        'file_pattern': 'YYYY/MM/DD.pkl',
    }
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return base_cache_dir


if __name__ == '__main__':
    import calendar
    train_days_list = [3, 9, 15, 21, 27]
    train_days_2024 = []
    for month in range(1, 13):
        _, days_in_month = calendar.monthrange(2024, month)
        for d in train_days_list:
            if d <= days_in_month:
                train_days_2024.append((2024, month, d))
    

    val_days = []
    for month in range(1, 13):
        _, days_in_month = calendar.monthrange(2025, month)
        val_days.append((2025, month, min(10, days_in_month)))
    

    test_days = []
    for month in range(1, 13):
        _, days_in_month = calendar.monthrange(2025, month)
        day = min(15, days_in_month)
        test_days.append((2025, month, day))
    
    print("\n" + "="*60)
    print("Extracting KG features for REQUIRED DAYS ONLY")
    print("="*60)
    print(f"\n2024 Training days ({len(train_days_2024)} days):")
    for y, m, d in train_days_2024:
        print(f"  {y}/{m:02d}/{d:02d}")
    
    print(f"\n2025 Validation days ({len(val_days)} days):")
    for y, m, d in val_days:
        print(f"  {y}/{m:02d}/{d:02d}")
    
    print(f"\n2025 Test days ({len(test_days)} days):")
    for y, m, d in test_days:
        print(f"  {y}/{m:02d}/{d:02d}")
    
 
    print("\n" + "="*60)
    print("Extracting 2024 training days...")
    print("="*60)
    
 
    train_days_by_month_2024 = {}
    for y, m, d in train_days_2024:
        if m not in train_days_by_month_2024:
            train_days_by_month_2024[m] = []
        train_days_by_month_2024[m].append(d)
    
    for month in range(1, 13):
        if month in train_days_by_month_2024:
            extract_and_cache_features(
                years=[2024],
                months=[month],
                days=train_days_by_month_2024[month],
                output_path=os.path.join(CONFIG.paths.output_dir, 'kg_features_train.pkl'),
            )
    
 
    print("\n" + "="*60)
    print("Extracting 2025 validation days...")
    print("="*60)
    
    val_days_by_month_2025 = {}
    for y, m, d in val_days:
        if m not in val_days_by_month_2025:
            val_days_by_month_2025[m] = []
        val_days_by_month_2025[m].append(d)
    
    for month in range(1, 13):
        if month in val_days_by_month_2025:
            extract_and_cache_features(
                years=[2025],
                months=[month],
                days=val_days_by_month_2025[month],
                output_path=os.path.join(CONFIG.paths.output_dir, 'kg_features_val.pkl'),
            )
    
    print("\n" + "="*60)
    print("Extracting 2025 test days...")
    print("="*60)
    
    test_days_by_month_2025 = {}
    for y, m, d in test_days:
        if m not in test_days_by_month_2025:
            test_days_by_month_2025[m] = []
        test_days_by_month_2025[m].append(d)
    
    for month in range(1, 13):
        if month in test_days_by_month_2025:
            extract_and_cache_features(
                years=[2025],
                months=[month],
                days=test_days_by_month_2025[month],
                output_path=os.path.join(CONFIG.paths.output_dir, 'kg_features_test.pkl'),
            )
    
    print('\n✅ All required KG features extracted and cached!')
