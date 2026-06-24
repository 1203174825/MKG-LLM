"""
Stage 2: KG-Constrained LLM Fine-Tuning (Qwen2-1.5B + LoRA + KG Alignment).

Architecture:
  1. Frozen Stage 1 KG encoder (Main GAT + Chain GAT + Network GAT) -> 384-dim
  2. KG Alignment Layer -> projects to Qwen embedding space (1536-dim x n_tokens)
  3. Injected as soft prompts at special token positions in Qwen input
  4. Qwen2-1.5B with LoRA fine-tuning
  5. Output: natural language delay prediction

Why LLM exceeds pure KG reasoning:
  - LLM learns implicit patterns NOT encoded in KG (e.g., morning rush + weather synergy)
  - Smooths KG noise (missing edges, weak relations, inaccurate fusion gates)
  - Provides soft reasoning via analogy/generalization beyond KG hard paths
  - KG gives structured knowledge; LLM interpolates and fills gaps
"""
import os
import re
import json
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, TaskType, get_peft_model
from accelerate import Accelerator

from src.models.stage1 import Stage1Model
from src.models.stage2 import KGAlignmentLayer
from src.data.aeolus_dataset import TabularDataset
from src.data.kg_builder import DailyKGBuilder
from src.utils.config import CONFIG
from src.utils.metrics import compute_cls_metrics, compute_reg_metrics


SIGNAL_TOKEN_ID = 151925
N_KG_TOKENS = 5
PROMPT_SYSTEM = (
    "As an expert in flight delay prediction with extensive knowledge in "
    "aviation operations, weather impacts, and air traffic management, "
    "you can assess flight delay status. Based on the flight information "
    "and knowledge graph features, answer my questions."
)

DELAY_MEAN = 5.9
DELAY_STD = 23.6


def trace_delay_reason(sample: dict) -> dict:
    """Trace primary delay cause from available features.
    
    Uses a rule-based system combining weather, congestion, carrier, and time factors.
    For non-delayed flights (delay < 15min), returns 'All normal' as the reason.
    Returns a dict with 'reason', 'confidence', and 'details'.
    """
    flight = sample.get('flight_data', {})
    delay = sample.get('delay', 0)
    is_delayed = sample.get('is_delayed', 0)
    
    # Non-delayed flights always show 'All normal'
    if not is_delayed:
        return {
            'reason': 'All normal',
            'confidence': 1.0,
            'all_reasons': [('All normal', 1.0)],
        }
    
    reasons = {}
    
    # Weather at origin
    o_prcp = flight.get('O_PRCP', 0) or 0
    o_wspd = flight.get('O_WSPD', 0) or 0
    d_prcp = flight.get('D_PRCP', 0) or 0
    d_wspd = flight.get('D_WSPD', 0) or 0
    
    weather_score = (o_prcp + d_prcp) * 2 + (o_wspd + d_wspd) * 0.5
    if weather_score > 5:
        reasons['Weather'] = min(weather_score / 10, 1.0)
    
    # Congestion
    o_flights = flight.get('O_DAY_FLIGHTS', 0) or 0
    d_flights = flight.get('D_DAY_FLIGHTS', 0) or 0
    cong_score = (o_flights + d_flights) / 2000
    if cong_score > 0.5:
        reasons['Congestion'] = min(cong_score, 1.0)
    
    # Peak hour
    if flight.get('IS_PEAK_HOUR', 0):
        reasons['Peak Hour'] = 0.6
    
    # Carrier delay pattern (using aircraft type and airline)
    carrier = flight.get('OP_CARRIER', '')
    if carrier in ['UA', 'AA', 'DL']:
        reasons['Carrier Operations'] = 0.3
    
    # Sort by score
    sorted_reasons = sorted(reasons.items(), key=lambda x: x[1], reverse=True)
    
    if sorted_reasons:
        primary_reason, confidence = sorted_reasons[0]
    else:
        primary_reason = 'Other factors'
        confidence = 0.2
    
    return {
        'reason': primary_reason,
        'confidence': confidence,
        'all_reasons': sorted_reasons,
    }


def set_deterministic(seed: int = 42):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_delay_from_text(output_text: str) -> float:
    """Extract delay minutes from LLM output."""
    patterns = [
        r'delay is (\d+(?:\.\d+)?)\s*minutes',
        r'predicted delay[:\s]*(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*minutes',
        r'about\s*(\d+(?:\.\d+)?)',
        r'approximately\s*(\d+(?:\.\d+)?)',
    ]
    for pat in patterns:
        match = re.search(pat, output_text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return -1.0


class Stage2Dataset(Dataset):
    """Dataset for Stage 2 LLM fine-tuning."""
    
    def __init__(self, samples: list, kg_builder: DailyKGBuilder, stage1_model: Stage1Model, 
                 precomputed_kg_feats: dict = None):
        self.samples = samples
        self.kg_builder = kg_builder
        self.stage1_model = stage1_model
        self.precomputed_kg_feats = precomputed_kg_feats  # Cache for KG features
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            CONFIG.paths.llm_weight_dir, trust_remote_code=True
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Create tabular dataset for KG building
        self._tabular_datasets = {}
        for sample in samples:
            year = sample['year']
            if year not in self._tabular_datasets:
                self._tabular_datasets[year] = TabularDataset(year=year, months=CONFIG.data.train_months)
    
    def _get_tabular_ds(self, year: int):
        if year not in self._tabular_datasets:
            self._tabular_datasets[year] = TabularDataset(year=year, months=list(range(1, 13)))
        return self._tabular_datasets[year]
    
    def __len__(self):
        return len(self.samples)
    
    def _build_kg_feat(self, sample: dict) -> torch.Tensor:
        """Build KG features for a single sample using Stage 1 pipeline."""
        # Check if we have precomputed features
        if self.precomputed_kg_feats is not None:
            key = (sample['year'], sample['month'], sample['day'], sample['idx'])
            if key in self.precomputed_kg_feats:
                return self.precomputed_kg_feats[key]
            else:
                print(f"WARNING: KG feature not found for {key}, computing on-the-fly")
        
        # Fallback: compute on-the-fly (slow path)
        year, month, day = sample['year'], sample['month'], sample['day']
        # Get tabular data for this day
        tabular_ds = self._get_tabular_ds(year)
        tabular_df = tabular_ds.get_daily_batches(year, month, day)
        if tabular_df is None or len(tabular_df) == 0:
            return torch.zeros(CONFIG.kg.gat_out_dim)
        
        # Build KG graph
        g, time_enc, n_flights, g_chain, g_network, airport_flight_map = \
            self.kg_builder.build(year, month, day, tabular_df)
        
        # Move to device
        device = next(self.stage1_model.parameters()).device
        g = g.to(device)
        g_chain = g_chain.to(device) if g_chain is not None else None
        g_network = g_network.to(device) if g_network is not None else None
        time_enc = time_enc.to(device)
        
        # Get node features (25-dim from kg_builder, Step 7 unified)
        feat = g.ndata.get('feat', torch.zeros(g.num_nodes(), 25, device=device))
        
        # For evaluation mode, use g_main instead of blocks
        # Also need to prepare chain_feat and network_feat
        chain_feat = None
        if g_chain is not None and g_chain.num_nodes() > 0:
            chain_feat = g_chain.ndata.get('feat', torch.zeros(g_chain.num_nodes(), 25, device=device))
        
        network_feat = None
        network_edge_feat = None
        if g_network is not None and g_network.num_nodes() > 0:
            network_feat = g_network.ndata.get('feat', torch.zeros(g_network.num_nodes(), 25, device=device))
            network_edge_feat = g_network.edata.get('feat')
            if network_edge_feat is not None:
                network_edge_feat = network_edge_feat.to(device)
        
        # Prepare airport_flight_map with proper format
        airport_map_b = None
        if airport_flight_map is not None:
            airport_map_b = {
                'flight_node_offset': airport_flight_map.get('flight_node_offset', 0),
                'origin_ap_ids': airport_flight_map['origin_ap_ids'].to(device),
                'dest_ap_ids': airport_flight_map['dest_ap_ids'].to(device),
            }
        
        # Target flight node ID
        target_idx = torch.tensor([sample['idx']], dtype=torch.long, device=device)
        
        # Run through Stage 1 model in evaluation mode
        with torch.no_grad():
            out = self.stage1_model(
                blocks=None,  # Use full graph mode
                feat=feat,
                etypes_list=None,
                time_enc=time_enc,
                target_idx=target_idx,
                g_main=g,
                g_chain=g_chain, chain_feat=chain_feat,
                g_network=g_network, network_feat=network_feat,
                network_edge_feat=network_edge_feat,
                airport_flight_map=airport_map_b,
                flight_nids=target_idx,
            )
            # out['e_f'] is (1, gat_out_dim) for the target flight
            kg_feat = out['e_f'][0]
        
        return kg_feat.cpu()
    
    def _make_instruction(self, sample: dict) -> str:
        flight = sample['flight_data']
        dep_time = int(flight.get('CRS_DEP_TIME_MIN', 480))
        
        origin = flight.get('ORIGIN_INDEX', 'N/A')
        dest = flight.get('DEST_INDEX', 'N/A')
        carrier = flight.get('OP_CARRIER', 'N/A')
        fl_num = flight.get('OP_CARRIER_FL_NUM', 'N/A')
        
        # Weather at origin
        o_temp = flight.get('O_TEMP', 0)
        o_prcp = flight.get('O_PRCP', 0)
        o_wspd = flight.get('O_WSPD', 0)
        # Weather at destination
        d_temp = flight.get('D_TEMP', 0)
        d_prcp = flight.get('D_PRCP', 0)
        d_wspd = flight.get('D_WSPD', 0)
        
        # Airport congestion
        origin_cum_delay = flight.get('ORIGIN_CUM_DELAY_2H', 0)
        
        # Previous flight delays (causal features)
        prev_dep_delay = flight.get('PREV_DEP_DELAY', 0)
        prev_arr_delay = flight.get('PREV_ARR_DELAY', 0)
        
        # Peak hour
        is_peak = flight.get('IS_PEAK_HOUR', 0)
        
        state_desc = (
            f"Flight {fl_num} from {origin} to {dest}, "
            f"scheduled departure at {dep_time//60:02d}:{dep_time%60:02d}, "
            f"operated by {carrier}. "
            f"Origin weather: temp={o_temp:.0f}F, precip={o_prcp:.2f}in, wind={o_wspd:.1f}mph. "
            f"Destination weather: temp={d_temp:.0f}F, precip={d_prcp:.2f}in, wind={d_wspd:.1f}mph. "
            f"Origin airport congestion (last 2h): {origin_cum_delay:.1f} min total delays. "
            f"Previous flight of same aircraft: departure delay={prev_dep_delay:.1f}min, "
            f"arrival delay={prev_arr_delay:.1f}min. "
            f"{'Peak hour.' if is_peak else 'Off-peak.'}"
        )
        instruction = (
            f"Based on the knowledge graph features and flight information: "
            f"{state_desc}, "
            f"#kg_placeholder#, "
            f"predict whether this flight will be delayed (departure delay >= 15 minutes) "
            f"and estimate the delay in minutes. "
            f"First state if the flight is delayed or on time, then provide the "
            f"estimated delay in minutes."
        )
        return instruction
    
    def _make_response(self, sample: dict) -> str:
        delay = sample['delay']
        is_delayed = sample['is_delayed']
        if is_delayed:
            return f"The flight is delayed. The predicted delay is {delay:.0f} minutes."
        else:
            return f"The flight is on time. The predicted delay is {delay:.0f} minutes."
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        instruction = self._make_instruction(sample)
        response = self._make_response(sample)
        
        signal_ids = torch.arange(N_KG_TOKENS) + SIGNAL_TOKEN_ID
        
        user_part1 = f'<|im_start|>system\n{PROMPT_SYSTEM}\n<|im_end|>\n<|im_start|>user\n'
        user_part2 = instruction.replace('#kg_placeholder#', '')
        user_part3 = '\n<|im_end|>\n<|im_start|>assistant\n'
        
        full_text = user_part1 + user_part2 + user_part3 + response + '\n<|im_end|>'
        
        # Encode full text
        encoded = self.tokenizer.encode(full_text, add_special_tokens=False)
        encoded_tensor = torch.tensor(encoded, dtype=torch.long)
        
        # Get signal token positions
        token_ids = torch.arange(len(encoded))
        signal_positions = []
        for i, tid in enumerate(encoded):
            if SIGNAL_TOKEN_ID <= tid < SIGNAL_TOKEN_ID + N_KG_TOKENS:
                signal_positions.append(i)
        
        # Find label mask positions (after assistant token)
        assistant_pos = None
        for i in range(len(encoded) - 1, -1, -1):
            if i >= 2 and encoded[i-1] == self.tokenizer.convert_tokens_to_ids('<|im_start|>') and \
               encoded[i] == self.tokenizer.encode('assistant', add_special_tokens=False)[0]:
                assistant_pos = i
                break
        
        if assistant_pos is None:
            # Fallback: find '<|im_start|>assistant'
            for i in range(len(encoded)):
                if encoded[i] == self.tokenizer.convert_tokens_to_ids('<|im_end|>') and \
                   i + 2 < len(encoded) and encoded[i+2] == self.tokenizer.encode('assistant', add_special_tokens=False)[0]:
                    assistant_pos = i + 2
                    break
        
        if assistant_pos is not None:
            label_mask = (token_ids > assistant_pos + 1).long()
        else:
            label_mask = torch.zeros_like(token_ids)
        
        return {
            'input_ids': encoded_tensor,
            'labels': encoded_tensor,
            'label_mask': label_mask,
            'kg_feat': self._build_kg_feat(sample),
            'delay': sample['delay'],
            'is_delayed': sample['is_delayed'],
        }


def select_training_days(year=2024, days_per_month=1):
    """Select specific days from each month for training (like Stage 1).
    Uses direct file scanning to avoid TabularDataset overhead.
    For year 2024, selects days 1, 6, 11, 16, 21, 26 of each month.
    """
    import os
    import calendar
    from src.utils.config import CONFIG
    
    tabular_dir = CONFIG.paths.tabular_dir
    selected = []
    
    for month in range(1, 13):
        month_dir = os.path.join(tabular_dir, str(year), f"{month:02d}")
        if not os.path.exists(month_dir):
            continue
        
        # For 2024, select days 1, 6, 11, 16, 21, 26 (matching Stage 1's cache)
        if year == 2024:
            for day in [1, 6, 11, 16, 21, 26]:
                fname = f"flight_with_weather_{year % 100:02d}{month:02d}{day:02d}.csv"
                if os.path.exists(os.path.join(month_dir, fname)):
                    selected.append((year, month, day))
        else:
            # For other years, use the existing logic
            available_days = []
            _, days_in_month = calendar.monthrange(year, month)
            for day in range(1, days_in_month + 1):
                fname = f"flight_with_weather_{year % 100:02d}{month:02d}{day:02d}.csv"
                if os.path.exists(os.path.join(month_dir, fname)):
                    available_days.append(day)
            
            if not available_days:
                continue
            if len(available_days) <= days_per_month:
                chosen = available_days
            else:
                step = len(available_days) // days_per_month
                chosen = [available_days[i * step] for i in range(days_per_month)]
            
            selected.extend([(year, month, d) for d in chosen])
    
    return selected


def collect_samples_for_days(days: list, max_samples_per_day: int = None) -> list:
    """Collect samples from specific (year, month, day) tuples."""
    from src.data.aeolus_dataset import TabularDataset
    import os
    
    samples = []
    
    for year, month, day in days:
        print(f"  Loading {year}/{month:02d}/{day:02d}...", end=" ", flush=True)
        
        # Direct CSV read instead of TabularDataset to avoid overhead
        tabular_dir = CONFIG.paths.tabular_dir
        fname = f"flight_with_weather_{year % 100:02d}{month:02d}{day:02d}.csv"
        fpath = os.path.join(tabular_dir, str(year), f"{month:02d}", fname)
        
        if not os.path.exists(fpath):
            print("SKIP (file not found)")
            continue
        
        tabular_df = pd.read_csv(fpath)
        tabular_df = tabular_df.drop(
            columns=[c for c in CONFIG.data.forbidden_cols if c in tabular_df.columns], 
            errors='ignore'
        )
        print(f"{len(tabular_df)} flights", end="")
        
        # Optionally subsample flights per day to control memory
        # Use same fixed seed as Stage 1: 42 + year*10000 + month*100 + day
        if max_samples_per_day is not None and len(tabular_df) > max_samples_per_day:
            seed = 42 + year * 10000 + month * 100 + day
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(tabular_df), max_samples_per_day, replace=False)
            tabular_df = tabular_df.iloc[indices]
            print(f" -> subsampled to {len(tabular_df)}", end="")
        
        print()
        
        for idx, row in tabular_df.iterrows():
            delay = row[CONFIG.data.target_col]
            is_delayed = int(delay >= CONFIG.data.delay_threshold)
            samples.append({
                'year': year, 'month': month, 'day': day,
                'idx': idx,
                'delay': delay,
                'is_delayed': is_delayed,
                'flight_data': row.to_dict(),
                'date': f"{year}/{month:02d}/{day:02d}",
            })
    
    return samples


class Stage2Trainer:
    """Custom training loop for Stage 2."""
    
    def __init__(self, stage1_ckpt: str, accelerator: Accelerator):
        self.accelerator = accelerator
        self.device = accelerator.device
        
        # Load Stage 1 model (frozen)
        self.stage1_model = Stage1Model()
        state = torch.load(stage1_ckpt, map_location='cpu')
        model_state = state.get('model_state', state.get('model', state))
        self.stage1_model.load_state_dict(model_state, strict=False)
        self.stage1_model.eval()
        for p in self.stage1_model.parameters():
            p.requires_grad = False
        
        # KG Alignment Layer
        self.aligner = KGAlignmentLayer(n_tokens=N_KG_TOKENS)
        
        # Load Qwen
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(
            CONFIG.paths.llm_weight_dir, trust_remote_code=True
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Use bfloat16 for best stability + performance (wider dynamic range than fp16)
        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            CONFIG.paths.llm_weight_dir,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        # Apply LoRA (optimized based on dataset size ~5000 samples)
        # Reference: fine_tuning.py uses r=4/32, lr=1e-4, batch=1
        # Our task: larger input space (flight delays vs bearing fault), so r=8 is appropriate
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,  # alpha = 2 * r (common practice)
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        self.qwen_model = get_peft_model(self.qwen_model, lora_config)
        self.qwen_model.print_trainable_parameters()
        
        # Replace embedding with modified version
        original_embedding = self.qwen_model.get_input_embeddings()
        from src.models.stage2 import ModifiedEmbedding
        self.modified_embedding = ModifiedEmbedding(original_embedding, self.aligner)
        self.qwen_model.set_input_embeddings(self.modified_embedding)
        
        # Option B: Simplified scalar gated fusion (replaces Cross-Attention + Per-Dimension Gate)
        # KG baseline + LLM enhancement + scalar gate for stability

        # LLM pooling: extract features from instruction hidden states
        self.llm_pool = torch.nn.Sequential(
            torch.nn.Linear(1536, 384),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
        )
        
        # KG pooling: project raw KG features (384-dim) to 384-dim
        self.kg_pool = torch.nn.Linear(384, 384)
        
        # Scalar gate: single learnable parameter alpha ∈ [0,1]
        # alpha → 1: fully relies on LLM
        # alpha → 0: fully relies on KG (preserves Stage 1 performance)
        # Wrap with Module to ensure correct device placement and optimizer compatibility
        self.fusion_gate = torch.nn.Module()
        self.fusion_gate.alpha = torch.nn.Parameter(torch.tensor(0.5))
        
        # Simple prediction heads
        self.cls_head = torch.nn.Linear(384, 1)
        self.reg_head = torch.nn.Linear(384, 1)
        
        # Fixed weights for multi-task loss (avoids gradient amplification from learnable uncertainty)
        self.lm_weight = 1.0
        self.cls_weight = 1.0
        self.reg_weight = 0.5  # regression typically has smaller loss scale
        
        self.signal_token_id = SIGNAL_TOKEN_ID
        
        # Move everything to device
        self.aligner = self.aligner.to(self.device)
        self.qwen_model = self.qwen_model.to(self.device)
        self.modified_embedding = self.modified_embedding.to(self.device)
        self.llm_pool = self.llm_pool.to(self.device)
        self.kg_pool = self.kg_pool.to(self.device)
        self.fusion_gate = self.fusion_gate.to(self.device)
        self.cls_head = self.cls_head.to(self.device)
        self.reg_head = self.reg_head.to(self.device)
        
        # Optimizer: simplified fusion layers + LoRA
        self.trainable_params = (
            list(self.aligner.parameters()) +
            list(self.llm_pool.parameters()) +
            list(self.kg_pool.parameters()) +
            list(self.fusion_gate.parameters()) +
            list(self.cls_head.parameters()) +
            list(self.reg_head.parameters()) +
            [p for p in self.qwen_model.parameters() if p.requires_grad]
        )
        self.optimizer = torch.optim.AdamW(
            self.trainable_params, lr=CONFIG.stage2.llm_lr, weight_decay=0.01
        )
        
        # Scheduler
        self.num_training_steps = 0
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.num_training_steps, eta_min=1e-6
        )
    
    def get_stage1_model(self):
        """Return the Stage 1 model for dataset use."""
        return self.stage1_model
    
    def get_kg_features(self, samples: list, kg_builder: DailyKGBuilder) -> torch.Tensor:
        """Extract KG features for a batch of samples - not used anymore, features built in Dataset."""
        raise NotImplementedError("KG features are now built in Stage2Dataset.__getitem__")
    
    def train_epoch(self, dataset: Stage2Dataset, kg_builder: DailyKGBuilder,
                    batch_size: int = 2):
        """Train for one epoch with dual-task heads (LM + Classification + Regression)."""
        self.aligner.train()
        self.qwen_model.train()
        self.llm_pool.train()
        self.kg_pool.train()
        self.cls_head.train()
        self.reg_head.train()
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                               collate_fn=self._collate_fn, num_workers=0)
        
        total_loss = 0.0
        all_cls_preds = []
        all_cls_labels = []
        all_reg_preds = []
        all_reg_labels = []
        
        for step, batch in enumerate(tqdm(dataloader, desc="Training LLM")):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            label_mask = batch['label_mask'].to(self.device)
            kg_feat = batch['kg_feat'].to(self.device)
            delays = batch['delay'].to(self.device)
            is_delayed = batch['is_delayed'].to(self.device)
            
            # Data NaN check: skip batch if KG features contain NaN/Inf
            if torch.isnan(kg_feat).any() or torch.isinf(kg_feat).any():
                tqdm.write(f"  ⚠️ NaN/Inf in kg_feat at step {step}, skipping batch (bad data)")
                continue
            
            # Normalize delay for regression: (delay - mean) / std
            delays_norm = (delays - DELAY_MEAN) / DELAY_STD
            
            # Get embeddings with KG features injected
            inputs_embeds = self.modified_embedding(input_ids, kg_feat=kg_feat)
            
            # Critical: check aligner weights for NaN (stops training if corrupted)
            for name, param in self.aligner.named_parameters():
                if torch.isnan(param).any():
                    raise RuntimeError(f"FATAL: NaN in aligner.{name} at step {step} — model weights corrupted. Reduce lr or add more gradient clipping.")
            
            # Forward through Qwen with hidden states
            outputs = self.qwen_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True,
            )
            
            # Hidden states NaN check: fp16 overflow in Qwen forward
            all_hidden = outputs.hidden_states[-1].float()
            if torch.isnan(all_hidden).any() or torch.isinf(all_hidden).any():
                tqdm.write(f"  ⚠️ NaN/Inf in Qwen hidden_states at step {step}, skipping batch (fp16 overflow)")
                del outputs, inputs_embeds
                torch.cuda.empty_cache()
                continue
            # Option B: Scalar gated fusion
            all_hidden = outputs.hidden_states[-1].float()
            kg_tokens = self.aligner(kg_feat)  # (batch, 5, 1536)
            batch_actual = kg_tokens.shape[0]
            
            # Pool instruction-only hidden states (exclude response tokens)
            instr_mask = (1 - label_mask).float().unsqueeze(-1)  # (batch, seq_len, 1)
            masked_hidden = all_hidden * instr_mask
            instr_count = instr_mask.sum(dim=1).clamp(min=1)
            llm_pooled = self.llm_pool(masked_hidden.sum(dim=1) / instr_count)  # (batch, 384)
            
            # KG feature projection (kg_feat is 384-dim from Stage1)
            kg_pooled = self.kg_pool(kg_feat)  # (batch, 384)
            
            # Scalar gated fusion
            alpha = torch.sigmoid(self.fusion_gate.alpha)  # Scalar ∈ [0,1]
            fused = alpha * llm_pooled + (1 - alpha) * kg_pooled  # (batch, 384)
            
            # NaN safety check
            if torch.isnan(fused).any() or torch.isinf(fused).any():
                tqdm.write(f"  ⚠️ NaN/Inf in fused at step {step}, skipping batch")
                del outputs, inputs_embeds, all_hidden, kg_tokens
                torch.cuda.empty_cache()
                continue
            
            # Predict
            cls_logits = self.cls_head(fused).squeeze(-1)
            reg_pred_norm = self.reg_head(fused).squeeze(-1)
            
            # Denormalize regression output back to minutes
            reg_pred = reg_pred_norm * DELAY_STD + DELAY_MEAN
            
            # Fixed-weight multi-task loss (avoids gradient amplification from uncertainty weights)
            lm_loss = outputs.loss
            cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                cls_logits, is_delayed.float()
            )
            reg_loss = torch.nn.functional.huber_loss(reg_pred_norm, delays_norm, delta=1.0)
            
            loss = self.lm_weight * lm_loss + self.cls_weight * cls_loss + self.reg_weight * reg_loss
            
            self.accelerator.backward(loss)
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
            
            total_loss += loss.item()
            
            # Collect predictions for metrics
            all_cls_preds.append(torch.sigmoid(cls_logits).detach().cpu())
            all_cls_labels.append(is_delayed.detach().cpu())
            all_reg_preds.append(reg_pred.detach().cpu())
            all_reg_labels.append(delays.detach().cpu())
            
            if step % 1000 == 0:
                tqdm.write(
                    f"  Step {step}: loss={loss.item():.4f}, "
                    f"lm={lm_loss.item():.4f}, cls={cls_loss.item():.4f}, reg={reg_loss.item():.4f}"
                )
        
        # Compute metrics
        all_cls_preds = torch.cat(all_cls_preds, dim=0)
        all_cls_labels = torch.cat(all_cls_labels, dim=0)
        all_reg_preds = torch.cat(all_reg_preds, dim=0)
        all_reg_labels = torch.cat(all_reg_labels, dim=0)
        
        cls_metrics = compute_cls_metrics(all_cls_preds, all_cls_labels)
        reg_metrics = compute_reg_metrics(all_reg_preds, all_reg_labels)
        
        # Find best classification threshold (finer grid + Youden index)
        from sklearn.metrics import f1_score as sk_f1
        from sklearn.metrics import roc_curve
        cls_probs = all_cls_preds.numpy().flatten()
        cls_labels = all_cls_labels.numpy().flatten()
        
        # Threshold search by F1 score (0.01 step)
        best_thresh = 0.5
        best_f1 = 0.0
        for t in np.arange(0.1, 0.9, 0.01):
            preds = (cls_probs >= t).astype(int)
            f1_t = sk_f1(cls_labels, preds, zero_division=0)
            if f1_t > best_f1:
                best_f1 = f1_t
                best_thresh = t
        
        # Youden index (sensitivity + specificity - 1)
        if len(np.unique(cls_labels)) > 1:
            fpr, tpr, thresholds = roc_curve(cls_labels, cls_probs)
            youden = tpr - fpr
            best_youden_idx = np.argmax(youden)
            youden_thresh = thresholds[best_youden_idx] if best_youden_idx < len(thresholds) else 0.5
            youden_best = np.max(youden)
        else:
            youden_thresh = 0.5
            youden_best = 0.0
        
        # Accuracy at each threshold
        preds_05 = (cls_probs >= 0.5).astype(int)
        acc_at_05 = (preds_05 == cls_labels).mean()
        preds_best = (cls_probs >= best_thresh).astype(int)
        acc_at_best = (preds_best == cls_labels).mean()
        preds_youden = (cls_probs >= youden_thresh).astype(int)
        acc_at_youden = (preds_youden == cls_labels).mean()
        
        tqdm.write(f"\n{'='*70}")
        tqdm.write(f"  TRAINING METRICS")
        tqdm.write(f"{'='*70}")
        tqdm.write(f"  Classification: AUC={cls_metrics['auc']:.4f}, "
                   f"F1={cls_metrics['f1']:.4f} (@0.5), Best F1={best_f1:.4f} (@{best_thresh:.2f}), "
                   f"Accuracy={cls_metrics['accuracy']:.4f}")
        tqdm.write(f"  Classification: Precision={cls_metrics['precision']:.4f}, "
                   f"Recall={cls_metrics['recall']:.4f}")
        tqdm.write(f"  Threshold Tuning: F1={best_f1:.4f}@{best_thresh:.2f}, "
                   f"Youden={youden_best:.4f}@{youden_thresh:.2f}, "
                   f"Acc@0.5={acc_at_05:.4f}, Acc@Best={acc_at_best:.4f}, Acc@Youden={acc_at_youden:.4f}")
        tqdm.write(f"  Regression: MAE={reg_metrics['mae']:.2f}min, "
                   f"RMSE={reg_metrics['rmse']:.2f}min, R²={reg_metrics['r2']:.4f}")
        alpha = torch.sigmoid(self.fusion_gate.alpha).item()
        tqdm.write(f"  Fusion: alpha={alpha:.4f} (LLM weight={alpha:.2f}, KG weight={1-alpha:.2f})")
        tqdm.write(f"{'='*70}")
        
        # Print structured output examples: 1 correct on-time, 1 correct delayed, 1 wrong prediction
        tqdm.write("\n  Structured Prediction Examples with Delay Reason:")
        
        # Find samples of each type
        correct_on_time_idx = None
        correct_delayed_idx = None
        wrong_idx = None
        
        for i in range(len(all_cls_preds)):
            is_delayed_pred = bool(cls_probs[i] >= best_thresh)
            is_delayed_actual = bool(cls_labels[i] >= 0.5)
            
            if is_delayed_actual and is_delayed_pred and correct_delayed_idx is None:
                correct_delayed_idx = i
            elif not is_delayed_actual and not is_delayed_pred and correct_on_time_idx is None:
                correct_on_time_idx = i
            elif is_delayed_actual != is_delayed_pred and wrong_idx is None:
                wrong_idx = i
            
            if correct_on_time_idx and correct_delayed_idx and wrong_idx:
                break
        
        # Print examples
        example_indices = []
        if correct_on_time_idx is not None:
            example_indices.append((correct_on_time_idx, "Correct: ON TIME predicted ON TIME"))
        if correct_delayed_idx is not None:
            example_indices.append((correct_delayed_idx, "Correct: DELAYED predicted DELAYED"))
        if wrong_idx is not None:
            example_indices.append((wrong_idx, "WRONG: Misclassification"))
        
        if not example_indices:
            example_indices = [(i, f"Sample {i}") for i in range(min(3, len(all_cls_preds)))]
        
        for idx, desc in example_indices:
            is_delayed_pred = bool(cls_probs[idx] >= best_thresh)
            delay_pred = all_reg_preds[idx].item()
            delay_actual = all_reg_labels[idx].item()
            is_delayed_actual = bool(cls_labels[idx] >= 0.5)
            status = "DELAYED" if is_delayed_pred else "ON TIME"
            status_actual = "DELAYED" if is_delayed_actual else "ON TIME"
            flight = dataset.samples[idx]['flight_data']
            dep_time = int(flight.get('CRS_DEP_TIME_MIN', 480))
            date_str = dataset.samples[idx].get('date', 'Unknown')
            
            delay_info = trace_delay_reason(dataset.samples[idx])
            reason_str = delay_info['reason']
            reason_conf = delay_info['confidence']
            
            all_reasons_str = ', '.join([f"{r}({c:.2f})" for r, c in delay_info['all_reasons']])
            if not all_reasons_str:
                all_reasons_str = "No specific factors identified"
            
            example = (
                f"    [{desc}]\n"
                f"    {date_str}: Flight {flight.get('OP_CARRIER_FL_NUM', 'N/A')} "
                f"from {flight.get('ORIGIN_INDEX', '?')} to {flight.get('DEST_INDEX', '?')} "
                f"at {dep_time//60:02d}:{dep_time%60:02d}\n"
                f"      → Prediction: {status}, Estimated Delay: {delay_pred:.1f} min "
                f"(Actual: {status_actual}, {delay_actual:.1f} min)\n"
                f"      → Primary Delay Reason: {reason_str} (confidence: {reason_conf:.2f})\n"
                f"      → All Factors: {all_reasons_str}"
            )
            tqdm.write(example)
        
        return total_loss / len(dataloader)
    
    def _get_last_token_hidden(self, logits, input_ids, label_mask):
        """Extract hidden state of the last non-padded token for each sample.
        
        Uses the model to re-compute hidden states, then picks the last token
        position that has label_mask=1 (i.e., the generated response tokens).
        """
        # Re-run forward pass to get hidden states
        # outputs.logits shape: (batch, seq_len, vocab_size)
        # We need hidden states, so we use the model with output_hidden_states=True
        
        # Use label_mask to find last response token position per sample
        # label_mask[i, j] = 1 means token j is part of the response
        batch_size = input_ids.shape[0]
        last_token_indices = label_mask.sum(dim=1) - 1  # last response token index
        last_token_indices = torch.clamp(last_token_indices, min=0)
        
        # For the current implementation, we need hidden states.
        # Since we only have logits, we'll use a workaround:
        # Get hidden states from the base model
        return self._extract_last_hidden(input_ids, last_token_indices)
    
    def _extract_last_hidden(self, input_ids, last_token_indices):
        """Extract hidden states from Qwen model for specific token positions."""
        with torch.no_grad():
            outputs = self.qwen_model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
            )
            # Last layer hidden states: (batch, seq_len, hidden_dim)
            hidden_states = outputs.hidden_states[-1]
            # Pick the hidden state at last_token_indices for each sample
            batch_indices = torch.arange(len(last_token_indices), device=self.device)
            return hidden_states[batch_indices, last_token_indices]
    
    def _collate_fn(self, batch: list) -> dict:
        """Collate function for DataLoader."""
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
            torch.cat([x, torch.full((max_len - len(x),), self.tokenizer.pad_token_id, dtype=torch.long)])
            for x in input_ids
        ])
        labels = torch.stack([
            torch.cat([x, torch.full((max_len - len(x),), -100, dtype=torch.long)])
            for x in labels
        ])
        label_masks = torch.stack([
            torch.cat([x, torch.ones(max_len - len(x), dtype=torch.long)])
            for x in label_masks
        ])
        kg_feats = torch.stack([
            torch.tensor(f, dtype=torch.float) if isinstance(f, np.ndarray) else f
            for f in kg_feats
        ])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'label_mask': label_masks,
            'kg_feat': kg_feats,
            'delay': torch.tensor(delays, dtype=torch.float),
            'is_delayed': torch.tensor(is_delayed, dtype=torch.long),
        }
    
    def validate(self, dataset: Stage2Dataset, kg_builder: DailyKGBuilder,
                 batch_size: int = 2):
        """Validate the model with dual-task heads."""
        self.aligner.eval()
        self.qwen_model.eval()
        self.llm_pool.eval()
        self.kg_pool.eval()
        self.cls_head.eval()
        self.reg_head.eval()
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                               collate_fn=self._collate_fn, num_workers=0)
        
        total_loss = 0.0
        all_cls_preds = []
        all_cls_labels = []
        all_reg_preds = []
        all_reg_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating LLM"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                label_mask = batch['label_mask'].to(self.device)
                kg_feat = batch['kg_feat'].to(self.device)
                delays = batch['delay'].to(self.device)
                is_delayed = batch['is_delayed'].to(self.device)
                
                inputs_embeds = self.modified_embedding(input_ids, kg_feat=kg_feat)
                outputs = self.qwen_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_hidden_states=True,
                )
                
                # Option B: Scalar gated fusion
                all_hidden = outputs.hidden_states[-1].float()
                kg_tokens = self.aligner(kg_feat)  # (batch, 5, 1536)
                
                # Pool instruction-only hidden states (exclude response tokens)
                instr_mask = (1 - label_mask).float().unsqueeze(-1)
                masked_hidden = all_hidden * instr_mask
                instr_count = instr_mask.sum(dim=1).clamp(min=1)
                llm_pooled = self.llm_pool(masked_hidden.sum(dim=1) / instr_count)  # (batch, 512)
                
                # KG feature projection
                kg_pooled = self.kg_pool(kg_feat)  # (batch, 384)
                
                # Scalar gated fusion
                alpha = torch.sigmoid(self.fusion_gate.alpha)  # Scalar ∈ [0,1]
                fused = alpha * llm_pooled + (1 - alpha) * kg_pooled  # (batch, 384)
                
                # Predict
                cls_logits = self.cls_head(fused).squeeze(-1)
                reg_pred_norm = self.reg_head(fused).squeeze(-1)
                
                # Denormalize regression output back to minutes
                reg_pred = reg_pred_norm * DELAY_STD + DELAY_MEAN
                
                # Normalize delay for loss computation
                delays_norm = (delays - DELAY_MEAN) / DELAY_STD
                
                # Fixed-weight loss (same as training)
                lm_loss = outputs.loss
                cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    cls_logits, is_delayed.float()
                )
                reg_loss = torch.nn.functional.huber_loss(reg_pred_norm, delays_norm, delta=1.0)
                
                loss = self.lm_weight * lm_loss + self.cls_weight * cls_loss + self.reg_weight * reg_loss
                
                total_loss += loss.item()
                
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
        
        # Find best classification threshold (finer grid + Youden index)
        from sklearn.metrics import f1_score as sk_f1
        from sklearn.metrics import roc_curve
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
        
        if len(np.unique(cls_labels)) > 1:
            fpr, tpr, thresholds = roc_curve(cls_labels, cls_probs)
            youden = tpr - fpr
            best_youden_idx = np.argmax(youden)
            youden_thresh = thresholds[best_youden_idx] if best_youden_idx < len(thresholds) else 0.5
            youden_best = np.max(youden)
        else:
            youden_thresh = 0.5
            youden_best = 0.0
        
        preds_05 = (cls_probs >= 0.5).astype(int)
        acc_at_05 = (preds_05 == cls_labels).mean()
        preds_best = (cls_probs >= best_thresh).astype(int)
        acc_at_best = (preds_best == cls_labels).mean()
        preds_youden = (cls_probs >= youden_thresh).astype(int)
        acc_at_youden = (preds_youden == cls_labels).mean()
        
        print(f"\n{'='*70}")
        print(f"  VALIDATION/TEST METRICS")
        print(f"{'='*70}")
        print(f"  Classification: AUC={cls_metrics['auc']:.4f}, "
              f"F1={cls_metrics['f1']:.4f} (@0.5), Best F1={best_f1:.4f} (@{best_thresh:.2f}), "
              f"Accuracy={cls_metrics['accuracy']:.4f}")
        print(f"  Classification: Precision={cls_metrics['precision']:.4f}, "
              f"Recall={cls_metrics['recall']:.4f}")
        print(f"  Threshold Tuning: F1={best_f1:.4f}@{best_thresh:.2f}, "
              f"Youden={youden_best:.4f}@{youden_thresh:.2f}, "
              f"Acc@0.5={acc_at_05:.4f}, Acc@Best={acc_at_best:.4f}, Acc@Youden={acc_at_youden:.4f}")
        print(f"  Regression: MAE={reg_metrics['mae']:.2f}min, "
              f"RMSE={reg_metrics['rmse']:.2f}min, R²={reg_metrics['r2']:.4f}")
        print(f"{'='*70}")
        
        # Print structured output examples: 1 correct on-time, 1 correct delayed, 1 wrong prediction
        print("\n  Structured Prediction Examples with Delay Reason:")
        
        correct_on_time_idx = None
        correct_delayed_idx = None
        wrong_idx = None
        
        for i in range(len(all_cls_preds)):
            is_delayed_pred = bool(cls_probs[i] >= best_thresh)
            is_delayed_actual = bool(cls_labels[i] >= 0.5)
            
            if is_delayed_actual and is_delayed_pred and correct_delayed_idx is None:
                correct_delayed_idx = i
            elif not is_delayed_actual and not is_delayed_pred and correct_on_time_idx is None:
                correct_on_time_idx = i
            elif is_delayed_actual != is_delayed_pred and wrong_idx is None:
                wrong_idx = i
            
            if correct_on_time_idx and correct_delayed_idx and wrong_idx:
                break
        
        example_indices = []
        if correct_on_time_idx is not None:
            example_indices.append((correct_on_time_idx, "Correct: ON TIME predicted ON TIME"))
        if correct_delayed_idx is not None:
            example_indices.append((correct_delayed_idx, "Correct: DELAYED predicted DELAYED"))
        if wrong_idx is not None:
            example_indices.append((wrong_idx, "WRONG: Misclassification"))
        
        if not example_indices:
            example_indices = [(i, f"Sample {i}") for i in range(min(3, len(all_cls_preds)))]
        
        for idx, desc in example_indices:
            is_delayed_pred = bool(cls_probs[idx] >= best_thresh)
            delay_pred = all_reg_preds[idx].item()
            delay_actual = all_reg_labels[idx].item()
            is_delayed_actual = bool(cls_labels[idx] >= 0.5)
            
            status = "DELAYED" if is_delayed_pred else "ON TIME"
            status_actual = "DELAYED" if is_delayed_actual else "ON TIME"
            flight = dataset.samples[idx]['flight_data']
            dep_time = int(flight.get('CRS_DEP_TIME_MIN', 480))
            date_str = dataset.samples[idx].get('date', 'Unknown')
            
            delay_info = trace_delay_reason(dataset.samples[idx])
            reason_str = delay_info['reason']
            reason_conf = delay_info['confidence']
            all_reasons_str = ', '.join([f"{r}({c:.2f})" for r, c in delay_info['all_reasons']])
            if not all_reasons_str:
                all_reasons_str = "No specific factors identified"
            
            example = (
                f"    [{desc}]\n"
                f"    {date_str}: Flight {flight.get('OP_CARRIER_FL_NUM', 'N/A')} "
                f"from {flight.get('ORIGIN_INDEX', '?')} to {flight.get('DEST_INDEX', '?')} "
                f"at {dep_time//60:02d}:{dep_time%60:02d}\n"
                f"      → Prediction: {status}, Estimated Delay: {delay_pred:.1f} min "
                f"(Actual: {status_actual}, {delay_actual:.1f} min)\n"
                f"      → Primary Delay Reason: {reason_str} (confidence: {reason_conf:.2f})\n"
                f"      → All Factors: {all_reasons_str}"
            )
            print(example)
        
        return {
            'loss': total_loss / len(dataloader),
            'auc': cls_metrics['auc'],
            'f1': cls_metrics['f1'],
            'best_f1': best_f1,
            'best_threshold': best_thresh,
            'youden_f1': youden_best,
            'youden_threshold': youden_thresh,
            'mae': reg_metrics['mae'],
            'rmse': reg_metrics['rmse'],
            'r2': reg_metrics['r2'],
            'accuracy': cls_metrics['accuracy'],
            'precision': cls_metrics['precision'],
            'recall': cls_metrics['recall'],
        }
    
    def save_checkpoint(self, save_dir: str, epoch: int = 0, save_dir_name: str = None):
        """Save LoRA adapter, alignment layer, optimizer state, and training metadata."""
        save_dir = os.path.join(save_dir, save_dir_name) if save_dir_name else save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.qwen_model.save_pretrained(os.path.join(save_dir, 'lora_adapter'))
        torch.save(self.aligner.state_dict(), os.path.join(save_dir, 'aligner.pt'))
        torch.save(self.llm_pool.state_dict(), os.path.join(save_dir, 'llm_pool.pt'))
        torch.save(self.kg_pool.state_dict(), os.path.join(save_dir, 'kg_pool.pt'))
        torch.save(self.fusion_gate.state_dict(), os.path.join(save_dir, 'fusion_gate.pt'))
        torch.save(self.cls_head.state_dict(), os.path.join(save_dir, 'cls_head.pt'))
        torch.save(self.reg_head.state_dict(), os.path.join(save_dir, 'reg_head.pt'))
        torch.save({
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }, os.path.join(save_dir, 'training_state.pt'))
        print(f"Checkpoint saved to {save_dir}")
    
    def load_checkpoint(self, save_dir: str):
        """Resume training from saved checkpoint."""
        ckpt_dir = os.path.join(save_dir, 'lora_adapter')
        if not os.path.exists(ckpt_dir):
            return -1
        
        self.qwen_model.load_adapter(ckpt_dir, adapter_name='default')
        self.aligner.load_state_dict(torch.load(os.path.join(save_dir, 'aligner.pt'), map_location=self.device))
        self.llm_pool.load_state_dict(torch.load(os.path.join(save_dir, 'llm_pool.pt'), map_location=self.device))
        self.kg_pool.load_state_dict(torch.load(os.path.join(save_dir, 'kg_pool.pt'), map_location=self.device))
        self.fusion_gate.load_state_dict(torch.load(os.path.join(save_dir, 'fusion_gate.pt'), map_location=self.device))
        cls_head_path = os.path.join(save_dir, 'cls_head.pt')
        if os.path.exists(cls_head_path):
            self.cls_head.load_state_dict(torch.load(cls_head_path, map_location=self.device))
        reg_head_path = os.path.join(save_dir, 'reg_head.pt')
        if os.path.exists(reg_head_path):
            self.reg_head.load_state_dict(torch.load(reg_head_path, map_location=self.device))
        training_state = torch.load(os.path.join(save_dir, 'training_state.pt'), map_location=self.device)
        self.optimizer.load_state_dict(training_state['optimizer'])
        self.scheduler.load_state_dict(training_state['scheduler'])
        print(f"Resumed from epoch {training_state['epoch'] + 1}")
        return training_state['epoch']
    
    def generate_prediction(self, kg_feat: torch.Tensor, prompt: str,
                            max_new_tokens: int = 50) -> str:
        """Generate a structured prediction with both LLM text and dual-task outputs."""
        self.aligner.eval()
        self.qwen_model.eval()
        self.cls_head.eval()
        self.reg_head.eval()
        
        # Build prompt
        full_prompt = f'<|im_start|>system\n{PROMPT_SYSTEM}\n<|im_end|>\n<|im_start|>user\n{prompt}\n<|im_end|>\n<|im_start|>assistant\n'
        input_ids = self.tokenizer.encode(full_prompt, return_tensors='pt').to(self.device)
        
        # Inject KG features
        with torch.no_grad():
            inputs_embeds = self.modified_embedding(input_ids, kg_feat=kg_feat)
            
            # LLM text generation
            outputs = self.qwen_model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            generated_text = self.tokenizer.decode(outputs[0, input_ids.shape[1]:], skip_special_tokens=True)
            
            # Dual-task predictions from hidden states
            lm_outputs = self.qwen_model(inputs_embeds=inputs_embeds, output_hidden_states=True)
            last_hidden = lm_outputs.hidden_states[-1].float()
            last_token_idx = lm_outputs.logits.shape[1] - 1
            last_token_hidden = last_hidden[0, last_token_idx, :].unsqueeze(0)
            
            cls_logit = self.cls_head(last_token_hidden).squeeze(-1).item()
            reg_pred = self.reg_head(last_token_hidden).squeeze(-1).item()
            
            is_delayed = cls_logit >= 0
            delay_minutes = max(0, reg_pred)
            status = "DELAYED" if is_delayed else "ON TIME"
            
            # Build structured output
            structured_output = (
                f"\n{'='*60}\n"
                f"STRUCTURED PREDICTION\n"
                f"{'='*60}\n"
                f"Date: {prompt.split('Flight on ')[1].split(':')[0] if 'Flight on ' in prompt else 'Unknown'}\n"
                f"Status: {status}\n"
                f"Estimated Delay: {delay_minutes:.1f} minutes\n"
                f"{'='*60}\n\n"
                f"LLM Response:\n{generated_text}"
            )
        
        return structured_output


def load_kg_cache(cache_dir, years, months):
    """Load KG features from cache directory structure."""
    if not os.path.exists(cache_dir):
        return None
    
    import json
    import pickle
    
    metadata_file = os.path.join(cache_dir, 'metadata.json')
    if not os.path.exists(metadata_file):
        return None
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"\nLoading KG features from {cache_dir}...")
    print(f"  Total cached samples: {metadata.get('total_samples', 'unknown')}")
    print(f"  Cache size: {metadata.get('total_size_mb', 'unknown')} MB")
    
    all_feats = {}
    for year in years:
        year_str = f"{year:04d}"
        year_dir = os.path.join(cache_dir, year_str)
        if not os.path.exists(year_dir):
            print(f"  Skipping year {year} (no cache)")
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
                    all_feats[(year, month, day, idx)] = feat
    
    print(f"  [OK] Loaded {len(all_feats)} KG features")
    return all_feats


def kg_feats_builder(kg_builder: DailyKGBuilder, sample: dict) -> torch.Tensor:
    """Build KG features for a single sample."""
    year, month, day = sample['year'], sample['month'], sample['day']
    flight_df, airport_df, aircraft_df, airline_df, weather_df, _, _ = \
        kg_builder.build_daily_kg(year, month, day)
    feat = kg_builder.build_flight_features(flight_df)
    return feat[sample['idx']]


def train_stage2(stage1_checkpoint: str = None):
    """Main training function for Stage 2."""
    set_deterministic(CONFIG.train.seed)
    accelerator = Accelerator()
    device = accelerator.device
    
    if stage1_checkpoint is None:
        stage1_checkpoint = os.path.join(CONFIG.paths.output_dir, 'stage1_best.pt')
    
    print(f"\nLoading Stage 1 checkpoint from: {stage1_checkpoint}")
    trainer = Stage2Trainer(stage1_ckpt=stage1_checkpoint, accelerator=accelerator)
    
    # Initialize data loader and KG builder
    kg_builder = DailyKGBuilder()
    
    # Load precomputed KG features if available
    train_cache_dir = os.path.join(CONFIG.paths.output_dir, 'kg_features_cache')
    
    # Select training days: 2024, 15th day of each month (matching Stage 1)
    print("\nSelecting training days (2024, 15th day/month)...")
    train_days = []
    for month in range(1, 13):
        import calendar
        _, days_in_month = calendar.monthrange(2024, month)
        day = min(15, days_in_month)
        train_days.append((2024, month, day))
    print(f"Selected {len(train_days)} training days:")
    for y, m, d in train_days:
        print(f"  {y}/{m:02d}/{d:02d}")
    
    # Select validation days: 2024, 10th day of Q2/Q3/Q4/Q1 months (Feb/May/Aug/Nov)
    print("\nSelecting validation days (2024, Feb/May/Aug/Nov 10th)...")
    val_days = []
    for month in [2, 5, 8, 11]:
        import calendar
        _, days_in_month = calendar.monthrange(2024, month)
        day = min(10, days_in_month)
        val_days.append((2024, month, day))
    print(f"Selected {len(val_days)} validation days:")
    for y, m, d in val_days:
        print(f"  {y}/{m:02d}/{d:02d}")
    
    # Select test days: 2025, 15th day of each month
    print("\nSelecting test days (2025, 15th day/month)...")
    import calendar
    test_days = []
    for month in range(1, 13):
        _, days_in_month = calendar.monthrange(2025, month)
        day = min(15, days_in_month)
        test_days.append((2025, month, day))
    print(f"Selected {len(test_days)} test days:")
    for y, m, d in test_days:
        print(f"  {y}/{m:02d}/{d:02d}")
    
    # Load precomputed KG features (only for available years)
    train_kg_feats = load_kg_cache(train_cache_dir, [2024], list(range(1, 13)))
    val_kg_feats = load_kg_cache(train_cache_dir, [2024], [2, 5, 8, 11])
    test_kg_feats = load_kg_cache(train_cache_dir, [2025], list(range(1, 13)))
    
    if train_kg_feats is None:
        print(f"\n⚠️ No precomputed KG features found at {train_cache_dir}")
        print("   KG features will be computed on-the-fly (slower)")
    
    # Collect samples (all flights per day)
    print("\nCollecting training samples (ALL flights/day)...")
    train_samples = collect_samples_for_days(train_days, max_samples_per_day=None)
    print(f"Collected {len(train_samples)} training samples")
    
    print("\nCollecting validation samples (ALL flights/day)...")
    val_samples = collect_samples_for_days(val_days, max_samples_per_day=None)
    print(f"Collected {len(val_samples)} validation samples")
    
    print("\nCollecting test samples (ALL flights/day)...")
    test_samples = collect_samples_for_days(test_days, max_samples_per_day=None)
    print(f"Collected {len(test_samples)} test samples")
    
    # Create datasets with precomputed features
    stage1_model = trainer.get_stage1_model()
    train_dataset = Stage2Dataset(train_samples, kg_builder, stage1_model, 
                                  precomputed_kg_feats=train_kg_feats)
    val_dataset = Stage2Dataset(val_samples, kg_builder, stage1_model,
                                precomputed_kg_feats=val_kg_feats)
    test_dataset = Stage2Dataset(test_samples, kg_builder, stage1_model,
                                 precomputed_kg_feats=test_kg_feats)
    
    # Set scheduler steps
    trainer.num_training_steps = len(train_dataset) // CONFIG.stage2.llm_batch_size * CONFIG.stage2.llm_epochs
    trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, T_max=trainer.num_training_steps, eta_min=1e-6
    )
    
    # Check for existing checkpoint to resume
    latest_epoch = -1
    for e in range(1, CONFIG.stage2.llm_epochs + 1):
        ckpt_dir = os.path.join(CONFIG.paths.output_dir, f'stage2_epoch{e}')
        if os.path.exists(os.path.join(ckpt_dir, 'lora_adapter')):
            latest_epoch = e
    
    start_epoch = 0
    if latest_epoch >= 0:
        ckpt_dir = os.path.join(CONFIG.paths.output_dir, f'stage2_epoch{latest_epoch}')
        loaded_epoch = trainer.load_checkpoint(ckpt_dir)
        if loaded_epoch >= 0:
            start_epoch = loaded_epoch + 1
            print(f"\n>>> Resuming from epoch {start_epoch} <<<")
    
    # Epoch metrics table
    epoch_metrics = []
    best_val_auc = -1.0
    best_epoch = -1
    
    # Training loop
    print(f"\nTraining LLM for {CONFIG.stage2.llm_epochs} epochs...")
    for epoch in range(start_epoch, CONFIG.stage2.llm_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{CONFIG.stage2.llm_epochs}")
        print(f"{'='*60}")
        
        train_loss = trainer.train_epoch(train_dataset, kg_builder,
                                         batch_size=CONFIG.stage2.llm_batch_size)
        print(f"Train loss: {train_loss:.4f}")
        
        val_metrics = trainer.validate(val_dataset, kg_builder,
                                       batch_size=CONFIG.stage2.llm_batch_size)
        print(f"Validation loss: {val_metrics['loss']:.4f}")
        
        val_auc = val_metrics['auc']
        
        # Track best val AUC for Stage 2
        is_best = val_auc > best_val_auc
        if is_best:
            best_val_auc = val_auc
            best_epoch = epoch
            best_save_dir = os.path.join(CONFIG.paths.output_dir, 'stage2_best_val_auc')
            trainer.save_checkpoint(best_save_dir, epoch=epoch, save_dir_name=None)
            print(f"  [NEW BEST] Val AUC={val_auc:.4f} → saved to {best_save_dir}")
        
        # Also run test for logging (not for selection)
        test_metrics = trainer.validate(test_dataset, kg_builder,
                                        batch_size=CONFIG.stage2.llm_batch_size)
        print(f"Test loss: {test_metrics['loss']:.4f}")
        
        epoch_metrics.append({
            'epoch': epoch + 1,
            'val_auc': val_metrics['auc'],
            'val_best_f1': val_metrics['best_f1'],
            'val_best_threshold': val_metrics['best_threshold'],
            'val_mae': val_metrics['mae'],
            'val_rmse': val_metrics['rmse'],
            'val_r2': val_metrics['r2'],
            'test_auc': test_metrics['auc'],
            'test_best_f1': test_metrics['best_f1'],
            'test_best_threshold': test_metrics['best_threshold'],
            'test_mae': test_metrics['mae'],
            'test_rmse': test_metrics['rmse'],
            'test_r2': test_metrics['r2'],
        })
        
        # Save checkpoint (with optimizer/scheduler state for resume)
        save_dir = os.path.join(CONFIG.paths.output_dir, f'stage2_epoch{epoch+1}')
        trainer.save_checkpoint(save_dir, epoch=epoch)
        
        print(f"\n>>> Epoch {epoch+1} complete. To resume, simply run the script again <<<")
    
    # After all epochs: load best val AUC checkpoint and run FINAL test
    print(f"\n{'='*60}")
    print(f"BEST VALIDATION AUC: {best_val_auc:.4f} at epoch {best_epoch+1}")
    print(f"Loading best checkpoint for final test evaluation...")
    print(f"{'='*60}")
    
    best_ckpt_dir = os.path.join(CONFIG.paths.output_dir, 'stage2_best_val_auc')
    if os.path.exists(os.path.join(best_ckpt_dir, 'lora_adapter')):
        trainer.load_checkpoint(best_ckpt_dir)
        best_test_metrics = trainer.validate(test_dataset, kg_builder,
                                             batch_size=CONFIG.stage2.llm_batch_size)
        print(f"\n{'='*70}")
        print(f"  FINAL TEST RESULTS (Best Val AUC Model @ Epoch {best_epoch+1})")
        print(f"{'='*70}")
        print(f"  Classification: AUC={best_test_metrics['auc']:.4f}, "
              f"F1={best_test_metrics['best_f1']:.4f} (@{best_test_metrics['best_threshold']:.2f}), "
              f"Accuracy={best_test_metrics['accuracy']:.4f}")
        print(f"  Regression: MAE={best_test_metrics['mae']:.2f}min, "
              f"RMSE={best_test_metrics['rmse']:.2f}min, R²={best_test_metrics['r2']:.4f}")
        print(f"{'='*70}")
    else:
        print("  WARNING: Best checkpoint not found, using last epoch results")
    
    # Print epoch comparison table
    print(f"\n{'='*100}")
    print(f"  Stage 2: Epoch-by-Epoch Comparison + Stage 1 Baseline")
    print(f"{'='*100}")
    print(f"  {'Epoch':>5} | {'Val AUC':>7} {'Val F1':>7} {'Val Thresh':>10} | {'Val MAE':>8} {'Val RMSE':>9} {'Val R²':>7} | {'Test AUC':>8} {'Test F1':>7} {'Test Thresh':>11} | {'Test MAE':>8} {'Test RMSE':>9} {'Test R²':>7}")
    print(f"  {'-'*5}-+-{'-'*7}-{'-'*7}-{'-'*10}-+-{'-'*8}-{'-'*9}-{'-'*7}-+-{'-'*8}-{'-'*7}-{'-'*11}-+-{'-'*8}-{'-'*9}-{'-'*7}")
    
    stage1_baseline = {
        'auc': 0.8469,
        'best_f1': 0.6299,
        'mae': 9.88,
        'rmse': 20.37,
        'r2': 0.4228,
    }
    print(f"  Stage1| {stage1_baseline['auc']:>7.4f} {stage1_baseline['best_f1']:>7.4f} {'N/A':>10} | {stage1_baseline['mae']:>8.2f} {stage1_baseline['rmse']:>9.2f} {stage1_baseline['r2']:>7.4f} | {'N/A':>8} {'N/A':>7} {'N/A':>11} | {'N/A':>8} {'N/A':>9} {'N/A':>7}")
    
    for m in epoch_metrics:
        print(f"  {m['epoch']:>5} | {m['val_auc']:>7.4f} {m['val_best_f1']:>7.4f} {m['val_best_threshold']:>10.2f} | {m['val_mae']:>8.2f} {m['val_rmse']:>9.2f} {m['val_r2']:>7.4f} | {m['test_auc']:>8.4f} {m['test_best_f1']:>7.4f} {m['test_best_threshold']:>11.2f} | {m['test_mae']:>8.2f} {m['test_rmse']:>9.2f} {m['test_r2']:>7.4f}")
    
    print(f"\n{'='*100}")
    print(f"\nStage 2 training complete!")
    return trainer


if __name__ == "__main__":
    import sys
    import os
    # Add project root directory to Python path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.utils.config import CONFIG
    # Set output directory to Output/src
    CONFIG.paths.output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Output', 'src')
    train_stage2()
