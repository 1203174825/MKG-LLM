"""
Project configuration and hyperparameters.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional

# Project root directory
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==============================================================================
# Paths
# ==============================================================================
@dataclass
class Paths:
    dataset_root: str = os.path.join(_PROJECT_ROOT, "Dataset")
    llm_weight_dir: str = os.path.join(_PROJECT_ROOT, "LLM", "qwen_weight")
    script_dir: str = os.path.join(_PROJECT_ROOT, "scripts")
    output_dir: str = os.path.join(_PROJECT_ROOT, "Output")

    # Data modalities
    tabular_dir: str = os.path.join(_PROJECT_ROOT, "Dataset", "Flight_Tabular")
    chain_dir: str = os.path.join(_PROJECT_ROOT, "Dataset", "Flight_Chain")
    network_dir: str = os.path.join(_PROJECT_ROOT, "Dataset", "Flight_Network")


# ==============================================================================
# Dataset
# ==============================================================================
@dataclass
class DataConfig:
    train_years: List[int] = field(default_factory=lambda: [2024])
    train_months: List[int] = field(default_factory=lambda: list(range(1, 13)))  # All 12 months
    val_years: List[int] = field(default_factory=lambda: [2024])
    val_months: List[int] = field(default_factory=lambda: [2])  # February for validation
    test_years: List[int] = field(default_factory=lambda: [2025])
    test_months: List[int] = field(default_factory=lambda: list(range(1, 13)))  # All 12 months

    target_col: str = "DEP_DELAY"
    delay_threshold: int = 15  # minutes, for binary classification
    max_seq_len: int = 6       # max chain length (same as Aeolus_V2)

    # Feature columns
    cat_cols: List[str] = field(default_factory=lambda: [
        "OP_CARRIER", "OP_CARRIER_FL_NUM", "ORIGIN_INDEX", "DEST_INDEX",
        "TAIL_NUM", "FL_MONTH", "FL_DAY", "FL_WEEK",
        "ORIGIN_TIER", "DEST_TIER"  # Airport tier (discrete level)
    ])
    cont_cols: List[str] = field(default_factory=lambda: [
        "CRS_DEP_TIME_MIN", "CRS_ARR_TIME_MIN", "CRS_ELAPSED_TIME",
        "FLIGHTS",
        "PREV_DEP_DELAY", "PREV_ARR_DELAY",  # Predecessor flight already flown, delay values known (causal features)
        "ORIGIN_CUM_DELAY_2H",      # Origin airport cumulative delay in past 2h (real-time congestion score)
        "O_TEMP", "O_PRCP", "O_WSPD",
        "D_TEMP", "D_PRCP", "D_WSPD",
        "O_LATITUDE", "O_LONGITUDE", "D_LATITUDE", "D_LONGITUDE",
        "O_DAY_FLIGHTS", "D_DAY_FLIGHTS", "IS_PEAK_HOUR",
    ])
    forbidden_cols: List[str] = field(default_factory=lambda: [
        "DEP_TIME", "ARR_TIME",
        "ACTUAL_ELAPSED_TIME", "AIR_TIME",
        "TAXI_OUT", "TAXI_IN", "CANCELLED", "DIVERTED"
    ])


# ==============================================================================
# Knowledge Graph
# ==============================================================================
@dataclass
class KGConfig:
    entity_types: List[str] = field(default_factory=lambda: [
        "Flight", "Aircraft", "Airport", "Airline", "Weather", "TimeSlot"
    ])
    relation_types: List[str] = field(default_factory=lambda: [
        "delay_propagates_to",
        "departs_from", "arrives_at",
        "operated_by", "flown_by",
        "has_weather_origin", "has_weather_dest",
        "in_timeslot"
    ])
    # GAT (replaces R-GCN)
    gat_layers: int = 2
    gat_hidden_dim: int = 128      # Reduced from 256 to 128 (prevent overfitting)
    gat_out_dim: int = 384         # Reduced from 512 to 384 (prevent overfitting)
    gat_num_heads: int = 4         # Multi-head attention
    neighbor_samples: int = 10    # per layer per node
    
    # Time encoding
    time_encoding_dim: int = 64
    time_encoding_max_period: int = 365  # days


# ==============================================================================
# Stage 1
# ==============================================================================
@dataclass
class Stage1Config:
    feature_gate_bottleneck: int = 16   # 64 -> 16 -> 64
    task_attention_hidden: int = 16     # MLP(64 -> 16 -> 2)
    lr: float = 5e-4  # Reduced from 1e-3 to 5e-4, GAT + multitask more stable
    weight_decay: float = 0.03  # Increased from 0.01 to 0.03 (prevent overfitting)
    batch_size: int = 1024  # Restored from 512 to 1024 (dropout + wd provide sufficient regularization)
    epochs: int = 10
    patience: int = 5
    gradient_clip: float = 1.0
    reg_loss_weight: float = 1.0  # Regression loss weight, reduced to 1.0 to avoid over-focusing on regression
    reg_target_max: float = 120.0  # Regression target normalization max value (120 minutes)


# ==============================================================================
# Stage 2 (LLM Fine-tuning)
# ==============================================================================
@dataclass
class Stage2Config:
    alignment_dim: int = 1536      # Qwen2-1.5B hidden size
    lambda_init: float = 0.1       # scaling coefficient initial value
    lambda_min: float = 0.05       # sigmoid lower bound
    lambda_max: float = 0.55       # sigmoid upper bound

    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # Task Attention (3-way)
    task_attention_hidden: int = 128  # MLP(1536 -> 128 -> 3)

    # Loss weights
    loss_cls_weight: float = 1.0      # fixed base (will be learned by TaskAttn)
    loss_reg_weight: float = 1.0      # fixed base
    loss_KG_lambda: float = 1.0       # weight within Task Attention
    loss_KG_margin: float = 0.8
    loss_gen_gamma: float = 0.01      # fixed, not learned

    # Optimizer
    lr: float = 2e-4
    lr_lora: float = 2e-4
    lr_other: float = 1e-3
    weight_decay: float = 0.01
    batch_size: int = 8
    gradient_accumulation_steps: int = 4  # effective batch = 32
    epochs: int = 10
    patience: int = 3
    gradient_clip: float = 1.0

    # LLM specific (for train_stage2.py)
    llm_lr: float = 1e-4
    llm_batch_size: int = 4
    llm_epochs: int = 5  # Train to 5 epochs total

    # Generation
    gen_max_length: int = 64
    gen_temperature: float = 0.7


# ==============================================================================
# Training (general)
# ==============================================================================
@dataclass
class TrainConfig:
    seed: int = 42
    fp16: bool = False  # GAT + multitask unstable, mixed precision disabled
    bf16: bool = False          # RTX 3090 supports both; fp16 is faster
    num_workers: int = 4
    device: str = "cuda"
    num_gpus: int = 1            # 1x RTX 3090
    deepspeed_config: str = ""   # "zeRO-2" or "zeRO-3"


# ==============================================================================
# Master config
# ==============================================================================
@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    data: DataConfig = field(default_factory=DataConfig)
    kg: KGConfig = field(default_factory=KGConfig)
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    train: TrainConfig = field(default_factory=TrainConfig)


CONFIG = Config()
