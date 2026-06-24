"""
Aeolus_V2 dataset loader.
Loads tri-modal data (Tabular, Chain, Network) from daily files.
"""
import os
import glob
import calendar
import torch
import dgl
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from src.utils.config import CONFIG


def get_daily_files(data_dir: str, year: int, month: int) -> List[str]:
    """Get all daily data files for a given year/month."""
    pattern = os.path.join(data_dir, str(year), f"{month:02d}", "*")
    files = sorted(glob.glob(pattern))
    return files


def parse_date_from_filename(filename: str) -> str:
    """Extract YYMMDD from filename."""
    base = os.path.basename(filename)
    # e.g., flight_with_weather_160101.csv -> 160101
    parts = base.split("_")
    for p in parts:
        if p.isdigit() and len(p) == 6:
            return p
    return ""


# =============================================================================
# Tabular Dataset
# =============================================================================
class TabularDataset(Dataset):
    """Loads Flight_Tabular CSV files for given date range."""

    def __init__(self, year: int, months: Optional[List[int]] = None,
                 transform=None, is_train: bool = True):
        self.tabular_dir = CONFIG.paths.tabular_dir
        self.year = year
        self.months = months or list(range(1, 13))
        self.transform = transform
        self.is_train = is_train
        self.config = CONFIG.data

        self.files: List[str] = []
        for m in self.months:
            self.files.extend(get_daily_files(self.tabular_dir, year, m))

        # Pre-scan: count total rows for indexing
        self._file_row_counts: List[int] = []
        self._file_indices: List[Tuple[int, int]] = []  # (file_idx, row_idx)
        self.data_cache: Dict[str, pd.DataFrame] = {}  # optional cache

    def __len__(self) -> int:
        return len(self._file_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_idx, row_idx = self._file_indices[idx]
        df = self._load_file(file_idx)
        row = df.iloc[row_idx]

        # Extract features (leakage-free)
        cat_feats = torch.tensor([
            self._encode_cat(col, row[col]) for col in self.config.cat_cols
        ], dtype=torch.long)
        cont_feats = torch.tensor(
            [row[col] for col in self.config.cont_cols], dtype=torch.float32
        )
        target_delay = torch.tensor(row[self.config.target_col], dtype=torch.float32)
        delay_label = (target_delay >= self.config.delay_threshold).float()

        return {
            "cat_feats": cat_feats,          # (8,)
            "cont_feats": cont_feats,        # (14,)
            "delay": target_delay,           # (1,)
            "label": delay_label,            # (1,)
        }

    def _load_file(self, file_idx: int) -> pd.DataFrame:
        fpath = self.files[file_idx]
        if fpath in self.data_cache:
            return self.data_cache[fpath]
        df = pd.read_csv(fpath)
        # Drop forbidden columns
        df = df.drop(columns=[c for c in self.config.forbidden_cols if c in df.columns])
        self.data_cache[fpath] = df
        return df

    def _encode_cat(self, col: str, val) -> int:
        """Simple hash-based category encoding (replaced by learned embedding)."""
        return hash(f"{col}_{val}") % 100000

    def build_index(self):
        """Build file/row index for all data."""
        self._file_indices = []
        for fi in range(len(self.files)):
            df = self._load_file(fi)
            for ri in range(len(df)):
                self._file_indices.append((fi, ri))

    def get_daily_batches(self, year: int, month: int, day: int) -> pd.DataFrame:
        """Get a single day's full DataFrame (for KG construction)."""
        fname = f"flight_with_weather_{year % 100:02d}{month:02d}{day:02d}.csv"
        fpath = os.path.join(self.tabular_dir, str(year), f"{month:02d}", fname)
        if not os.path.exists(fpath):
            return pd.DataFrame()
        df = pd.read_csv(fpath)
        df = df.drop(columns=[c for c in self.config.forbidden_cols if c in df.columns], errors='ignore')
        return df


# =============================================================================
# Chain Dataset (Flight_Chain .pt files)
# =============================================================================
class ChainDataset(Dataset):
    """Loads daily Flight_Chain .pt files."""

    def __init__(self, year: int, months: Optional[List[int]] = None):
        self.chain_dir = CONFIG.paths.chain_dir
        self.year = year
        self.months = months or list(range(1, 13))

        self.files: List[str] = []
        for m in self.months:
            self.files.extend(get_daily_files(self.chain_dir, year, m))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict:
        """Return one day's chain data as a dict."""
        data = torch.load(self.files[idx], map_location="cpu")
        # data structure (from Aeolus_V2 build_chain_fast.py):
        #   dense_feat: (N, max_seq_len, 7)
        #   sparse_feat: (N, max_seq_len, 9)  -- last dim = TAIL_NUM_ENC
        #   labels: (N, max_seq_len)  -- DEP_DELAY per flight in chain
        #   delays: (N, max_seq_len)  -- same as labels
        #   valid_len: (N,)  -- actual chain length
        return data


# =============================================================================
# Network Dataset (Flight_Network .dgl files)
# =============================================================================
class NetworkDataset(Dataset):
    """Loads daily Flight_Network .dgl files."""

    def __init__(self, year: int, months: Optional[List[int]] = None):
        self.network_dir = CONFIG.paths.network_dir
        self.year = year
        self.months = months or list(range(1, 13))

        self.files: List[str] = []
        for m in self.months:
            self.files.extend(get_daily_files(self.network_dir, year, m))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dgl.DGLGraph:
        """Return one day's DGL graph."""
        g = dgl.load_graphs(self.files[idx])[0][0]
        return g


# =============================================================================
# Unified Daily Data Bundle
# =============================================================================
@dataclass
class DailyData:
    """All data for a single day, across all three modalities."""
    date: str                       # YYMMDD
    year: int
    month: int
    day: int
    tabular: pd.DataFrame           # Raw tabular data
    chain: Optional[Dict] = None    # Chain .pt data (or None)
    network: Optional[dgl.DGLGraph] = None  # Network .dgl graph (or None)


class AeolusDataLoader:
    """Orchestrates loading of all three modalities for a date range."""

    def __init__(self, year: int, months: Optional[List[int]] = None):
        self.year = year
        self.months = months or list(range(1, 13))

    def get_daily(self, month: int, day: int) -> DailyData:
        """Get all data for a specific day."""
        date_str = f"{self.year % 100:02d}{month:02d}{day:02d}"

        # Tabular
        tabular = TabularDataset(self.year, [month]).get_daily_batches(
            self.year, month, day
        )

        # Chain
        chain_path = os.path.join(
            CONFIG.paths.chain_dir, str(self.year), f"{month:02d}",
            f"flight_chain_{date_str}.pt"
        )
        chain = torch.load(chain_path, map_location="cpu", weights_only=True) if os.path.exists(chain_path) else None

        # Network
        network_path = os.path.join(
            CONFIG.paths.network_dir, str(self.year), f"{month:02d}",
            f"flight_network_{date_str}.dgl"
        )
        network = None
        if os.path.exists(network_path):
            network = dgl.load_graphs(network_path)[0][0]

        return DailyData(
            date=date_str, year=self.year, month=month, day=day,
            tabular=tabular, chain=chain, network=network
        )

    def iter_days(self):
        """Iterate over all days in the configured range."""
        for m in self.months:
            _, days_in_month = calendar.monthrange(self.year, m)
            for d in range(1, days_in_month + 1):
                yield self.get_daily(m, d)
