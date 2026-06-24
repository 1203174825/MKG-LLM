import torch
import torch.nn as nn
import torch.nn.functional as F


class ChainDelayEncoder(nn.Module):

    def __init__(self, max_seq: int = 6, hidden: int = 64,
                 gru_layers: int = 1, output_dim: int = 64):
        super().__init__()
        self.max_seq = max_seq
        self.output_dim = output_dim

        self.input_proj = nn.Linear(1, hidden)
        self.gru = nn.GRU(hidden, hidden, gru_layers, batch_first=True)
        self.output_proj = nn.Linear(hidden, output_dim)

    def forward(self, delays: torch.Tensor,
                valid_len: torch.Tensor) -> torch.Tensor:
        N, S = delays.shape[:2]

        h = self.input_proj(delays)  # (N, S, 64)

        valid_len_cpu = valid_len.cpu().clamp(max=self.max_seq)
        packed = nn.utils.rnn.pack_padded_sequence(
            h, valid_len_cpu, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=S)

        return self.output_proj(out)  # (N, S, 64)
