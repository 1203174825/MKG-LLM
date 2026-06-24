import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    def __init__(self, dim: int = 512):
        super().__init__()
        self.gate = nn.Linear(dim * 2, 1)

    def forward(self, h_main: torch.Tensor,
                h_chain: torch.Tensor) -> tuple:
        g = torch.sigmoid(self.gate(torch.cat([h_main, h_chain], dim=-1)))
        h = g * h_main + (1 - g) * h_chain
        return h, g
