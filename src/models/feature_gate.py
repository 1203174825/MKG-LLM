"""
Feature Gating Network: dimension-wise adaptive feature selection.
Replaces SENet GAP channel attention (GAP degenerates on single vectors).
"""
import torch
import torch.nn as nn


class FeatureGate(nn.Module):
    """
    Feature Gating Network.
    Learns a per-dimension gate vector g in (0,1) and applies: x_out = x * g.

    Unlike SENet channel attention, this operates directly on dimensions
    without GAP pooling, avoiding the identity-map problem for single vectors.
    """

    def __init__(self, dim: int = 64, bottleneck_ratio: int = 2):
        super().__init__()
        bottleneck = dim // bottleneck_ratio  # e.g., 64/2 = 32

        self.gate = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, dim),
            nn.Sigmoid(),
        )
        # Initialize: bias the last layer so initial gates are near 0.5
        self._init_weights()

    def _init_weights(self):
        for m in self.gate.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, dim) node embeddings
        Returns:
            (batch, dim) gated embeddings
        """
        gate = self.gate(x)  # (batch, dim)
        return x * gate


class SENetChannelAttention(nn.Module):
    """
    SENet-style channel attention with GAP.
    Included for ablation: GAP degenerates on single vectors (identity map).
    """

    def __init__(self, dim: int = 64, bottleneck_ratio: int = 4):
        super().__init__()
        bottleneck = dim // bottleneck_ratio
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # GAP: degenerates for single vectors
            nn.Flatten(),
            nn.Linear(dim, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, dim)
        # GAP with AdaptiveAvgPool1d(1) on (B, C, L=1): pools each channel over L=1.
        # Since L=1, this is IDENTITY per channel. The bottleneck MLP does all the work.
        # GAP layer is redundant but harmless (kept for fair ablation comparison).
        gate = self.se(x.unsqueeze(-1))
        return x * gate
