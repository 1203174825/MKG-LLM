"""
R-GCN with time-aware relation transformation and neighbor sampling.
Basis decomposition for parameter efficiency. Mean aggregation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import RelGraphConv
from src.utils.config import CONFIG


class TimeAwareRGCNSingleLayer(nn.Module):
    """
    Single R-GCN layer with time-aware relation transformation.
    W_r(t) = W_r_base + TimeEncode(t) projection
    """

    def __init__(self, in_dim: int, out_dim: int, num_rels: int, num_bases: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases

        # Standard R-GCN conv (uses basis decomposition internally)
        self.conv = RelGraphConv(
            in_feat=in_dim,
            out_feat=out_dim,
            num_rels=num_rels,
            regularizer="basis",
            num_bases=num_bases,
            bias=True,
            activation=None,
            self_loop=True,
            dropout=0.1,
        )
        # Time encoding projection: 64-dim time_enc -> (out_dim,)
        self.time_proj = nn.Linear(64, out_dim, bias=False)

    def forward(self, g, feat: torch.Tensor,
                time_enc: torch.Tensor,
                use_block_etypes: bool = True) -> torch.Tensor:
        """
        Args:
            g: DGL graph or block (homogeneous with etype edge data)
            feat: node features (N, in_dim)
            time_enc: time encoding (64,)
        Returns:
            updated node features (N, out_dim)
        """
        # Extract etypes from edata (homogeneous graph with etype data)
        if dgl.ETYPE in g.edata:
            etypes = g.edata[dgl.ETYPE]
        else:
            etypes = torch.zeros(g.num_edges(), dtype=torch.long, device=feat.device)
        out = self.conv(g, feat, etypes)
        
        # Time-aware bias
        time_bias = self.time_proj(time_enc)  # (out_dim,)
        out = out + time_bias.unsqueeze(0)     # (N, out_dim)
        return F.relu(out)


class TimeAwareRGCN(nn.Module):
    """
    Multi-layer time-aware R-GCN with neighbor sampling support.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_rels: int, num_bases: int, num_layers: int = 2):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        # Input layer: in_dim -> hidden_dim
        self.layers.append(TimeAwareRGCNSingleLayer(
            in_dim, hidden_dim, num_rels, num_bases
        ))
        # Hidden layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 1):
            self.layers.append(TimeAwareRGCNSingleLayer(
                hidden_dim, hidden_dim, num_rels, num_bases
            ))
        # Output projection (optional): hidden_dim -> out_dim
        self.out_proj = nn.Linear(hidden_dim, out_dim) if hidden_dim != out_dim else nn.Identity()

    def forward(self, blocks: list, feats: list,
                etypes_list: list, norms_list: list,
                time_enc: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with neighbor sampling (GraphSAGE-style).

        Args:
            blocks: list of DGL blocks from sampler
            feats: list of node features per layer
            etypes_list: list of edge types per layer (unused, handled internally)
            norms_list: list of edge norms per layer (unused, handled internally)
            time_enc: time encoding (64,)
        Returns:
            output features (batch_nodes, out_dim)
        """
        h = feats[0]
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h, time_enc)
        h = self.out_proj(h)
        return h

    def forward_full(self, g, feat: torch.Tensor,
                     time_enc: torch.Tensor) -> torch.Tensor:
        """
        Full-graph forward pass (for small graphs or inference).
        """
        h = feat
        for layer in self.layers:
            h = layer(g, h, time_enc)
        h = self.out_proj(h)
        return h
