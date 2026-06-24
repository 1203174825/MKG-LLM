"""
Relational Graph Attention Network (RGAT) with edge-feature-aware attention.
Combines R-GCN's relation-aware transformations with GAT's attention mechanism,
now incorporating edge features (INTERVAL_TIME, aircraft info) into attention.
h_i^(l+1) = σ( Σ_{r∈R} Σ_{j∈N_i^r} α_{ij}^r(e_ij) · W_r · h_j^(l) )
Memory-optimized for large graphs with DGL block support.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn


def edge_softmax_with_mask(block, logits, mask):
    from dgl.nn.pytorch import edge_softmax
    return edge_softmax(block, logits)


class RGATLayer(nn.Module):
    """Relational GAT layer with edge-feature-aware attention."""

    def __init__(self, in_dim: int, out_dim: int, num_etypes: int = 9,
                 dropout: float = 0.1, negative_slope: float = 0.2,
                 use_edge_feat: bool = True, edge_feat_dim: int = 1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_etypes = num_etypes
        self.negative_slope = negative_slope
        self.use_edge_feat = use_edge_feat

        self.fc_src = nn.ParameterList([
            nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_etypes)
        ])
        self.fc_dst = nn.ParameterList([
            nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_etypes)
        ])

        self.attn_l = nn.ParameterList([
            nn.Linear(out_dim, 1, bias=False) for _ in range(num_etypes)
        ])
        self.attn_r = nn.ParameterList([
            nn.Linear(out_dim, 1, bias=False) for _ in range(num_etypes)
        ])

        if use_edge_feat:
            self.edge_proj = nn.ParameterList([
                nn.Linear(edge_feat_dim, out_dim, bias=False) for _ in range(num_etypes)
            ])
            self.attn_e = nn.ParameterList([
                nn.Linear(out_dim, 1, bias=False) for _ in range(num_etypes)
            ])

        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(out_dim)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for fc in self.fc_src:
            nn.init.xavier_normal_(fc.weight, gain=gain)
        for fc in self.fc_dst:
            nn.init.xavier_normal_(fc.weight, gain=gain)
        for attn in self.attn_l:
            nn.init.xavier_normal_(attn.weight, gain=gain)
        for attn in self.attn_r:
            nn.init.xavier_normal_(attn.weight, gain=gain)
        if self.use_edge_feat:
            for proj in self.edge_proj:
                nn.init.xavier_normal_(proj.weight, gain=gain)
            for attn_e in self.attn_e:
                nn.init.xavier_normal_(attn_e.weight, gain=gain)

    def forward(self, block, etypes, edge_feat=None):
        with block.local_scope():
            feat_src = block.srcdata['h']
            feat_dst = block.dstdata['h']

            h_out = torch.zeros(block.number_of_dst_nodes(), self.out_dim,
                               device=feat_dst.device, dtype=feat_dst.dtype)

            for r in range(self.num_etypes):
                rel_mask = (etypes == r)
                if not rel_mask.any():
                    continue

                h_src_r = self.fc_src[r](feat_src)
                h_dst_r = self.fc_dst[r](feat_dst)

                # Standard GAT attention: el uses SRC features, er uses DST features
                # Attention score e_ij = el(src_i) + er(dst_j)
                el_r = self.attn_l[r](h_src_r)
                er_r = self.attn_r[r](h_dst_r)

                block.srcdata.update({'ft_r': h_src_r, 'el_r': el_r})
                block.dstdata.update({'er_r': er_r})

                block.apply_edges(fn.u_add_v('el_r', 'er_r', 'e_r'))
                e_r = block.edata.pop('e_r')

                if self.use_edge_feat and edge_feat is not None:
                    edge_h = self.edge_proj[r](edge_feat)
                    e_e = self.attn_e[r](edge_h)
                    e_r = e_r + e_e

                e_r = F.leaky_relu(e_r, self.negative_slope)
                e_r[~rel_mask] = float('-inf')

                block.edata['a_r'] = edge_softmax_with_mask(block, e_r, rel_mask)
                block.edata['a_r'] = torch.nan_to_num(block.edata['a_r'], nan=0.0)
                block.edata['a_r'] = self.dropout(block.edata['a_r'])

                block.update_all(fn.u_mul_e('ft_r', 'a_r', 'm'), fn.sum('m', 'h_r'))
                h_out = h_out + block.dstdata['h_r']

            out = h_out + torch.mean(torch.stack([fc(feat_dst) for fc in self.fc_dst]), dim=0)
            out = self.batch_norm(out)
            return out


class MultiHeadRGAT(nn.Module):
    """Multi-head RGAT layer with edge-feature support."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4,
                 num_etypes: int = 9, dropout: float = 0.1,
                 negative_slope: float = 0.2, use_edge_feat: bool = True,
                 edge_feat_dim: int = 1):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim

        self.heads = nn.ModuleList([
            RGATLayer(in_dim, out_dim, num_etypes, dropout, negative_slope,
                      use_edge_feat, edge_feat_dim)
            for _ in range(num_heads)
        ])

    def forward(self, block, etypes, edge_feat=None):
        head_outs = []
        for head in self.heads:
            with block.local_scope():
                block.srcdata['h'] = block.srcdata['h'].clone()
                block.dstdata['h'] = block.dstdata['h'].clone()
                head_out = head(block, etypes, edge_feat)
            head_outs.append(head_out)
        return torch.cat(head_outs, dim=-1)


class RGATModel(nn.Module):
    """
    Full RGAT model with edge-feature-aware attention.
    Architecture: RGAT (2 layers, 4 heads) with INTERVAL_TIME edges.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 512,
                 num_layers: int = 2, num_heads: int = 4, num_etypes: int = 9,
                 dropout: float = 0.1, negative_slope: float = 0.2,
                 use_edge_feat: bool = True, edge_feat_dim: int = 1):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_edge_feat = use_edge_feat

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(dropout),
        )

        self.rgat_layers = nn.ModuleList()

        if num_layers == 1:
            # Single layer: hidden → out directly, 1 head
            self.rgat_layers.append(
                MultiHeadRGAT(hidden_dim, out_dim, 1, num_etypes,
                             dropout, negative_slope, use_edge_feat, edge_feat_dim)
            )
        else:
            # First layer: hidden → hidden, num_heads heads
            self.rgat_layers.append(
                MultiHeadRGAT(hidden_dim, hidden_dim, num_heads, num_etypes,
                             dropout, negative_slope, use_edge_feat, edge_feat_dim)
            )
            # Middle layers
            for _ in range(num_layers - 2):
                self.rgat_layers.append(
                    MultiHeadRGAT(hidden_dim * num_heads, hidden_dim, num_heads,
                                 num_etypes, dropout, negative_slope,
                                 use_edge_feat, edge_feat_dim)
                )
            # Last layer: hidden*heads → out, 1 head
            self.rgat_layers.append(
                MultiHeadRGAT(hidden_dim * num_heads, out_dim, 1, num_etypes,
                             dropout, negative_slope, use_edge_feat, edge_feat_dim)
            )

        self.output_proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(dropout),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_proj:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def forward(self, blocks, feat, etypes_list, time_enc, edge_feat_list=None):
        h = self.input_proj(feat)

        for i, (block, etypes) in enumerate(zip(blocks, etypes_list)):
            rgat_layer = self.rgat_layers[i]

            n_src = block.number_of_src_nodes()
            n_dst = block.number_of_dst_nodes()

            block.srcdata['h'] = h
            block.dstdata['h'] = h[-n_dst:]

            ef = edge_feat_list[i] if edge_feat_list else None
            h = rgat_layer(block, etypes, ef)

            if i < len(self.rgat_layers) - 1:
                h = F.leaky_relu(h, negative_slope=0.2)

        h = self.output_proj(h)
        return h

    def forward_full(self, g, feat, time_enc, edge_feat=None):
        h = self.input_proj(feat)

        if dgl.ETYPE in g.edata:
            etypes = g.edata[dgl.ETYPE]
        else:
            etypes = torch.zeros(g.num_edges(), dtype=torch.long, device=feat.device)

        for i, rgat_layer in enumerate(self.rgat_layers):
            g.srcdata['h'] = h
            g.dstdata['h'] = h
            h = rgat_layer(g, etypes, edge_feat)
            if i < len(self.rgat_layers) - 1:
                h = F.leaky_relu(h, negative_slope=0.2)

        h = self.output_proj(h)
        return h
