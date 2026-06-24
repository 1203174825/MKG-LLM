"""
Stage 1 (Step 7 - Single Shared GAT): 
  One shared-parameter GAT processes all three graph modalities.
  Fusion: learnable weighted fusion (3 params) + FeatureGate + PredHeads.

Supports:
  - feature_mask_indices: zero out specified feature dims for cleaner ablation
  - use_mlp_only: replace entire KG+GAT pipeline with a simple MLP (no-graph baseline)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from src.models.gat import RGATModel
from src.models.feature_gate import FeatureGate
from src.models.prediction_heads import Stage1PredictionHeads
from src.utils.config import CONFIG


class Stage1Model(nn.Module):
    """Stage 1: Single shared GAT (25-dim) + Learnable weighted fusion + Adapters.

    Architecture:
      feat ── adapter_main ──┐
      feat ── adapter_chain ──┼── Shared RGATModel(25, 512, 8 etypes)
      feat ── adapter_network┘                (≈388K)
      → Learnable Weighted Fusion (3 params) → FeatureGate → PredHeads

    Ablation modes:
      - feature_mask_indices: zero out specific feature dims
      - use_mlp_only: skip all graphs, use MLP on raw features (no-KG baseline)
    """

    def __init__(self, skip_modalities=None, use_mean_fusion=False,
                 feature_mask_indices=None, use_mlp_only=False,
                 delay_reg_weight=5.0):
        super().__init__()
        self.skip_modalities = skip_modalities or []
        self.use_mean_fusion = use_mean_fusion
        self.feature_mask_indices = feature_mask_indices or []
        self.use_mlp_only = use_mlp_only
        self.delay_reg_weight = delay_reg_weight
        cfg = CONFIG.kg

        self.feat_proj = nn.Sequential(
            nn.Linear(25, 64), nn.ReLU(),
            nn.Linear(64, 25),
        )

        if use_mlp_only:
            self.mlp = nn.Sequential(
                nn.Linear(25, 32),
                nn.ReLU(),
                nn.Linear(32, cfg.gat_out_dim),
            )
        else:
            self.shared_gat = RGATModel(
                in_dim=25,
                hidden_dim=cfg.gat_hidden_dim,
                out_dim=cfg.gat_out_dim,
                num_layers=cfg.gat_layers,
                num_heads=cfg.gat_num_heads,
                num_etypes=len(cfg.relation_types),
                dropout=0.1,
                negative_slope=0.2,
                use_edge_feat=True,
                edge_feat_dim=1,
            )

            if not use_mean_fusion:
                self.fusion_logits = nn.Parameter(torch.zeros(3))
                self.adapter_delta_main = nn.Sequential(
                    nn.Linear(25, 8, bias=False), nn.ReLU(), nn.Linear(8, 25, bias=False),
                )
                self.adapter_delta_chain = nn.Sequential(
                    nn.Linear(25, 8, bias=False), nn.ReLU(), nn.Linear(8, 25, bias=False),
                )
                self.adapter_delta_network = nn.Sequential(
                    nn.Linear(25, 8, bias=False), nn.ReLU(), nn.Linear(8, 25, bias=False),
                )
                nn.init.zeros_(self.adapter_delta_main[2].weight)
                nn.init.zeros_(self.adapter_delta_chain[2].weight)
                nn.init.zeros_(self.adapter_delta_network[2].weight)

        self.feature_gate = FeatureGate(
            dim=cfg.gat_out_dim, bottleneck_ratio=2,
        )

        self.pred_heads = Stage1PredictionHeads(input_dim=cfg.gat_out_dim)
        self.log_var_cls = nn.Parameter(torch.tensor(0.0))
        self.log_var_reg = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(0.3)

    def _apply_feature_mask(self, feat):
        if self.feature_mask_indices:
            feat = feat.clone()
            feat[..., self.feature_mask_indices] = 0.0
        return feat

    def _pad_edge_feat(self, ef, target_dim=1):
        if ef is None:
            return None
        if ef.shape[-1] < target_dim:
            pad = torch.zeros(*ef.shape[:-1], target_dim - ef.shape[-1],
                              device=ef.device, dtype=ef.dtype)
            return torch.cat([ef, pad], dim=-1)
        return ef[..., :target_dim]

    def forward(self, blocks: list, feat: torch.Tensor,
                etypes_list: list, time_enc: torch.Tensor,
                target_idx: torch.Tensor = None,
                edge_feat_list: list = None,
                g_main=None,
                g_chain=None, chain_feat=None,
                g_network=None, network_feat=None,
                network_edge_feat=None,
                airport_flight_map=None,
                flight_nids=None) -> dict:
        
        # Apply feature masking before any GAT/Adapter processing
        feat = self._apply_feature_mask(feat)
        feat = self.feat_proj(feat)

        if self.use_mlp_only:
            h = self.mlp(feat)
            if target_idx is not None:
                h = h[target_idx]
            e_f = h
        else:
            w = None
            adapter_main = getattr(self, 'adapter_delta_main', None)
            adapter_chain = getattr(self, 'adapter_delta_chain', None)
            adapter_network = getattr(self, 'adapter_delta_network', None)
            if not self.use_mean_fusion:
                w = F.softmax(self.fusion_logits, dim=0)
            has_main = 'main' not in self.skip_modalities
            has_chain = 'chain' not in self.skip_modalities
            has_network = 'network' not in self.skip_modalities

            e_f = None
            w_sum = 0.0
            fusion_cnt = 0

            # 1. Main KG → shared GAT
            if has_main:
                feat_main = feat + adapter_main(feat) if adapter_main is not None else feat
                if blocks is not None:
                    edge_feat_padded = [self._pad_edge_feat(ef) for ef in (edge_feat_list or [])]
                    h_main = self.shared_gat(blocks, feat_main, etypes_list, time_enc,
                                             edge_feat_list=edge_feat_padded)
                else:
                    h_main = self.shared_gat.forward_full(
                        g_main, feat_main, time_enc,
                        edge_feat=self._pad_edge_feat(g_main.edata.get('feat') if g_main is not None else None))

                h_main_tgt = h_main[target_idx] if target_idx is not None else h_main
                if self.use_mean_fusion:
                    e_f = h_main_tgt
                    fusion_cnt = 1
                else:
                    e_f = w[0] * h_main_tgt
                    w_sum = w[0]

            # 2. Chain graph → shared GAT (IDENTICAL to src)
            #    Uses forward_full (no neighbor sampling): chain graph is small by design
            if has_chain and g_chain is not None and chain_feat is not None and flight_nids is not None:
                chain_feat_ad = self._apply_feature_mask(chain_feat)
                chain_feat_ad = self.feat_proj(chain_feat_ad)
                chain_feat_ad = chain_feat_ad + adapter_chain(chain_feat_ad) if adapter_chain is not None else chain_feat_ad
                h_chain_full = self.shared_gat.forward_full(
                    g_chain, chain_feat_ad, time_enc,
                    edge_feat=self._pad_edge_feat(g_chain.edata.get('feat')))
                h_c = h_chain_full[flight_nids]
                if self.use_mean_fusion:
                    e_f = h_c if e_f is None else e_f + h_c
                    fusion_cnt += 1
                else:
                    e_f = w[1] * h_c if e_f is None else e_f + w[1] * h_c
                    w_sum = w_sum + w[1]

            # 3. Network graph → shared GAT
            if (has_network and g_network is not None and network_feat is not None
                    and airport_flight_map is not None and flight_nids is not None):
                network_feat_ad = self._apply_feature_mask(network_feat)
                network_feat_ad = self.feat_proj(network_feat_ad)
                network_feat_ad = network_feat_ad + adapter_network(network_feat_ad) if adapter_network is not None else network_feat_ad
                h_network_full = self.shared_gat.forward_full(
                    g_network, network_feat_ad, time_enc,
                    edge_feat=self._pad_edge_feat(network_edge_feat))
                flight_offset = airport_flight_map.get('flight_node_offset', 0)
                h_n = h_network_full[flight_nids + flight_offset]
                if self.use_mean_fusion:
                    e_f = h_n if e_f is None else e_f + h_n
                    fusion_cnt += 1
                else:
                    e_f = w[2] * h_n if e_f is None else e_f + w[2] * h_n
                    w_sum = w_sum + w[2]

            if self.use_mean_fusion:
                e_f = e_f / fusion_cnt
            else:
                e_f = e_f / max(w_sum, 1e-8)

        e_f_gated = self.feature_gate(e_f)
        e_f_gated = self.dropout(e_f_gated)
        cls_logits, reg_pred = self.pred_heads(e_f_gated)

        return {
            "e_f": e_f,
            "e_f_gated": e_f_gated,
            "cls_logits": cls_logits,
            "reg_pred": reg_pred,
            "gate_chain": torch.ones(e_f.shape[0], 1, device=e_f.device) * 0.5,
            "gate_network": torch.ones(e_f.shape[0], 1, device=e_f.device) * 0.5,
        }

    def forward_tabular(self, combined_feats: torch.Tensor) -> tuple:
        combined_feats = self.feat_proj(combined_feats)
        if self.use_mlp_only:
            h = self.mlp(combined_feats)
        else:
            h = self.shared_gat.output_proj(combined_feats)
        e_f_gated = self.feature_gate(h)
        e_f_gated = self.dropout(e_f_gated)
        cls_logits, reg_pred = self.pred_heads(e_f_gated)
        return cls_logits, reg_pred

    def compute_loss(self, cls_logits: torch.Tensor, reg_pred: torch.Tensor,
                     cls_target: torch.Tensor, reg_target: torch.Tensor,
                     batch_idx: int = 0) -> torch.Tensor:
        pos_weight = 4.5

        if cls_target.dim() == 1:
            cls_target = cls_target.unsqueeze(-1)
        bce = F.binary_cross_entropy_with_logits(
            cls_logits, cls_target,
            pos_weight=torch.tensor(pos_weight, device=cls_logits.device)
        )
        if reg_target.dim() == 1:
            reg_target = reg_target.unsqueeze(-1)

        reg_weights = torch.where(cls_target > 0.5,
                                  torch.tensor(self.delay_reg_weight, device=reg_pred.device),
                                  torch.tensor(1.0, device=reg_pred.device))
        per_sample_huber = F.smooth_l1_loss(reg_pred, reg_target, reduction='none')
        huber = (per_sample_huber * reg_weights).mean()

        loss_cls = (1.0 / (2.0 * torch.exp(self.log_var_cls))) * bce + self.log_var_cls
        loss_reg = (1.0 / (2.0 * torch.exp(self.log_var_reg))) * huber + self.log_var_reg

        loss = loss_cls + loss_reg

        weight_cls = 1.0 / torch.exp(self.log_var_cls)
        weight_reg = 1.0 / torch.exp(self.log_var_reg)
        total_weight = weight_cls + weight_reg
        alpha_cls = (weight_cls / total_weight).item()
        alpha_reg = (weight_reg / total_weight).item()

        return loss, {
            "loss": loss.item(),
            "bce": bce.item(),
            "huber": huber.item(),
            "alpha_cls": alpha_cls,
            "alpha_reg": alpha_reg,
            "log_var_cls": self.log_var_cls.item(),
            "log_var_reg": self.log_var_reg.item(),
        }