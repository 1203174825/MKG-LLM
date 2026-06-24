"""
Alignment layer + Scaled Residual Injection for Graph Token -> LLM.
Handles: distribution shift (LayerNorm + lambda), RoPE compatibility, no [CLS] token.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AlignmentLayer(nn.Module):
    """
    Projects KG embeddings (64-dim) to LLM space (1536-dim for Qwen2-1.5B).
    Includes LayerNorm for distribution matching.
    """

    def __init__(self, kg_dim: int = 64, llm_dim: int = 1536):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(kg_dim, llm_dim),
            nn.LayerNorm(llm_dim),  # Normalize to match Qwen2 embedding distribution
        )

    def forward(self, e_f: torch.Tensor) -> torch.Tensor:
        """
        Args:
            e_f: (batch, kg_dim) R-GCN output (original, not gated)
        Returns:
            (batch, llm_dim) aligned KG embedding
        """
        return self.proj(e_f)


class ScaledResidualInjector(nn.Module):
    """
    Injects KG knowledge into LLM via scaled residual addition to the
    last token's embedding (Qwen2 has no [CLS] token).


    def __init__(self, llm_dim: int = 1536,
                 lambda_init: float = 0.1,
                 lambda_min: float = 0.05,
                 lambda_max: float = 0.55):
        super().__init__()
        self.llm_dim = llm_dim
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.lambda_range = lambda_max - lambda_min

        # Learnable scaling coefficient (in logit space)
        # lambda_raw -> sigmoid -> scale to (lambda_min, lambda_max)
        init_logit = self._inverse_sigmoid(
            (lambda_init - lambda_min) / self.lambda_range
        )
        self.lambda_raw = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))

    @staticmethod
    def _inverse_sigmoid(y: float) -> float:
        """Compute logit for a given probability."""
        eps = 1e-7
        y = max(eps, min(1 - eps, y))
        return float(torch.tensor(y).logit())

    @property
    def lambda_scale(self) -> torch.Tensor:
        """Get constrained lambda in (lambda_min, lambda_max)."""
        return torch.sigmoid(self.lambda_raw) * self.lambda_range + self.lambda_min

    def forward(self, input_embeds: torch.Tensor,
                e_proj: torch.Tensor) -> torch.Tensor:
        """
        Apply scaled residual injection.

        Args:
            input_embeds: (batch, seq_len, llm_dim) LLM input embeddings
            e_proj: (batch, llm_dim) aligned KG embedding
        Returns:
            modified input_embeds with injection applied to last token
        """
        lam = self.lambda_scale  # scalar
        # Add to last token's embedding
        input_embeds[:, -1, :] = input_embeds[:, -1, :] + lam * e_proj
        return input_embeds


class KGAlignmentLoss(nn.Module):
    """
    Margin cosine loss for KG-LLM alignment.
    Both representations are explicitly L2-normalized before cosine computation
    to ensure numerical stability.
    """

    def __init__(self, margin: float = 0.8, llm_dim: int = 1536):
        super().__init__()
        self.margin = margin
        # Projection for KG embedding to match h_last dimension
        self.proj = nn.Linear(64, llm_dim, bias=False)

    def forward(self, h_last: torch.Tensor, e_f: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_last: (batch, llm_dim) LLM last token hidden state
            e_f: (batch, 64) KG embedding
        Returns:
            L_KG: scalar margin cosine loss
        """
        # Project KG to LLM space
        e_proj = self.proj(e_f)  # (batch, llm_dim)

        # Explicit L2 normalization BEFORE cosine
        h_norm = F.normalize(h_last, p=2, dim=-1)
        e_norm = F.normalize(e_proj, p=2, dim=-1)

        # Cosine similarity in [-1, 1]
        cos_sim = (h_norm * e_norm).sum(dim=-1)  # (batch,)

        # Margin loss: only penalize when cos_sim < margin
        loss = F.relu(self.margin - cos_sim)  # (batch,)
        return loss.mean()
