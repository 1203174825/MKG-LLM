"""
Prediction heads for classification and regression tasks.
Uses last token hidden state from Qwen2 (no [CLS] token).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    """Binary classification: on-time (<15min) vs. delayed (>=15min)."""

    def __init__(self, input_dim: int = 1536, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (batch, input_dim) last token hidden state
        Returns:
            logits: (batch, 1)
        """
        return self.net(h)

    def loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, 1)
            labels: (batch,) or (batch, 1) binary labels (0 or 1)
        Returns:
            BCE loss
        """
        labels = labels.view(-1, 1).float()
        return F.binary_cross_entropy_with_logits(logits, labels)


class RegressionHead(nn.Module):
    """Delay duration regression (DEP_DELAY in minutes).

    When use_sigmoid=True: outputs in [0, 1], target should be normalized to [0, 1].
    When use_sigmoid=False: outputs unbounded raw delay values.
    """

    def __init__(self, input_dim: int = 1536, hidden_dim: int = 64, use_sigmoid: bool = False):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (batch, input_dim) last token hidden state
        Returns:
            pred: (batch, 1) predicted delay
        """
        out = self.net(h)
        if self.use_sigmoid:
            out = torch.sigmoid(out)
        return out

    def loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Huber loss for robust regression.

        Args:
            pred: (batch, 1)
            target: (batch,) or (batch, 1)
        Returns:
            Huber loss
        """
        target = target.view(-1, 1).float()
        return F.huber_loss(pred, target, delta=1.0)


class Stage1PredictionHeads(nn.Module):
    """Combined prediction heads for Stage 1 (KG pre-training)."""

    def __init__(self, input_dim: int = 64):
        super().__init__()
        self.cls_head = ClassificationHead(input_dim, hidden_dim=16)
        self.reg_head = RegressionHead(input_dim, hidden_dim=16, use_sigmoid=False)

    def forward(self, h: torch.Tensor) -> tuple:
        """
        Returns:
            cls_logits: (batch, 1)
            reg_pred: (batch, 1)
        """
        return self.cls_head(h), self.reg_head(h)


class Stage2PredictionHeads(nn.Module):
    """Combined prediction heads for Stage 2 (LLM fine-tuning)."""

    def __init__(self, input_dim: int = 1536):
        super().__init__()
        self.cls_head = ClassificationHead(input_dim)
        self.reg_head = RegressionHead(input_dim)

    def forward(self, h: torch.Tensor) -> tuple:
        """
        Returns:
            cls_logits: (batch, 1)
            reg_pred: (batch, 1)
        """
        return self.cls_head(h), self.reg_head(h)
