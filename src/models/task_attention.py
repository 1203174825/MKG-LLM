
import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskAttention(nn.Module):


    def __init__(self, input_dim: int, num_tasks: int, hidden_dim: int = None):
        super().__init__()
        hidden = hidden_dim or max(input_dim // 4, 16)
        self.num_tasks = num_tasks

        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_tasks),
        )
        self._init_weights()
        
        # Temperature coefficient to prevent attention collapse
        self.temperature = 0.7

    def _init_weights(self):
        for m in self.attention.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                # Initialize bias to encourage balanced initial attention
                if m.out_features == self.num_tasks:
                    nn.init.constant_(m.bias, 0.0)
                else:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) feature representation
        Returns:
            alpha: (batch, num_tasks) normalized attention weights
        """
        logits = self.attention(x)  # (batch, num_tasks)
        # Apply temperature scaling to prevent attention collapse
        alpha = F.softmax(logits / self.temperature, dim=-1)  # (batch, num_tasks)
        return alpha

    def get_weighted_loss(self, losses: list, alpha: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted sum of losses using attention weights.

        Args:
            losses: list of tensors [(batch,), (batch,), ...] with length num_tasks
            alpha: (batch, num_tasks) attention weights
        Returns:
            weighted_loss: (batch,) weighted sum
        """
        # Stack losses: (batch, num_tasks)
        loss_stack = torch.stack(losses, dim=-1)  # (batch, num_tasks)
        weighted = (alpha * loss_stack).sum(dim=-1)  # (batch,)
        return weighted.mean()  # scalar
