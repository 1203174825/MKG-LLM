"""
Evaluation metrics for classification and regression tasks.
"""
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    mean_absolute_error, mean_squared_error,
)


def compute_cls_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> dict:
    """
    Compute binary classification metrics.

    Args:
        logits: (N, 1) or (N,) raw model outputs
        labels: (N,) or (N, 1) binary labels (0 or 1)
        threshold: classification threshold (default: 0.5)
    Returns:
        dict with auc, accuracy, precision, recall, f1
    """
    logits_np = logits.detach().cpu().numpy().flatten()
    labels_np = labels.detach().cpu().numpy().flatten().astype(int)
    preds_np = (logits_np >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(labels_np, preds_np),
        "precision": precision_score(labels_np, preds_np, zero_division=0),
        "recall": recall_score(labels_np, preds_np, zero_division=0),
        "f1": f1_score(labels_np, preds_np, zero_division=0),
    }

    # AUC-ROC (only if both classes present)
    if len(np.unique(labels_np)) > 1:
        metrics["auc"] = roc_auc_score(labels_np, logits_np)
        metrics["ap"] = average_precision_score(labels_np, logits_np)
    else:
        metrics["auc"] = 0.0
        metrics["ap"] = 0.0

    return metrics


def compute_reg_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    Compute regression metrics.

    Args:
        pred: (N, 1) or (N,) predicted delay in minutes
        target: (N,) or (N, 1) actual delay in minutes
    Returns:
        dict with mae, rmse, r2
    """
    pred_np = pred.detach().cpu().numpy().flatten()
    target_np = target.detach().cpu().numpy().flatten()

    ss_res = np.sum((target_np - pred_np) ** 2)
    ss_tot = np.sum((target_np - np.mean(target_np)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "mae": mean_absolute_error(target_np, pred_np),
        "rmse": np.sqrt(mean_squared_error(target_np, pred_np)),
        "r2": r2,
    }


def compute_cold_start_metrics(logits, labels, chain_mask):
    """
    Compute metrics for cold-start subset only.

    Args:
        logits: (N, 1) classification logits
        labels: (N,) binary labels
        chain_mask: (N,) bool, True if flight has preceded_by
    Returns:
        dict with cold-start metrics (chain-less subset)
    """
    cold_idx = ~chain_mask
    if cold_idx.sum() == 0:
        return {"cold_start_auc": 0.0, "cold_start_mae": 0.0, "cold_start_count": 0}

    return compute_cls_metrics(logits[cold_idx], labels[cold_idx])
