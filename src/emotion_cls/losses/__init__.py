"""Multi-label loss functions for encoder fine-tuning."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_pos_weight(y: torch.Tensor) -> torch.Tensor:
    """pos_weight[c] = neg_c / pos_c."""
    pos = y.sum(dim=0).clamp(min=1.0)
    neg = y.shape[0] - pos
    return (neg / pos).float()


def compute_class_weight(y: torch.Tensor) -> torch.Tensor:
    """w[c] = N / (|E| * pos_c)."""
    n, k = y.shape
    pos = y.sum(dim=0).clamp(min=1.0)
    return (n / (k * pos)).float()


def compute_alpha(y: torch.Tensor) -> torch.Tensor:
    """alpha[c] = neg_c / (pos_c + neg_c)."""
    pos = y.sum(dim=0)
    neg = y.shape[0] - pos
    return (neg / (pos + neg).clamp(min=1.0)).float()


class WeightedBCELoss(nn.Module):
    """BCEWithLogits with optional pos_weight and/or per-class multiplicative weights."""

    def __init__(
        self,
        pos_weight: torch.Tensor | None = None,
        class_weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.reduction = reduction
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)
        self.register_buffer("class_weight", class_weight if class_weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        if self.class_weight is not None:
            loss = loss * self.class_weight.unsqueeze(0)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class MultiLabelFocalLoss(nn.Module):
    """Sigmoid focal loss with per-class alpha (Lin et al., 2017)."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer("alpha", alpha if alpha is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        focal = ((1.0 - pt) ** self.gamma) * bce
        if self.alpha is not None:
            # alpha for positives, (1-alpha) for negatives
            alpha_t = self.alpha.unsqueeze(0) * targets + (1.0 - self.alpha.unsqueeze(0)) * (
                1.0 - targets
            )
            focal = alpha_t * focal
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


class AdaptiveFocalLoss(nn.Module):
    """Adaptive Focal Loss with frequency-based gamma (Labd et al., 2025).

    gamma[c] is larger for rarer classes: gamma[c] = gamma_base * (1 + log(N / pos_c)).
    """

    def __init__(
        self,
        pos_counts: torch.Tensor,
        gamma_base: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.reduction = reduction
        n = float(pos_counts.sum().clamp(min=1.0).item())
        gamma = gamma_base * (1.0 + torch.log((n / pos_counts.clamp(min=1.0))))
        self.register_buffer("gamma", gamma)
        self.register_buffer("alpha", alpha if alpha is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        focal = ((1.0 - pt) ** self.gamma.unsqueeze(0)) * bce
        if self.alpha is not None:
            alpha_t = self.alpha.unsqueeze(0) * targets + (1.0 - self.alpha.unsqueeze(0)) * (
                1.0 - targets
            )
            focal = alpha_t * focal
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


def build_loss(
    method: str,
    y_train: torch.Tensor,
    *,
    reduction: str = "mean",
    focal_gamma: float = 2.0,
) -> nn.Module:
    method = method.lower().replace("-", "_")
    if method in {"none", "baseline", ""}:
        return WeightedBCELoss(reduction=reduction)
    if method == "bce_pos_weight":
        return WeightedBCELoss(pos_weight=compute_pos_weight(y_train), reduction=reduction)
    if method == "bce_weight":
        return WeightedBCELoss(class_weight=compute_class_weight(y_train), reduction=reduction)
    if method == "focal":
        return MultiLabelFocalLoss(
            gamma=focal_gamma, alpha=compute_alpha(y_train), reduction=reduction
        )
    if method == "adaptive_focal":
        pos = y_train.sum(dim=0)
        return AdaptiveFocalLoss(
            pos_counts=pos,
            gamma_base=focal_gamma,
            alpha=compute_alpha(y_train),
            reduction=reduction,
        )
    raise ValueError(f"Unknown loss imbalance method: {method}")
