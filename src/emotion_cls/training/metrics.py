"""Evaluation metrics for multi-label emotion classification."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def top_k_from_logits(logits: np.ndarray, k: int = 3) -> np.ndarray:
    """Sigmoid + retain top-k labels per row."""
    # logits may already be probabilities; treat as logits and sigmoid
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = np.zeros_like(probs, dtype=int)
    kk = min(k, probs.shape[1])
    top = np.argpartition(-probs, kk - 1, axis=1)[:, :kk]
    for i in range(probs.shape[0]):
        preds[i, top[i]] = 1
    return preds


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    emotions: list[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["subset_accuracy"] = float(accuracy_score(y_true, y_pred))
    for avg in ("micro", "macro", "weighted"):
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, zero_division=0
        )
        out[f"{avg}_precision"] = float(p)
        out[f"{avg}_recall"] = float(r)
        out[f"{avg}_f1"] = float(f1)
    per_p, per_r, per_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    out["per_emotion"] = {
        e: {"precision": float(per_p[i]), "recall": float(per_r[i]), "f1": float(per_f1[i])}
        for i, e in enumerate(emotions)
    }
    return out
