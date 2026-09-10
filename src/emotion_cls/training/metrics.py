"""Evaluation metrics for multi-label emotion classification."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def top_k_from_logits(logits: np.ndarray, k: int = 3) -> np.ndarray:
    """Sigmoid + retain top-k labels per row.

    Deprecated for production use: this unconditionally assigns exactly k
    labels to every row regardless of confidence, which mismatches a label
    distribution where most instances carry fewer than k true labels (see
    ``threshold_topk``). Kept only for backward-compatible comparison against
    the earlier top-k rule.
    """
    # logits may already be probabilities; treat as logits and sigmoid
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = np.zeros_like(probs, dtype=int)
    kk = min(k, probs.shape[1])
    top = np.argpartition(-probs, kk - 1, axis=1)[:, :kk]
    for i in range(probs.shape[0]):
        preds[i, top[i]] = 1
    return preds


def threshold_topk(probs: np.ndarray, threshold: float = 0.5, k: int = 3) -> np.ndarray:
    """Per-label threshold with a top-1 fallback and a top-k cap.

    Each label is assigned independently via ``probs >= threshold``; if no
    label crosses the threshold, the single most confident label is kept
    (never predict an empty set); if more than ``k`` labels cross it, only
    the ``k`` most confident are kept. This lets predicted cardinality vary
    per instance instead of forcing exactly ``k`` labels every time.
    """
    raw = (probs >= threshold).astype(int)
    preds = np.zeros_like(raw)
    for i in range(probs.shape[0]):
        pos = np.where(raw[i] == 1)[0]
        if len(pos) == 0:
            pos = np.array([int(np.argmax(probs[i]))])
        if len(pos) > k:
            pos = pos[np.argsort(-probs[i, pos])[:k]]
        preds[i, pos] = 1
    return preds


def probs_from_logits(logits: np.ndarray) -> np.ndarray:
    """Sigmoid of raw logits."""
    return 1.0 / (1.0 + np.exp(-logits))


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
