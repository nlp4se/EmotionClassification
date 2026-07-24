"""Sampling-based imbalance mitigation (applied inside training folds only)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def undersample_multilabel(
    df: pd.DataFrame,
    emotions: list[str],
    cutoff: int,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Iteratively keep instances so each emotion has at most `cutoff` positives.

    Prefers preserving rows that cover under-represented emotions.
    """
    rng = np.random.default_rng(seed)
    labels = df[emotions].to_numpy(dtype=int)
    n = len(df)
    selected = np.zeros(n, dtype=bool)
    # Process emotions from rarest to most frequent
    order = np.argsort(labels.sum(axis=0))
    counts = np.zeros(len(emotions), dtype=int)

    # First pass: keep all rows that help rare labels under cutoff
    for ei in order:
        pos_idx = np.where(labels[:, ei] == 1)[0]
        rng.shuffle(pos_idx)
        for i in pos_idx:
            if counts[ei] >= cutoff:
                break
            if selected[i]:
                continue
            # Accept if no emotion would exceed cutoff
            if np.any(counts + labels[i] > cutoff):
                continue
            selected[i] = True
            counts += labels[i]

    # Second pass: fill remaining quota randomly where possible
    remaining = np.where(~selected)[0]
    rng.shuffle(remaining)
    for i in remaining:
        if np.any(counts + labels[i] > cutoff):
            continue
        selected[i] = True
        counts += labels[i]

    return df.loc[selected].reset_index(drop=True)


def inject_synthetic(
    human_df: pd.DataFrame,
    synthetic_ml: pd.DataFrame,
    emotions: list[str],
    n_per_emotion: int | None,
) -> pd.DataFrame:
    """Concatenate human fold with up to n synthetic positives per emotion."""
    if synthetic_ml is None or len(synthetic_ml) == 0 or n_per_emotion == 0:
        return human_df.reset_index(drop=True)
    parts = [human_df]
    for e in emotions:
        if e == "Neutral":
            continue
        sub = synthetic_ml[synthetic_ml[e] == 1]
        if n_per_emotion is not None:
            sub = sub.head(int(n_per_emotion))
        parts.append(sub)
    out = pd.concat(parts, ignore_index=True)
    return out
