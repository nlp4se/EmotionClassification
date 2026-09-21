#!/usr/bin/env python3
"""Report Jev (TypeSafe AI) results in the same shape as the RQ2 decoder
tables, for direct side-by-side comparison with the existing decoders.

Reads outputs/decoder_classify/jev-latest/<strategy>/t_na/cv_summary.json
(written by `emotion-cls classify-typesafe`) plus the per-fold predictions
for a per-emotion breakdown, and prints/saves a report mirroring
tab:rq2-strategy and tab:rq2-per-emotion.

Usage:
    python scripts/report_typesafe_rq2.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from emotion_cls.data.dataset import label_matrix, load_ground_truth, multilabel_folds
from emotion_cls.training.metrics import compute_metrics, threshold_topk

N_FOLDS = 10
SEED = 42

EMOTIONS = ["Joy", "Trust", "Fear", "Surprise", "Sadness", "Disgust", "Anger", "Anticipation", "Neutral"]
STRATEGIES = ["zero_shot", "few_shot_guidelines", "few_shot_guidelines_dataset"]
STRATEGY_LABEL = {
    "zero_shot": "Zero-shot",
    "few_shot_guidelines": "Few-shot (guid.)",
    "few_shot_guidelines_dataset": "Few-shot (dataset)",
}
OUT_ROOT = Path("outputs/decoder_classify/jev-latest")
MAX_LABELS = 3


def load_summary(strategy: str) -> dict | None:
    p = OUT_ROOT / strategy / "t_na" / "cv_summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_all_fold_probs(strategy: str, gt: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Stitch every fold's probs.npy back into one (N, 9) array, aligned to
    ground-truth labels the same way the real run computed them: by
    re-deriving the identical seeded fold split (multilabel_folds), not by
    joining on sentence text. The corpus has a handful of duplicate sentence
    strings (verified: 7 rows, e.g. multiple apps both reviewed as just
    "Educational"), so a text-keyed join silently drops/misattributes rows;
    positional alignment against the same split used at run time doesn't.
    """
    strat_dir = OUT_ROOT / strategy / "t_na"
    y = label_matrix(gt, EMOTIONS)
    all_probs, all_y = [], []
    for fold, (_, val_idx) in enumerate(multilabel_folds(y, N_FOLDS, SEED), start=1):
        fold_dir = strat_dir / f"fold_{fold}"
        probs = np.load(fold_dir / "probs.npy")
        val_df = gt.iloc[val_idx].reset_index(drop=True)
        assert len(probs) == len(val_df), f"fold {fold}: {len(probs)} probs vs {len(val_df)} val rows"
        all_probs.append(probs)
        all_y.append(label_matrix(val_df, EMOTIONS))
    return np.concatenate(all_probs, axis=0), np.concatenate(all_y, axis=0)


def main() -> None:
    gt = load_ground_truth("Datasets/GroundTruth.csv", emotions=EMOTIONS)

    print("=" * 100)
    print("Jev (TypeSafe AI, System One) -- RQ2-parallel results")
    print("=" * 100)
    print(f"{'Strategy':<22}{'P':>8}{'R':>8}{'Macro-F1':>14}{'Micro-F1':>10}{'Calls':>8}{'In.tok':>10}{'Out.tok':>10}")

    rows_for_table = []
    for strategy in STRATEGIES:
        summary = load_summary(strategy)
        if summary is None:
            print(f"{STRATEGY_LABEL[strategy]:<22} -- not yet run --")
            continue
        agg = summary["aggregate"]
        usage = summary.get("usage", {})
        macro = agg.get("macro_f1", {})
        micro = agg.get("micro_f1", {})

        # Macro/micro P/R aren't in cv_summary's aggregate (only F1 is) --
        # recompute pooled across all folds the same way the other RQ2 rows do.
        probs, y_true = load_all_fold_probs(strategy, gt)
        preds = threshold_topk(probs, k=MAX_LABELS)
        pooled = compute_metrics(y_true, preds, EMOTIONS)

        n_calls = summary.get("n_llm_calls", 0)
        in_tok = usage.get("prompt_tokens", 0) or 0
        out_tok = usage.get("completion_tokens", 0) or 0
        print(
            f"{STRATEGY_LABEL[strategy]:<22}"
            f"{pooled['macro_precision']:>8.3f}{pooled['macro_recall']:>8.3f}"
            f"{macro.get('mean', float('nan')):>8.3f}$\\pm${macro.get('std', float('nan')):.3f}"
            f"{micro.get('mean', float('nan')):>10.3f}"
            f"{n_calls:>8}{in_tok:>10.0f}{out_tok:>10.0f}"
        )
        rows_for_table.append((strategy, pooled, macro, micro, preds, y_true))

    if not rows_for_table:
        print("\nNo completed strategies yet.")
        return

    best_strategy, best_pooled, best_macro, _, best_preds, best_y_true = max(
        rows_for_table, key=lambda r: r[2].get("mean", -1)
    )
    print()
    print(f"Best strategy: {STRATEGY_LABEL[best_strategy]} (macro-F1 {best_macro['mean']:.3f} +- {best_macro['std']:.3f})")
    print()
    print("Per-emotion (best strategy), sorted by support descending:")
    per_emo = compute_metrics(best_y_true, best_preds, EMOTIONS)["per_emotion"]
    support = {e: int(best_y_true[:, j].sum()) for j, e in enumerate(EMOTIONS)}
    order = sorted(EMOTIONS, key=lambda e: -support[e])
    print(f"{'Emotion':<14}{'Support':>8}{'P':>8}{'R':>8}{'F1':>8}")
    for e in order:
        m = per_emo[e]
        print(f"{e:<14}{support[e]:>8}{m['precision']:>8.3f}{m['recall']:>8.3f}{m['f1']:>8.3f}")


if __name__ == "__main__":
    main()
