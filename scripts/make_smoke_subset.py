# Build a tiny, label-diverse ground-truth subset for end-to-end smoke testing.
#
#   python scripts/make_smoke_subset.py --n-per-emotion 8 --out Datasets/smoke/ground_truth_smoke.csv
#
# Keeps the original GroundTruth.csv schema (same columns, semicolon-separated) so
# every experiment command works unmodified against it via `--config configs/smoke.yaml`.

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from emotion_cls.config import load_default_config
from emotion_cls.data.dataset import load_ground_truth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-emotion", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="Datasets/smoke/ground_truth_smoke.csv")
    args = parser.parse_args()

    cfg = load_default_config()
    emotions = cfg["data"]["emotions"]

    # load_ground_truth drops Reject=1 / empty-sentence rows and resets the index,
    # so selection must happen on *this* frame directly (not re-looked-up against
    # the raw CSV, whose row positions no longer line up after that filtering).
    clean = load_ground_truth(cfg["data"]["ground_truth"], emotions=emotions)

    keep_idx: set[int] = set()
    for e in emotions:
        pos = clean.index[clean[e] == 1].tolist()
        sample = pd.Index(pos).to_series().sample(
            n=min(args.n_per_emotion, len(pos)), random_state=args.seed
        )
        keep_idx.update(sample.tolist())

    subset = clean.loc[sorted(keep_idx)].drop(columns=["source"]).reset_index(drop=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(out, sep=";", index=False)

    counts = clean.loc[sorted(keep_idx), emotions].sum()
    print(f"Wrote {len(subset)} rows -> {out}")
    print("Per-emotion positives in subset:")
    print(counts.to_string())


if __name__ == "__main__":
    main()
