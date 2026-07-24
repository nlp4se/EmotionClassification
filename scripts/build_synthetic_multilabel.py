# Convert provider-specific synthetic CSVs into one multilabel CSV for training.
#
#   python scripts/build_synthetic_multilabel.py \
#     --strategy few_shot_guidelines_dataset --genai Claude \
#     --n-per-emotion 100 --out Datasets/synthetic_multilabel.csv

from __future__ import annotations

import argparse
from pathlib import Path

from emotion_cls.config import load_default_config
from emotion_cls.data.dataset import load_synthetic_dir, synthetic_as_multilabel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="few_shot_guidelines_dataset")
    parser.add_argument("--genai", default=None, help="Claude | Gemini | GPT | Mistral")
    parser.add_argument("--n-per-emotion", type=int, default=None)
    parser.add_argument("--out", default="Datasets/synthetic_multilabel.csv")
    args = parser.parse_args()

    cfg = load_default_config()
    emotions = cfg["data"]["generation_emotions"]
    all_emotions = cfg["data"]["emotions"]
    synth = load_synthetic_dir(cfg["data"]["synthetic_dir"], emotions)
    ml = synthetic_as_multilabel(
        synth,
        emotions,
        strategy=args.strategy,
        genai=args.genai,
        n_per_emotion=args.n_per_emotion,
    )
    # Ensure full label schema (any emotion not in the generation set stays 0)
    for e in all_emotions:
        if e not in ml.columns:
            ml[e] = 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ml.to_csv(out, sep=";", index=False)
    print(f"Wrote {len(ml)} rows -> {out}")


if __name__ == "__main__":
    main()
