#!/usr/bin/env python3
"""Train and publish the paper's best binary-ensemble configuration.

Refits nine independent BERT-base binary classifiers (one per emotion, the
RQ4 winner: generative augmentation n=100 per emotion + focal loss,
macro-F1 0.530+-0.074 under 10-fold CV) on the FULL training pool, one
model per emotion, each saved to its own subfolder of a single local
directory so they can be pushed as one Hugging Face repo with nine
subfolders (`AutoModelForSequenceClassification.from_pretrained(repo_id,
subfolder="Joy")` etc.), rather than nine separate repos or one opaque
wrapper class.

Usage:
    python scripts/train_final_binary_ensemble.py --push --repo-id quim-motger/<name>
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments

from emotion_cls.config import load_default_config, load_env, resolve_path
from emotion_cls.imbalance.sampling import inject_synthetic
from emotion_cls.losses import build_loss
from emotion_cls.training.encoder import MultilabelTrainer, _exclude_tuning_holdout
from emotion_cls.data.dataset import load_ground_truth

ENCODER_KEY = "bert-base-cased"
HUB_ID = "bert-base-cased"
AUG_N = 100
FOCAL_GAMMA = 2.0
SYNTHETIC_PATH = "Datasets/synthetic_multilabel_best.csv"
LOCAL_DIR = "models/emotion-bert-base-binary-ensemble"

MODEL_CARD = """---
license: mit
language: en
tags:
- text-classification
- binary-classification
- emotion-classification
- app-reviews
- bert
- ensemble
base_model: bert-base-cased
datasets:
- nlp4se/app-review-emotions
metrics:
- f1
pipeline_tag: text-classification
---

# emotion-bert-base-binary-ensemble

Nine independent `bert-base-cased` binary classifiers, one per Plutchik
emotion plus Neutral, applied to English mobile app review sentences. This
is the binary-ensemble formulation's best-performing configuration found in
a broader comparison of encoder-only and decoder-only LLMs for this task:
each emotion gets its own independent encoder and decision boundary rather
than sharing one classification head, trained with focal loss on a
training pool augmented with LLM-generated synthetic reviews (100 per
emotion).

Under 10-fold cross-validation, this configuration reaches
**macro-F1 0.530 (+-0.074)**. It trails the multi-label formulation's best
configuration ([`{sibling_repo}`](https://huggingface.co/{sibling_repo}),
macro-F1 0.591) on this corpus, though binary ensemble is markedly more
robust to loss-reweighting miscalibration and does not share the
multi-label head's complete failure on rare emotions at baseline.

**This repo holds nine separate models, not one.** Each subfolder is an
independent `bert-base-cased` checkpoint with a single sigmoid output
(`num_labels=1`): presence/absence of that one emotion. There is no shared
backbone or parameter tying between them; assembling all nine into a
prediction requires nine forward passes, one per emotion, matching how the
paper reports inference cost for this formulation.

## Labels / subfolders

`Joy`, `Trust`, `Fear`, `Surprise`, `Sadness`, `Disgust`, `Anger`,
`Anticipation`, `Neutral` -- each is a subfolder in this repo containing
its own independent binary classifier.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

repo_id = "{repo_id}"
emotions = ["Joy", "Trust", "Fear", "Surprise", "Sadness", "Disgust", "Anger", "Anticipation", "Neutral"]
tok = AutoTokenizer.from_pretrained(repo_id, subfolder=emotions[0])  # tokenizer is identical across subfolders

text = "This app used to be great but the last update broke everything."
inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)

probs = {{}}
for emotion in emotions:
    model = AutoModelForSequenceClassification.from_pretrained(repo_id, subfolder=emotion)
    with torch.no_grad():
        probs[emotion] = torch.sigmoid(model(**inputs).logits)[0, 0].item()

predicted = [e for e, p in probs.items() if p >= 0.5]
print(predicted, probs)
```

Note: the paper's decision rule assigns a label when `p >= 0.5`, falls back
to the single most confident label if none cross that threshold, and caps
predictions at 3 labels (the maximum cardinality observed in the ground
truth). Replicate that rule yourself for scoring compatible with the paper.

## Training data

Same recipe as the sibling multi-label model
([`{sibling_repo}`](https://huggingface.co/{sibling_repo})): the human-labelled
ground truth of 1,112 sentences (Motger et al., 2025), refit on the full
training pool (all cross-validation folds combined) plus up to 100
LLM-generated synthetic reviews per emotion, but with focal loss
(gamma=2.0) in place of BCE positive weighting, and one independent model
per emotion instead of one shared head.

## Citation

If you use this model, please cite the ground-truth dataset paper
referenced in the parent replication package.
"""


def train_one_emotion(
    emotion: str,
    train_df: pd.DataFrame,
    tokenizer,
    cfg: dict,
    out_dir: Path,
) -> None:
    y_col = torch.tensor(train_df[emotion].astype(float).values, dtype=torch.float32).unsqueeze(1)
    loss_fn = build_loss("focal", y_col, reduction="mean", focal_gamma=FOCAL_GAMMA)

    model = AutoModelForSequenceClassification.from_pretrained(HUB_ID, num_labels=1, problem_type="multi_label_classification")

    train_ds = Dataset.from_dict(
        {
            "text": train_df["sentence"].tolist(),
            "labels": [[float(v)] for v in train_df[emotion].astype(int).tolist()],
        }
    )

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=int(cfg["training"]["max_length"]))

    train_ds = train_ds.map(tok, batched=True, remove_columns=["text"])

    emotion_dir = out_dir / emotion
    emotion_dir.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(emotion_dir / "_trainer_tmp"),
        num_train_epochs=3.0,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        weight_decay=0.01,
        warmup_ratio=0.1,
        adam_epsilon=1e-6,
        eval_strategy="no",
        save_strategy="no",
        seed=int(cfg["project"]["seed"]),
        report_to=[],
        fp16=False,
    )
    trainer = MultilabelTrainer(model=model, args=args, train_dataset=train_ds, processing_class=tokenizer, loss_fn=loss_fn)
    trainer.train()

    trainer.save_model(str(emotion_dir))
    tokenizer.save_pretrained(str(emotion_dir))
    shutil.rmtree(emotion_dir / "_trainer_tmp", ignore_errors=True)
    print(f"  [{emotion}] saved -> {emotion_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--sibling-repo", default="quim-motger/emotion-roberta-large-multilabel-genai-bce")
    parser.add_argument("--remove-local-after-push", action="store_true")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    args = parser.parse_args()

    load_env()
    cfg = load_default_config(args.config, args.models_config)
    emotions = cfg["data"]["emotions"]

    print("Loading ground truth and excluding the LR-tuning holdout...")
    df = load_ground_truth(cfg["data"]["ground_truth"], emotions=emotions, text_column=cfg["data"]["text_column"])
    df = _exclude_tuning_holdout(df, emotions, cfg)
    print(f"  base training pool: {len(df)} sentences")

    print(f"Injecting up to {AUG_N} synthetic positives per emotion...")
    synthetic_ml = pd.read_csv(resolve_path(SYNTHETIC_PATH), sep=";")
    train_df = inject_synthetic(df, synthetic_ml, emotions, AUG_N)
    print(f"  final training set: {len(train_df)} rows")

    tokenizer = AutoTokenizer.from_pretrained(HUB_ID)
    local_dir = resolve_path(LOCAL_DIR)
    local_dir.mkdir(parents=True, exist_ok=True)

    for i, emotion in enumerate(emotions, start=1):
        print(f"[{i}/{len(emotions)}] training {emotion} ...")
        train_one_emotion(emotion, train_df, tokenizer, cfg, local_dir)

    if args.push:
        if not args.repo_id:
            raise SystemExit("--push requires --repo-id")
        from huggingface_hub import HfApi

        card_text = MODEL_CARD.format(repo_id=args.repo_id, sibling_repo=args.sibling_repo)
        (local_dir / "README.md").write_text(card_text)

        print(f"Uploading {local_dir} -> {args.repo_id}")
        api = HfApi()
        api.create_repo(args.repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=str(local_dir), repo_id=args.repo_id, repo_type="model")
        print(f"Done: https://huggingface.co/{args.repo_id}")

        if args.remove_local_after_push:
            print(f"Removing local copy {local_dir}")
            shutil.rmtree(local_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
