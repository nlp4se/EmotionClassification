#!/usr/bin/env python3
"""Train and publish the paper's best encoder-only configuration.

Refits RoBERTa-large (multi-label head, generative augmentation n=100 per
emotion, BCE positive-weighted loss -- the RQ4 winner, macro-F1
0.591+-0.054 under 10-fold CV) on the FULL training pool (all CV folds
combined) rather than a single held-out fold, since a published checkpoint
has no "held-out set" of its own to report against: the paper's CV numbers
already establish expected performance, and this refit uses every available
positive example.

Usage:
    python scripts/train_final_model.py --push --repo-id quim-motger/<name>
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments

from emotion_cls.config import load_default_config, load_env, resolve_path
from emotion_cls.data.dataset import label_matrix, load_ground_truth
from emotion_cls.imbalance.sampling import inject_synthetic
from emotion_cls.losses import build_loss
from emotion_cls.training.encoder import MultilabelTrainer, _exclude_tuning_holdout, _tokenize_dataset

ENCODER_KEY = "roberta-large"
HUB_ID = "roberta-large"
AUG_N = 100
SYNTHETIC_PATH = "Datasets/synthetic_multilabel_best.csv"
LOCAL_DIR = "models/emotion-roberta-large-multilabel-genai-bce"

MODEL_CARD = """---
license: mit
language: en
tags:
- text-classification
- multi-label-classification
- emotion-classification
- app-reviews
- roberta
base_model: roberta-large
datasets:
- nlp4se/app-review-emotions
metrics:
- f1
pipeline_tag: text-classification
---

# emotion-roberta-large-multilabel-genai-bce

Fine-tuned `roberta-large` multi-label classifier over Plutchik's eight
emotions plus Neutral, applied to English mobile app review sentences.

This is the best-performing encoder-only configuration found in a broader
comparison of encoder-only and decoder-only LLMs for this task: a shared
multi-label classification head trained with a BCE loss that up-weights
each emotion's positive class in proportion to its rarity, on a training
pool augmented with LLM-generated synthetic reviews (100 per emotion) to
fix the baseline model's complete failure to detect Fear, Surprise and
Anger.

Under 10-fold cross-validation on the human-labelled ground truth, this
configuration reaches **macro-F1 0.591 (+-0.054)**, recovering all three
previously undetected emotions (Anger 0->0.501, Fear 0->0.410,
Surprise 0->0.346 F1) with no regression on the four majority emotions.

This specific checkpoint is refit on the **full** training pool (all
cross-validation folds combined, plus the same synthetic augmentation), so
it has no internal held-out set of its own -- treat the cross-validated
macro-F1 above as the expected-performance estimate for this recipe, not a
number to reproduce exactly from this single checkpoint.

## Labels

`Joy`, `Trust`, `Fear`, `Surprise`, `Sadness`, `Disgust`, `Anger`,
`Anticipation`, `Neutral` -- multi-label (0 or more may apply per sentence).

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForSequenceClassification.from_pretrained("{repo_id}")

emotions = ["Joy", "Trust", "Fear", "Surprise", "Sadness", "Disgust", "Anger", "Anticipation", "Neutral"]

text = "This app used to be great but the last update broke everything."
inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    probs = torch.sigmoid(model(**inputs).logits)[0]

predicted = [e for e, p in zip(emotions, probs) if p >= 0.5]
print(predicted, probs.tolist())
```

Note: the paper's decision rule assigns a label when `p >= 0.5`, falls back
to the single most confident label if none cross that threshold, and caps
predictions at 3 labels (the maximum cardinality observed in the ground
truth). Replicate that rule yourself for scoring compatible with the paper.

## Training data

- Human-labelled ground truth: 1,112 sentences from 257 apps across 10
  Google Play categories, adapted from Plutchik's eight basic emotions plus
  Neutral (Motger et al., 2025).
- Synthetic augmentation: up to 100 additional LLM-generated reviews per
  emotion (`claude-opus-4-6`, few-shot with guideline examples plus dataset
  exemplars), selected via an embedding-based augmentation-utility ranking
  before injection.

## Citation

If you use this model, please cite the ground-truth dataset paper
referenced in the parent replication package.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push", action="store_true", help="Upload to the Hugging Face Hub after training.")
    parser.add_argument("--repo-id", default=None, help="e.g. quim-motger/emotion-roberta-large-multilabel-genai-bce")
    parser.add_argument("--remove-local-after-push", action="store_true")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    args = parser.parse_args()

    load_env()
    cfg = load_default_config(args.config, args.models_config)
    emotions = cfg["data"]["emotions"]

    print("Loading ground truth and excluding the LR-tuning holdout (matches the paper's CV protocol)...")
    df = load_ground_truth(cfg["data"]["ground_truth"], emotions=emotions, text_column=cfg["data"]["text_column"])
    df = _exclude_tuning_holdout(df, emotions, cfg)
    print(f"  base training pool: {len(df)} sentences")

    print(f"Injecting up to {AUG_N} synthetic positives per emotion from {SYNTHETIC_PATH} ...")
    import pandas as pd

    synthetic_ml = pd.read_csv(resolve_path(SYNTHETIC_PATH), sep=";")
    train_df = inject_synthetic(df, synthetic_ml, emotions, AUG_N)
    print(f"  final training set: {len(train_df)} rows")

    y_train = torch.tensor(label_matrix(train_df, emotions), dtype=torch.float32)
    loss_fn = build_loss("bce_pos_weight", y_train, reduction="mean")

    print(f"Loading tokenizer/model: {HUB_ID}")
    tokenizer = AutoTokenizer.from_pretrained(HUB_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        HUB_ID, num_labels=len(emotions), problem_type="multi_label_classification"
    )
    train_ds = _tokenize_dataset(train_df, emotions, tokenizer, int(cfg["training"]["max_length"]))

    local_dir = resolve_path(LOCAL_DIR)
    local_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(local_dir / "_trainer_tmp"),
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

    print(f"Training on {len(train_df)} sentences for 3 epochs (LR=2e-5, batch=8)...")
    trainer = MultilabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        loss_fn=loss_fn,
    )
    trainer.train()

    print(f"Saving final model to {local_dir}")
    trainer.save_model(str(local_dir))
    tokenizer.save_pretrained(str(local_dir))
    shutil.rmtree(local_dir / "_trainer_tmp", ignore_errors=True)

    # id2label / label2id so downstream users get named outputs, not raw indices
    model.config.id2label = dict(enumerate(emotions))
    model.config.label2id = {e: i for i, e in enumerate(emotions)}
    model.config.problem_type = "multi_label_classification"
    model.config.save_pretrained(str(local_dir))

    if args.push:
        if not args.repo_id:
            raise SystemExit("--push requires --repo-id (e.g. quim-motger/emotion-roberta-large-multilabel-genai-bce)")
        from huggingface_hub import HfApi

        card_text = MODEL_CARD.format(repo_id=args.repo_id)
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
