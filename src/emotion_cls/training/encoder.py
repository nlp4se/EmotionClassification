"""Hugging Face + PyTorch fine-tuning for encoder-only emotion classifiers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from emotion_cls.config import encoder_hub_id, resolve_path
from emotion_cls.data.dataset import (
    inject_synthetic,
    label_matrix,
    load_ground_truth,
    multilabel_folds,
)
from emotion_cls.imbalance.sampling import undersample_multilabel
from emotion_cls.losses import build_loss
from emotion_cls.training.metrics import compute_metrics, top_k_from_logits


@dataclass
class FoldResult:
    fold: int
    metrics: dict[str, Any]


class MultilabelTrainer(Trainer):
    def __init__(self, *args, loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.loss_fn(outputs.logits, labels.float())
        return (loss, outputs) if return_outputs else loss


def _tokenize_dataset(df: pd.DataFrame, emotions: list[str], tokenizer, max_length: int) -> Dataset:
    texts = df["sentence"].tolist()
    labels = label_matrix(df, emotions).tolist()

    def tok(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        enc["labels"] = batch["labels"]
        return enc

    ds = Dataset.from_dict({"text": texts, "labels": labels})
    return ds.map(tok, batched=True, remove_columns=["text"])


def run_multilabel_cv(cfg: dict[str, Any], *, dry_run: bool = False) -> list[FoldResult]:
    emotions = cfg["data"]["emotions"]
    df = load_ground_truth(cfg["data"]["ground_truth"], emotions=emotions, text_column=cfg["data"]["text_column"])
    y = label_matrix(df, emotions)
    hub_id = encoder_hub_id(cfg, cfg["training"]["encoder"])
    max_labels = int(cfg["evaluation"]["max_labels"])
    n_folds = int(cfg["evaluation"]["n_folds"])
    seed = int(cfg["project"]["seed"])
    imb = cfg.get("imbalance", {})
    method = str(imb.get("method", "none"))
    methods = [m.strip() for m in method.split(",") if m.strip()]

    # Optional preloaded synthetic multilabel frame
    synthetic_ml = None
    if "genai_aug" in methods and imb.get("synthetic_ml_path"):
        synthetic_ml = pd.read_csv(resolve_path(imb["synthetic_ml_path"]), sep=";")

    tokenizer = AutoTokenizer.from_pretrained(hub_id)
    results: list[FoldResult] = []
    out_dir = resolve_path(cfg["project"]["output_dir"]) / "encoder_multilabel" / cfg["training"]["encoder"] / method.replace(",", "+")
    out_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"[dry-run] multilabel CV encoder={hub_id} folds={n_folds} imbalance={methods} n={len(df)}")
        return results

    for fold, (train_idx, val_idx) in enumerate(multilabel_folds(y, n_folds, seed), start=1):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        if "undersample" in methods and imb.get("undersample_cutoff"):
            train_df = undersample_multilabel(
                train_df, emotions, int(imb["undersample_cutoff"]), seed=seed + fold
            )
        if "genai_aug" in methods and synthetic_ml is not None:
            train_df = inject_synthetic(
                train_df,
                synthetic_ml,
                emotions,
                imb.get("aug_inject_n"),
            )

        y_train = torch.tensor(label_matrix(train_df, emotions), dtype=torch.float32)
        loss_name = next((m for m in methods if m in {"none", "bce_pos_weight", "bce_weight", "focal", "adaptive_focal"}), "none")
        loss_fn = build_loss(
            loss_name,
            y_train,
            reduction=str(imb.get("reduction", "mean")),
            focal_gamma=float(imb.get("focal_gamma", 2.0)),
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            hub_id,
            num_labels=len(emotions),
            problem_type="multi_label_classification",
        )
        train_ds = _tokenize_dataset(train_df, emotions, tokenizer, int(cfg["training"]["max_length"]))
        val_ds = _tokenize_dataset(val_df, emotions, tokenizer, int(cfg["training"]["max_length"]))

        fold_dir = out_dir / f"fold_{fold}"
        args = TrainingArguments(
            output_dir=str(fold_dir),
            num_train_epochs=float(cfg["training"]["epochs"]),
            learning_rate=float(cfg["training"]["learning_rate"]),
            per_device_train_batch_size=int(cfg["training"]["train_batch_size"]),
            per_device_eval_batch_size=int(cfg["training"]["eval_batch_size"]),
            weight_decay=float(cfg["training"]["weight_decay"]),
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            seed=seed,
            report_to=[],
            fp16=bool(cfg["training"].get("fp16", False)),
        )

        def hf_metrics(eval_pred):
            logits, labels = eval_pred
            preds = top_k_from_logits(np.asarray(logits), k=max_labels)
            m = compute_metrics(np.asarray(labels), preds, emotions)
            return {
                "macro_f1": m["macro_f1"],
                "micro_f1": m["micro_f1"],
                "subset_accuracy": m["subset_accuracy"],
            }

        trainer = MultilabelTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            compute_metrics=hf_metrics,
            loss_fn=loss_fn,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=int(cfg["training"].get("early_stopping_patience", 2)))],
        )
        trainer.train()
        pred_out = trainer.predict(val_ds)
        preds = top_k_from_logits(pred_out.predictions, k=max_labels)
        metrics = compute_metrics(label_matrix(val_df, emotions), preds, emotions)
        results.append(FoldResult(fold=fold, metrics=metrics))
        with open(fold_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    _write_cv_summary(results, out_dir / "cv_summary.json", emotions)
    return results


def run_binary_ensemble_cv(cfg: dict[str, Any], *, dry_run: bool = False) -> list[dict[str, Any]]:
    """Train one binary classifier per emotion on shared multilabel folds; assemble with top-k."""
    emotions = cfg["data"]["emotions"]
    df = load_ground_truth(cfg["data"]["ground_truth"], emotions=emotions, text_column=cfg["data"]["text_column"])
    hub_id = encoder_hub_id(cfg, cfg["training"]["encoder"])
    tokenizer = AutoTokenizer.from_pretrained(hub_id)
    out_dir = resolve_path(cfg["project"]["output_dir"]) / "encoder_binary" / cfg["training"]["encoder"]
    out_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"[dry-run] binary ensemble encoder={hub_id} folds={cfg['evaluation']['n_folds']} n={len(df)}")
        return []

    summary = _assemble_binary_on_multilabel_folds(cfg, hub_id, tokenizer)
    with open(out_dir / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def _assemble_binary_on_multilabel_folds(cfg, hub_id, tokenizer, dry_run=False):
    """Train binary heads on shared multilabel folds and apply top-k assembly."""
    emotions = cfg["data"]["emotions"]
    df = load_ground_truth(cfg["data"]["ground_truth"], emotions=emotions, text_column=cfg["data"]["text_column"])
    y = label_matrix(df, emotions)
    n_folds = int(cfg["evaluation"]["n_folds"])
    seed = int(cfg["project"]["seed"])
    max_labels = int(cfg["evaluation"]["max_labels"])
    summary = []

    for fold, (train_idx, val_idx) in enumerate(multilabel_folds(y, n_folds, seed), start=1):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        probs = np.zeros((len(val_df), len(emotions)), dtype=np.float32)
        for ei, emotion in enumerate(emotions):
            model = AutoModelForSequenceClassification.from_pretrained(hub_id, num_labels=2)
            train_ds = Dataset.from_dict({"text": train_df["sentence"].tolist(), "labels": train_df[emotion].astype(int).tolist()})
            val_ds = Dataset.from_dict({"text": val_df["sentence"].tolist(), "labels": val_df[emotion].astype(int).tolist()})

            def tok(batch):
                return tokenizer(batch["text"], truncation=True, max_length=int(cfg["training"]["max_length"]))

            train_ds = train_ds.map(tok, batched=True, remove_columns=["text"])
            val_ds = val_ds.map(tok, batched=True, remove_columns=["text"])
            args = TrainingArguments(
                output_dir=str(resolve_path(cfg["project"]["output_dir"]) / "_tmp_binary" / emotion / f"fold_{fold}"),
                num_train_epochs=float(cfg["training"]["epochs"]),
                learning_rate=float(cfg["training"]["learning_rate"]),
                per_device_train_batch_size=int(cfg["training"]["train_batch_size"]),
                per_device_eval_batch_size=int(cfg["training"]["eval_batch_size"]),
                report_to=[],
                save_strategy="no",
            )
            trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, tokenizer=tokenizer)
            trainer.train()
            logits = trainer.predict(val_ds).predictions
            probs[:, ei] = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]

        # Positive if above 0.5, then top-k by confidence
        raw = (probs >= 0.5).astype(int)
        # Enforce max_labels
        preds = np.zeros_like(raw)
        for i in range(len(val_df)):
            pos = np.where(raw[i] == 1)[0]
            if len(pos) == 0:
                pos = np.array([int(np.argmax(probs[i]))])
            if len(pos) > max_labels:
                pos = pos[np.argsort(-probs[i, pos])[:max_labels]]
            preds[i, pos] = 1
        metrics = compute_metrics(label_matrix(val_df, emotions), preds, emotions)
        summary.append({"fold": fold, "metrics": metrics})
    return summary


def _write_cv_summary(results: list[FoldResult], path: Path, emotions: list[str]) -> None:
    agg = {}
    keys = ["subset_accuracy", "micro_f1", "macro_f1", "weighted_f1"]
    for k in keys:
        vals = [r.metrics[k] for r in results]
        agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    per = {}
    for e in emotions:
        f1s = [r.metrics["per_emotion"][e]["f1"] for r in results]
        per[e] = {"f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s))}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"folds": [r.metrics for r in results], "aggregate": agg, "per_emotion": per}, f, indent=2)
