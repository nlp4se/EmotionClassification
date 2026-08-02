"""Hugging Face + PyTorch fine-tuning for encoder-only emotion classifiers.

Resumable at fold granularity (and per-emotion within binary ensemble folds).
Completed ``fold_N/metrics.json`` files are skipped on re-run.
"""

from __future__ import annotations

import json
import shutil
import time
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
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from emotion_cls.config import encoder_hub_id, resolve_path
from emotion_cls.data.dataset import (
    label_matrix,
    load_ground_truth,
    multilabel_folds,
)
from emotion_cls.experiment import ExperimentRun, timed
from emotion_cls.imbalance.sampling import inject_synthetic, undersample_multilabel
from emotion_cls.losses import build_loss
from emotion_cls.training.metrics import compute_metrics, top_k_from_logits


def _cleanup_trainer_artifacts(root: Path) -> None:
    """Remove HF Trainer checkpoints / leftover shards; keep metrics and predictions.

    Applied after every multilabel fold and every binary emotion trainer run.
    Walks ``root`` recursively so nested ``emotion_*/checkpoint-*`` dirs are cleared.
    """
    if not root.exists():
        return
    for path in sorted(root.rglob("checkpoint-*"), reverse=True):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    for path in root.rglob("runs"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    junk_names = {
        "model.safetensors",
        "pytorch_model.bin",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
        "rng_state.pth",
        "config.json",  # only remove if a sibling weight file was present; keep if alone? safer skip config
    }
    # Do not delete config.json — harmless and tiny; keep tokenizer files if any
    junk_names.discard("config.json")
    for name in junk_names:
        for p in root.rglob(name):
            if p.is_file():
                p.unlink(missing_ok=True)
    # Free GPU memory from the finished trainer/model if CUDA is in use
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



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
        if self.loss_fn is not None:
            self.loss_fn.to(outputs.logits.device)
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


def run_multilabel_cv(
    cfg: dict[str, Any],
    *,
    dry_run: bool = False,
    resume: bool = True,
) -> list[FoldResult]:
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
    resume = bool(cfg.get("experiment", {}).get("resume", resume))
    # Allow filesystem tags like undersample_c150 / genai_aug_n100
    methods_norm = []
    for m in methods:
        if m.startswith("undersample"):
            methods_norm.append("undersample")
        elif m.startswith("genai_aug"):
            methods_norm.append("genai_aug")
        else:
            methods_norm.append(m)

    synthetic_ml = None
    if "genai_aug" in methods_norm:
        synth_path = imb.get("synthetic_ml_path")
        if not synth_path:
            raise ValueError(
                "imbalance method 'genai_aug' requires --synthetic-ml-path "
                "(build it with scripts/build_synthetic_multilabel.py)."
            )
        synthetic_ml = pd.read_csv(resolve_path(synth_path), sep=";")

    if "mlsmote" in methods_norm:
        raise ValueError(
            "MLSMOTE is implemented (embedding-space) but not wired into the HF text Trainer yet. "
            "Omit 'mlsmote' from --imbalance for now."
        )

    out_dir = (
        resolve_path(cfg["project"]["output_dir"])
        / "encoder_multilabel"
        / cfg["training"]["encoder"]
        / method.replace(",", "+")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    run = ExperimentRun(
        out_dir,
        name=f"encoder_multilabel:{cfg['training']['encoder']}:{method}",
        cfg=cfg,
        resume=resume,
        extra_meta={"hub_id": hub_id, "imbalance": methods},
    )

    if dry_run:
        print(f"[dry-run] multilabel CV encoder={hub_id} folds={n_folds} imbalance={methods} n={len(df)}")
        run.finalize(status="dry_run")
        return []

    tokenizer = AutoTokenizer.from_pretrained(hub_id)
    results: list[FoldResult] = []

    for fold, (train_idx, val_idx) in enumerate(multilabel_folds(y, n_folds, seed), start=1):
        fold_dir = out_dir / f"fold_{fold}"

        if run.fold_done(fold):
            metrics = run.load_fold_metrics(fold)
            if metrics:
                clean = {k: v for k, v in metrics.items() if k != "_meta"}
                results.append(FoldResult(fold=fold, metrics=clean))
                print(f"[resume] skip fold {fold}")
                _write_cv_summary(results, out_dir / "cv_summary.json", emotions, run=run)
                continue

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        if "undersample" in methods_norm and imb.get("undersample_cutoff"):
            train_df = undersample_multilabel(
                train_df, emotions, int(imb["undersample_cutoff"]), seed=seed + fold
            )
        if "genai_aug" in methods_norm and synthetic_ml is not None:
            train_df = inject_synthetic(
                train_df,
                synthetic_ml,
                emotions,
                imb.get("aug_inject_n"),
            )

        y_train = torch.tensor(label_matrix(train_df, emotions), dtype=torch.float32)
        loss_name = next(
            (
                m
                for m in methods_norm
                if m in {"none", "bce_pos_weight", "bce_weight", "focal", "adaptive_focal"}
            ),
            "none",
        )
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

        fold_dir.mkdir(parents=True, exist_ok=True)
        args = TrainingArguments(
            output_dir=str(fold_dir),
            num_train_epochs=float(cfg["training"]["epochs"]),
            learning_rate=float(cfg["training"]["learning_rate"]),
            per_device_train_batch_size=int(cfg["training"]["train_batch_size"]),
            per_device_eval_batch_size=int(cfg["training"]["eval_batch_size"]),
            weight_decay=float(cfg["training"]["weight_decay"]),
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
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

        run.log_event("fold_started", fold=fold, n_train=len(train_df), n_val=len(val_df))
        t0 = timed()
        try:
            trainer = MultilabelTrainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                processing_class=tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer),
                compute_metrics=hf_metrics,
                loss_fn=loss_fn,
                callbacks=[
                    EarlyStoppingCallback(
                        early_stopping_patience=int(cfg["training"].get("early_stopping_patience", 2))
                    )
                ],
            )
            trainer.train()
            pred_out = trainer.predict(val_ds)
            preds = top_k_from_logits(pred_out.predictions, k=max_labels)
            metrics = compute_metrics(label_matrix(val_df, emotions), preds, emotions)
            elapsed = timed() - t0

            # Persist predictions for audit
            pred_rows = []
            for i, sent in enumerate(val_df["sentence"].tolist()):
                labs = [emotions[j] for j, v in enumerate(preds[i]) if v == 1]
                pred_rows.append({"idx": i, "sentence": sent, "pred": json.dumps(labs)})
            pd.DataFrame(pred_rows).to_csv(fold_dir / "predictions.csv", index=False)

            results.append(FoldResult(fold=fold, metrics=metrics))
            run.mark_fold_done(fold, metrics, elapsed_s=elapsed)
            _write_cv_summary(results, out_dir / "cv_summary.json", emotions, run=run)
            print(f"fold {fold} done in {elapsed:.1f}s macro_f1={metrics['macro_f1']:.4f}")
        finally:
            # Always drop checkpoints (success or crash mid-fold)
            _cleanup_trainer_artifacts(fold_dir)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    run.finalize(status="completed", n_folds=len(results))
    return results


def run_binary_ensemble_cv(
    cfg: dict[str, Any],
    *,
    dry_run: bool = False,
    resume: bool = True,
) -> list[dict[str, Any]]:
    """Train one binary classifier per emotion on shared multilabel folds; assemble with top-k."""
    emotions = cfg["data"]["emotions"]
    df = load_ground_truth(cfg["data"]["ground_truth"], emotions=emotions, text_column=cfg["data"]["text_column"])
    hub_id = encoder_hub_id(cfg, cfg["training"]["encoder"])
    out_dir = resolve_path(cfg["project"]["output_dir"]) / "encoder_binary" / cfg["training"]["encoder"]
    out_dir.mkdir(parents=True, exist_ok=True)
    resume = bool(cfg.get("experiment", {}).get("resume", resume))

    run = ExperimentRun(
        out_dir,
        name=f"encoder_binary:{cfg['training']['encoder']}",
        cfg=cfg,
        resume=resume,
        extra_meta={"hub_id": hub_id},
    )

    if dry_run:
        print(f"[dry-run] binary ensemble encoder={hub_id} folds={cfg['evaluation']['n_folds']} n={len(df)}")
        run.finalize(status="dry_run")
        return []

    tokenizer = AutoTokenizer.from_pretrained(hub_id)
    summary = _assemble_binary_on_multilabel_folds(cfg, hub_id, tokenizer, run=run, resume=resume)
    ExperimentRun._atomic_write_json(out_dir / "cv_summary.json", {"folds": summary})
    run.finalize(status="completed", n_folds=len(summary))
    return summary


def _assemble_binary_on_multilabel_folds(
    cfg,
    hub_id,
    tokenizer,
    *,
    run: ExperimentRun,
    resume: bool = True,
):
    """Train binary heads on shared multilabel folds and apply top-k assembly."""
    emotions = cfg["data"]["emotions"]
    df = load_ground_truth(cfg["data"]["ground_truth"], emotions=emotions, text_column=cfg["data"]["text_column"])
    y = label_matrix(df, emotions)
    n_folds = int(cfg["evaluation"]["n_folds"])
    seed = int(cfg["project"]["seed"])
    max_labels = int(cfg["evaluation"]["max_labels"])
    summary = []
    out_root = run.out_dir

    for fold, (train_idx, val_idx) in enumerate(multilabel_folds(y, n_folds, seed), start=1):
        fold_dir = out_root / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        if run.fold_done(fold):
            metrics = run.load_fold_metrics(fold)
            if metrics:
                clean = {k: v for k, v in metrics.items() if k != "_meta"}
                summary.append({"fold": fold, "metrics": clean})
                print(f"[resume] skip binary fold {fold}")
                continue

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        probs_path = fold_dir / "probs.npy"
        done_path = fold_dir / "emotions_done.json"

        probs = np.zeros((len(val_df), len(emotions)), dtype=np.float32)
        done: list[str] = []
        if resume and probs_path.exists() and done_path.exists():
            try:
                probs = np.load(probs_path)
                done = json.loads(done_path.read_text(encoding="utf-8"))
                print(f"[resume] fold {fold}: emotions done={done}")
            except Exception:  # noqa: BLE001
                probs = np.zeros((len(val_df), len(emotions)), dtype=np.float32)
                done = []

        t0 = timed()
        run.log_event("fold_started", fold=fold, head="binary", n_val=len(val_df))

        for ei, emotion in enumerate(emotions):
            if emotion in done:
                continue
            model = AutoModelForSequenceClassification.from_pretrained(hub_id, num_labels=2)
            train_ds = Dataset.from_dict(
                {"text": train_df["sentence"].tolist(), "labels": train_df[emotion].astype(int).tolist()}
            )
            val_ds = Dataset.from_dict(
                {"text": val_df["sentence"].tolist(), "labels": val_df[emotion].astype(int).tolist()}
            )

            def tok(batch):
                return tokenizer(batch["text"], truncation=True, max_length=int(cfg["training"]["max_length"]))

            train_ds = train_ds.map(tok, batched=True, remove_columns=["text"])
            val_ds = val_ds.map(tok, batched=True, remove_columns=["text"])
            emotion_dir = fold_dir / f"emotion_{emotion}"
            args = TrainingArguments(
                output_dir=str(emotion_dir),
                num_train_epochs=float(cfg["training"]["epochs"]),
                learning_rate=float(cfg["training"]["learning_rate"]),
                per_device_train_batch_size=int(cfg["training"]["train_batch_size"]),
                per_device_eval_batch_size=int(cfg["training"]["eval_batch_size"]),
                report_to=[],
                save_strategy="epoch",
                save_total_limit=1,
            )
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                processing_class=tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer),
            )
            try:
                trainer.train()
                logits = trainer.predict(val_ds).predictions
                probs[:, ei] = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
            finally:
                _cleanup_trainer_artifacts(emotion_dir)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            done.append(emotion)
            np.save(probs_path, probs)
            ExperimentRun._atomic_write_json(done_path, done)
            run.log_event("binary_emotion_done", fold=fold, emotion=emotion)

        raw = (probs >= 0.5).astype(int)
        preds = np.zeros_like(raw)
        for i in range(len(val_df)):
            pos = np.where(raw[i] == 1)[0]
            if len(pos) == 0:
                pos = np.array([int(np.argmax(probs[i]))])
            if len(pos) > max_labels:
                pos = pos[np.argsort(-probs[i, pos])[:max_labels]]
            preds[i, pos] = 1
        metrics = compute_metrics(label_matrix(val_df, emotions), preds, emotions)
        elapsed = timed() - t0
        summary.append({"fold": fold, "metrics": metrics})
        run.mark_fold_done(fold, metrics, elapsed_s=elapsed)
        ExperimentRun._atomic_write_json(out_root / "cv_summary.json", {"folds": summary})
        _cleanup_trainer_artifacts(fold_dir)
        print(f"binary fold {fold} done in {elapsed:.1f}s macro_f1={metrics['macro_f1']:.4f}")

    return summary


def _write_cv_summary(
    results: list[FoldResult],
    path: Path,
    emotions: list[str],
    *,
    run: ExperimentRun | None = None,
) -> None:
    agg = {}
    keys = ["subset_accuracy", "micro_f1", "macro_f1", "weighted_f1"]
    for k in keys:
        vals = [r.metrics[k] for r in results]
        agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    per = {}
    for e in emotions:
        f1s = [r.metrics["per_emotion"][e]["f1"] for r in results]
        per[e] = {"f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s))}
    payload = {
        "folds": [{"fold": r.fold, **r.metrics} for r in results],
        "aggregate": agg,
        "per_emotion": per,
        "n_folds_completed": len(results),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if run is not None:
        payload["usage"] = run.usage_totals.as_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    ExperimentRun._atomic_write_json(path, payload)
