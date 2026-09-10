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

from emotion_cls.config import encoder_hub_id, encoder_training_overrides, resolve_path
from emotion_cls.data.dataset import (
    label_matrix,
    load_ground_truth,
    multilabel_folds,
    tuning_holdout_mask,
)
from emotion_cls.experiment import ExperimentRun, timed
from emotion_cls.imbalance.sampling import inject_synthetic, undersample_multilabel
from emotion_cls.losses import build_loss
from emotion_cls.training.metrics import compute_metrics, probs_from_logits, threshold_topk, top_k_from_logits


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



def _exclude_tuning_holdout(df: pd.DataFrame, emotions: list[str], cfg: dict[str, Any]) -> pd.DataFrame:
    """Drop the rows reserved for LR tuning (see `tune-encoder`) from the CV pool.

    Keeps the reported 10-fold CV disjoint from whatever data hyperparameter
    search looked at, so a tuned learning rate can't optimistically bias the
    reported metric. Controlled by ``tuning.enabled`` (default on); the same
    holdout mask (fixed seed) is shared by every encoder for a fair, apples-
    to-apples comparison.
    """
    tuning_cfg = cfg.get("tuning", {})
    if not bool(tuning_cfg.get("enabled", True)):
        return df
    y_full = label_matrix(df, emotions)
    mask = tuning_holdout_mask(
        y_full,
        n_splits=int(tuning_cfg.get("holdout_n_splits", 7)),
        seed=int(tuning_cfg.get("holdout_seed", 43)),
    )
    return df.loc[~mask].reset_index(drop=True)


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
    df = _exclude_tuning_holdout(df, emotions, cfg)
    y = label_matrix(df, emotions)
    hub_id = encoder_hub_id(cfg, cfg["training"]["encoder"])
    overrides = encoder_training_overrides(cfg, cfg["training"]["encoder"])
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
            learning_rate=float(overrides.get("learning_rate", cfg["training"]["learning_rate"])),
            per_device_train_batch_size=int(cfg["training"]["train_batch_size"]),
            per_device_eval_batch_size=int(cfg["training"]["eval_batch_size"]),
            weight_decay=float(cfg["training"]["weight_decay"]),
            warmup_ratio=float(cfg["training"].get("warmup_ratio", 0.0)),
            # HF default (1e-8) is too small for DeBERTa-v3's parameter scale and
            # reliably blows up to NaN loss after a single optimizer step; 1e-6
            # (what Microsoft's own DeBERTa fine-tuning scripts use) fixes it and
            # is a safe, standard value for the other encoder families too.
            adam_epsilon=float(cfg["training"].get("adam_epsilon", 1e-6)),
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
            probs = probs_from_logits(pred_out.predictions)
            preds = threshold_topk(probs, k=max_labels)
            metrics = compute_metrics(label_matrix(val_df, emotions), preds, emotions)
            elapsed = timed() - t0

            # Raw per-emotion probabilities, so the assembly rule (threshold,
            # cap, fallback) can be revisited later without retraining.
            np.save(fold_dir / "probs.npy", probs)

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


def tune_encoder_lr(
    cfg: dict[str, Any],
    *,
    dry_run: bool = False,
    resume: bool = True,
) -> dict[str, Any]:
    """Grid-search ``learning_rate`` for one encoder against the tuning holdout.

    Trains each candidate once on the full CV pool (holdout excluded, exactly
    like the real run) for the configured epoch budget, scores it on the
    holdout, and persists the winner to
    ``outputs/tuning/<encoder>/learning_rate.json``. That file is then read
    automatically by :func:`emotion_cls.config.encoder_training_overrides`,
    so a subsequent ``train-encoder`` call picks it up with no extra flags.
    Never touches the CV test folds -- see ``tuning_holdout_mask``.
    """
    emotions = cfg["data"]["emotions"]
    encoder_key = cfg["training"]["encoder"]
    hub_id = encoder_hub_id(cfg, encoder_key)
    tuning_cfg = cfg.get("tuning", {})

    df_full = load_ground_truth(cfg["data"]["ground_truth"], emotions=emotions, text_column=cfg["data"]["text_column"])
    y_full = label_matrix(df_full, emotions)
    mask = tuning_holdout_mask(
        y_full,
        n_splits=int(tuning_cfg.get("holdout_n_splits", 7)),
        seed=int(tuning_cfg.get("holdout_seed", 43)),
    )
    train_df = df_full.loc[~mask].reset_index(drop=True)
    val_df = df_full.loc[mask].reset_index(drop=True)
    grid = [float(x) for x in tuning_cfg.get("learning_rate_grid", [cfg["training"]["learning_rate"]])]

    out_dir = resolve_path(cfg["project"]["output_dir"]) / "tuning" / encoder_key
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "learning_rate.json"

    if dry_run:
        print(f"[dry-run] tune-encoder encoder={hub_id} grid={grid} n_train={len(train_df)} n_val={len(val_df)}")
        return {}

    if resume and result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        print(f"[resume] tune-encoder {encoder_key}: already tuned, best_lr={existing['best_learning_rate']:g}")
        return existing

    tokenizer = AutoTokenizer.from_pretrained(hub_id)
    max_labels = int(cfg["evaluation"]["max_labels"])
    max_length = int(cfg["training"]["max_length"])
    train_ds = _tokenize_dataset(train_df, emotions, tokenizer, max_length)
    val_ds = _tokenize_dataset(val_df, emotions, tokenizer, max_length)
    y_train = torch.tensor(label_matrix(train_df, emotions), dtype=torch.float32)
    loss_fn = build_loss("none", y_train, reduction="mean")
    seed = int(cfg["project"]["seed"])

    print(f"tune-encoder {encoder_key}: n_train={len(train_df)} n_val={len(val_df)} grid={grid}")
    trials: list[dict[str, Any]] = []
    for lr in grid:
        trial_dir = out_dir / f"trial_lr{lr:g}"
        model = AutoModelForSequenceClassification.from_pretrained(
            hub_id, num_labels=len(emotions), problem_type="multi_label_classification"
        )
        args = TrainingArguments(
            output_dir=str(trial_dir),
            num_train_epochs=float(cfg["training"]["epochs"]),
            learning_rate=lr,
            per_device_train_batch_size=int(cfg["training"]["train_batch_size"]),
            per_device_eval_batch_size=int(cfg["training"]["eval_batch_size"]),
            weight_decay=float(cfg["training"]["weight_decay"]),
            warmup_ratio=float(cfg["training"].get("warmup_ratio", 0.0)),
            adam_epsilon=float(cfg["training"].get("adam_epsilon", 1e-6)),
            eval_strategy="no",
            save_strategy="no",
            report_to=[],
            seed=seed,
            disable_tqdm=True,
            logging_strategy="no",
            fp16=bool(cfg["training"].get("fp16", False)),
        )
        t0 = timed()
        try:
            trainer = MultilabelTrainer(
                model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer), loss_fn=loss_fn,
            )
            trainer.train()
            pred_out = trainer.predict(val_ds)
            probs = probs_from_logits(pred_out.predictions)
            preds = threshold_topk(probs, k=max_labels)
            metrics = compute_metrics(label_matrix(val_df, emotions), preds, emotions)
            elapsed = timed() - t0
            trials.append({"learning_rate": lr, "macro_f1": metrics["macro_f1"], "micro_f1": metrics["micro_f1"], "elapsed_s": elapsed})
            print(f"  lr={lr:g}: macro_f1={metrics['macro_f1']:.4f} micro_f1={metrics['micro_f1']:.4f} ({elapsed:.1f}s)")
        finally:
            shutil.rmtree(trial_dir, ignore_errors=True)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    best = max(trials, key=lambda t: t["macro_f1"])
    result = {
        "encoder": encoder_key,
        "hub_id": hub_id,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "trials": trials,
        "best_learning_rate": best["learning_rate"],
        "best_macro_f1": best["macro_f1"],
    }
    ExperimentRun._atomic_write_json(result_path, result)
    print(f"tune-encoder {encoder_key}: best_lr={best['learning_rate']:g} (macro_f1={best['macro_f1']:.4f}) -> {result_path}")
    return result


def run_binary_ensemble_cv(
    cfg: dict[str, Any],
    *,
    dry_run: bool = False,
    resume: bool = True,
) -> list[dict[str, Any]]:
    """Train one binary classifier per emotion on shared multilabel folds; assemble with top-k.

    Imbalance mitigation (loss reweighting, undersampling, generative augmentation) is applied
    identically to :func:`run_multilabel_cv`, just with a per-emotion K=1 target instead of the
    shared K=|E| one: each binary classifier reuses the exact same loss classes from
    ``emotion_cls.losses`` (unchanged formulas), and the same undersample/inject_synthetic calls
    on the shared training-fold pool before the per-emotion split.
    """
    emotions = cfg["data"]["emotions"]
    df = load_ground_truth(cfg["data"]["ground_truth"], emotions=emotions, text_column=cfg["data"]["text_column"])
    df = _exclude_tuning_holdout(df, emotions, cfg)
    hub_id = encoder_hub_id(cfg, cfg["training"]["encoder"])
    imb = cfg.get("imbalance", {})
    method = str(imb.get("method", "none"))
    out_dir = resolve_path(cfg["project"]["output_dir"]) / "encoder_binary" / cfg["training"]["encoder"]
    if method not in {"none", "baseline", ""}:
        # "none" keeps the pre-existing flat directory (baseline already computed there);
        # every other method gets its own subdirectory, mirroring run_multilabel_cv.
        out_dir = out_dir / method.replace(",", "+")
    out_dir.mkdir(parents=True, exist_ok=True)
    resume = bool(cfg.get("experiment", {}).get("resume", resume))

    run = ExperimentRun(
        out_dir,
        name=f"encoder_binary:{cfg['training']['encoder']}:{method}",
        cfg=cfg,
        resume=resume,
        extra_meta={"hub_id": hub_id, "imbalance": method},
    )

    if dry_run:
        print(
            f"[dry-run] binary ensemble encoder={hub_id} folds={cfg['evaluation']['n_folds']} "
            f"imbalance={method} n={len(df)}"
        )
        run.finalize(status="dry_run")
        return []

    tokenizer = AutoTokenizer.from_pretrained(hub_id)
    fold_results = _assemble_binary_on_multilabel_folds(cfg, hub_id, tokenizer, run=run, resume=resume)
    _write_cv_summary(fold_results, out_dir / "cv_summary.json", emotions, run=run)
    run.finalize(status="completed", n_folds=len(fold_results))
    return [{"fold": r.fold, "metrics": r.metrics} for r in fold_results]


def _assemble_binary_on_multilabel_folds(
    cfg,
    hub_id,
    tokenizer,
    *,
    run: ExperimentRun,
    resume: bool = True,
) -> list[FoldResult]:
    """Train binary heads on shared multilabel folds and apply top-k assembly.

    Each per-emotion classifier is a single-logit (num_labels=1) sigmoid head trained with
    BCEWithLogits-family losses from ``emotion_cls.losses`` via ``MultilabelTrainer`` with K=1 --
    the exact same loss classes/formulas used by ``run_multilabel_cv``, just evaluated on one
    emotion's target instead of all nine jointly. This keeps "the same imbalance method" literal
    rather than reimplementing a parallel softmax/cross-entropy variant. Data-level methods
    (undersample, genai_aug) are applied to the shared per-fold training pool before the
    per-emotion split, identically to the multi-label path.
    """
    emotions = cfg["data"]["emotions"]
    df = load_ground_truth(cfg["data"]["ground_truth"], emotions=emotions, text_column=cfg["data"]["text_column"])
    df = _exclude_tuning_holdout(df, emotions, cfg)
    y = label_matrix(df, emotions)
    overrides = encoder_training_overrides(cfg, cfg["training"]["encoder"])
    n_folds = int(cfg["evaluation"]["n_folds"])
    seed = int(cfg["project"]["seed"])
    max_labels = int(cfg["evaluation"]["max_labels"])

    imb = cfg.get("imbalance", {})
    method = str(imb.get("method", "none"))
    methods = [m.strip() for m in method.split(",") if m.strip()]
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

    loss_name = next(
        (m for m in methods_norm if m in {"none", "bce_pos_weight", "bce_weight", "focal", "adaptive_focal"}),
        "none",
    )

    results: list[FoldResult] = []
    out_root = run.out_dir

    for fold, (train_idx, val_idx) in enumerate(multilabel_folds(y, n_folds, seed), start=1):
        fold_dir = out_root / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        if run.fold_done(fold):
            metrics = run.load_fold_metrics(fold)
            if metrics:
                clean = {k: v for k, v in metrics.items() if k != "_meta"}
                results.append(FoldResult(fold=fold, metrics=clean))
                print(f"[resume] skip binary fold {fold}")
                continue

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        if "undersample" in methods_norm and imb.get("undersample_cutoff"):
            train_df = undersample_multilabel(
                train_df, emotions, int(imb["undersample_cutoff"]), seed=seed + fold
            )
        if "genai_aug" in methods_norm and synthetic_ml is not None:
            train_df = inject_synthetic(
                train_df, synthetic_ml, emotions, imb.get("aug_inject_n"),
            )

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
        run.log_event("fold_started", fold=fold, head="binary", n_val=len(val_df), imbalance=method)

        for ei, emotion in enumerate(emotions):
            if emotion in done:
                continue

            y_col = torch.tensor(train_df[emotion].astype(float).values, dtype=torch.float32).unsqueeze(1)
            loss_fn = build_loss(
                loss_name,
                y_col,
                reduction=str(imb.get("reduction", "mean")),
                focal_gamma=float(imb.get("focal_gamma", 2.0)),
            )

            model = AutoModelForSequenceClassification.from_pretrained(
                hub_id, num_labels=1, problem_type="multi_label_classification"
            )
            train_ds = Dataset.from_dict(
                {
                    "text": train_df["sentence"].tolist(),
                    "labels": [[float(v)] for v in train_df[emotion].astype(int).tolist()],
                }
            )
            val_ds = Dataset.from_dict(
                {
                    "text": val_df["sentence"].tolist(),
                    "labels": [[float(v)] for v in val_df[emotion].astype(int).tolist()],
                }
            )

            def tok(batch):
                return tokenizer(batch["text"], truncation=True, max_length=int(cfg["training"]["max_length"]))

            train_ds = train_ds.map(tok, batched=True, remove_columns=["text"])
            val_ds = val_ds.map(tok, batched=True, remove_columns=["text"])
            emotion_dir = fold_dir / f"emotion_{emotion}"
            args = TrainingArguments(
                output_dir=str(emotion_dir),
                num_train_epochs=float(cfg["training"]["epochs"]),
                learning_rate=float(overrides.get("learning_rate", cfg["training"]["learning_rate"])),
                per_device_train_batch_size=int(cfg["training"]["train_batch_size"]),
                per_device_eval_batch_size=int(cfg["training"]["eval_batch_size"]),
                warmup_ratio=float(cfg["training"].get("warmup_ratio", 0.0)),
                adam_epsilon=float(cfg["training"].get("adam_epsilon", 1e-6)),
                report_to=[],
                save_strategy="epoch",
                save_total_limit=1,
            )
            trainer = MultilabelTrainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                processing_class=tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer),
                loss_fn=loss_fn,
            )
            try:
                trainer.train()
                logits = trainer.predict(val_ds).predictions
                probs[:, ei] = torch.sigmoid(torch.tensor(logits)).numpy()[:, 0]
            finally:
                _cleanup_trainer_artifacts(emotion_dir)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            done.append(emotion)
            np.save(probs_path, probs)
            ExperimentRun._atomic_write_json(done_path, done)
            run.log_event("binary_emotion_done", fold=fold, emotion=emotion)

        preds = threshold_topk(probs, k=max_labels)
        metrics = compute_metrics(label_matrix(val_df, emotions), preds, emotions)
        elapsed = timed() - t0
        results.append(FoldResult(fold=fold, metrics=metrics))
        run.mark_fold_done(fold, metrics, elapsed_s=elapsed)
        _write_cv_summary(results, out_root / "cv_summary.json", emotions, run=run)
        _cleanup_trainer_artifacts(fold_dir)
        print(f"binary fold {fold} done in {elapsed:.1f}s macro_f1={metrics['macro_f1']:.4f} imbalance={method}")

    return results


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
