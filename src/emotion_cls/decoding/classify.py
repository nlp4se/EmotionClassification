"""Decoder-only zero/few-shot emotion classification (RQ2).

Resumable: completed folds are skipped; within a fold, finished sentence indices
are skipped and predictions are flushed after every call.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from emotion_cls.config import decoder_spec, resolve_path
from emotion_cls.data.dataset import label_matrix, load_ground_truth, multilabel_folds
from emotion_cls.decoding.clients import build_client, parse_json_payload
from emotion_cls.decoding.ollama_lifecycle import maybe_ensure_from_spec, maybe_remove_from_spec
from emotion_cls.decoding.prompts import classification_messages, normalize_strategy
from emotion_cls.experiment import ExperimentRun, timed, utc_now
from emotion_cls.training.metrics import compute_metrics


def _normalize_labels(raw: Any, emotions: list[str], max_labels: int) -> list[str]:
    if isinstance(raw, dict):
        raw = raw.get("emotions") or raw.get("labels") or []
    if not isinstance(raw, list):
        return []
    allowed = {e.lower(): e for e in emotions}
    out = []
    for item in raw:
        key = str(item).strip().lower()
        if key in allowed and allowed[key] not in out:
            out.append(allowed[key])
        if len(out) >= max_labels:
            break
    return out


def _pred_path(fold_dir: Path) -> Path:
    return fold_dir / "predictions.csv"


def _load_partial_predictions(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "idx" not in df.columns:
        return {}
    out: dict[int, dict[str, Any]] = {}
    for _, row in df.iterrows():
        idx = int(row["idx"])
        pred = row.get("pred", "[]")
        if isinstance(pred, str):
            try:
                pred_list = ast.literal_eval(pred)
            except (ValueError, SyntaxError):
                pred_list = []
        else:
            pred_list = list(pred) if pred is not None else []
        out[idx] = {
            "idx": idx,
            "sentence": row.get("sentence", ""),
            "pred": pred_list,
            "error": row.get("error", "") if pd.notna(row.get("error", "")) else "",
            "raw": row.get("raw", "") if pd.notna(row.get("raw", "")) else "",
            "timestamp": row.get("timestamp", ""),
            "prompt_tokens": row.get("prompt_tokens", ""),
            "completion_tokens": row.get("completion_tokens", ""),
            "total_tokens": row.get("total_tokens", ""),
            "latency_ms": row.get("latency_ms", ""),
            "model_id": row.get("model_id", ""),
            "backend": row.get("backend", ""),
        }
    return out


def _flush_predictions(path: Path, rows_by_idx: dict[int, dict[str, Any]]) -> None:
    rows = [rows_by_idx[i] for i in sorted(rows_by_idx)]
    # Serialize list preds as JSON strings for CSV safety
    serializable = []
    for r in rows:
        item = dict(r)
        if isinstance(item.get("pred"), list):
            item["pred"] = json.dumps(item["pred"])
        serializable.append(item)
    pd.DataFrame(serializable).to_csv(path, index=False)


def run_decoder_classification(
    cfg: dict[str, Any],
    decoder_key: str,
    *,
    dry_run: bool = False,
    resume: bool = True,
    unload_ollama: bool | None = None,
) -> Path:
    emotions = cfg["data"]["emotions"]
    df = load_ground_truth(cfg["data"]["ground_truth"], emotions=emotions)
    y = label_matrix(df, emotions)
    strategy = normalize_strategy(cfg["decoding"]["strategy"])
    max_labels = int(cfg["evaluation"]["max_labels"])
    n_folds = int(cfg["evaluation"]["n_folds"])
    seed = int(cfg["project"]["seed"])
    few_k = int(cfg["decoding"].get("few_shot_k", 5))
    temperature = float(cfg["decoding"].get("temperature", 0.0))
    guidelines_path = cfg["data"].get("guidelines")
    review_col = cfg["data"].get("review_column", "review")
    resume = bool(cfg.get("experiment", {}).get("resume", resume))

    spec = decoder_spec(cfg, decoder_key)
    # Isolate temperature in the output path so sweeps are independently resumable
    temp_tag = f"t{temperature:g}"
    out_dir = (
        resolve_path(cfg["project"]["output_dir"])
        / "decoder_classify"
        / decoder_key
        / strategy
        / temp_tag
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    run = ExperimentRun(
        out_dir,
        name=f"decoder_classify:{decoder_key}:{strategy}:{temp_tag}",
        cfg=cfg,
        resume=resume,
        extra_meta={
            "decoder": decoder_key,
            "strategy": strategy,
            "temperature": temperature,
            "model": spec,
        },
    )

    if dry_run:
        print(
            f"[dry-run] decoder classify model={spec['model_id']} "
            f"backend={spec['backend']} strategy={strategy} temperature={temperature} "
            f"guidelines={guidelines_path}"
        )
        run.finalize(status="dry_run")
        return out_dir

    maybe_ensure_from_spec(spec, cfg)
    try:
        client = build_client(spec)
        fold_metrics: list[dict[str, Any]] = []

        for fold, (train_idx, val_idx) in enumerate(multilabel_folds(y, n_folds, seed), start=1):
            fold_dir = out_dir / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            pred_file = _pred_path(fold_dir)

            if run.fold_done(fold):
                metrics = run.load_fold_metrics(fold)
                if metrics:
                    # strip internal _meta for summary aggregates if present
                    clean = {k: v for k, v in metrics.items() if k != "_meta"}
                    fold_metrics.append(clean)
                    print(f"[resume] skip fold {fold} (metrics present)")
                    continue

            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx].reset_index(drop=True)
            few = None
            if strategy == "few_shot_guidelines_dataset":
                few = {}
                for e in emotions:
                    ex = train_df.loc[train_df[e] == 1, "sentence"].tolist()[:few_k]
                    few[e] = ex

            rows_by_idx = _load_partial_predictions(pred_file) if resume else {}
            if rows_by_idx:
                print(f"[resume] fold {fold}: {len(rows_by_idx)}/{len(val_df)} sentences already done")

            t_fold = timed()
            run.log_event("fold_started", fold=fold, n_val=len(val_df), n_done=len(rows_by_idx))

            for i in range(len(val_df)):
                if i in rows_by_idx and not rows_by_idx[i].get("error"):
                    # Keep successful rows; retry rows that had parse/API errors
                    if rows_by_idx[i].get("pred") not in (None, "", "[]", []):
                        continue
                    # empty pred without error might be legitimate; skip if timestamp set
                    if rows_by_idx[i].get("timestamp") and not rows_by_idx[i].get("error"):
                        continue

                sentence = str(val_df.at[i, "sentence"])
                review_context = (
                    str(val_df.at[i, review_col])
                    if review_col in val_df.columns
                    else None
                )
                messages = classification_messages(
                    sentence,
                    emotions,
                    strategy,
                    few_shot_examples=few,
                    guidelines_path=guidelines_path,
                    review_context=review_context,
                )
                try:
                    result = client.chat(messages, temperature=temperature)
                    run.log_llm_call(
                        purpose="classify",
                        result=result,
                        fold=fold,
                        unit_id=f"fold{fold}:idx{i}",
                    )
                    try:
                        payload = parse_json_payload(result.text)
                        labels = _normalize_labels(payload, emotions, max_labels)
                        err = ""
                        raw = ""
                    except Exception as exc:  # noqa: BLE001
                        labels = []
                        err = str(exc)
                        raw = result.text[:500]
                    rows_by_idx[i] = {
                        "idx": i,
                        "sentence": sentence,
                        "pred": labels,
                        "error": err,
                        "raw": raw,
                        "timestamp": utc_now(),
                        "prompt_tokens": result.usage.prompt_tokens,
                        "completion_tokens": result.usage.completion_tokens,
                        "total_tokens": result.usage.total_tokens,
                        "latency_ms": round(result.latency_ms, 2),
                        "model_id": result.model_id,
                        "backend": result.backend,
                    }
                except Exception as exc:  # noqa: BLE001
                    run.log_event("llm_error", fold=fold, unit_id=f"fold{fold}:idx{i}", error=str(exc))
                    rows_by_idx[i] = {
                        "idx": i,
                        "sentence": sentence,
                        "pred": [],
                        "error": str(exc),
                        "raw": "",
                        "timestamp": utc_now(),
                        "prompt_tokens": "",
                        "completion_tokens": "",
                        "total_tokens": "",
                        "latency_ms": "",
                        "model_id": spec.get("model_id", ""),
                        "backend": spec.get("backend", ""),
                    }

                _flush_predictions(pred_file, rows_by_idx)

            preds = np.zeros((len(val_df), len(emotions)), dtype=int)
            for i in range(len(val_df)):
                labels = rows_by_idx.get(i, {}).get("pred", [])
                if isinstance(labels, str):
                    try:
                        labels = ast.literal_eval(labels)
                    except (ValueError, SyntaxError):
                        labels = []
                for lab in labels or []:
                    if lab in emotions:
                        preds[i, emotions.index(lab)] = 1

            metrics = compute_metrics(label_matrix(val_df, emotions), preds, emotions)
            elapsed = timed() - t_fold
            run.mark_fold_done(fold, metrics, elapsed_s=elapsed)
            fold_metrics.append(metrics)
            print(f"fold {fold} done in {elapsed:.1f}s macro_f1={metrics['macro_f1']:.4f}")

        summary = {
            "folds": fold_metrics,
            "aggregate": _aggregate_fold_metrics(fold_metrics),
            "usage": run.usage_totals.as_dict(),
            "n_llm_calls": run.n_llm_calls,
            "total_latency_ms": run.total_latency_ms,
        }
        ExperimentRun._atomic_write_json(out_dir / "cv_summary.json", summary)
        run.finalize(status="completed", n_folds=len(fold_metrics))
        return out_dir
    finally:
        force = bool(unload_ollama) if unload_ollama is not None else False
        maybe_remove_from_spec(spec, cfg, force=force)

def _aggregate_fold_metrics(fold_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not fold_metrics:
        return {}
    keys = ["subset_accuracy", "micro_f1", "macro_f1", "weighted_f1"]
    agg = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics if k in m]
        if vals:
            agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return agg
