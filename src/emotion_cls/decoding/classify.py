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
from emotion_cls.decoding.prompts import (
    classification_messages_batch,
    normalize_strategy,
)
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


def _validate_batch_response(
    payload: Any,
    expected_ids: list[int],
    emotions: list[str],
    max_labels: int,
) -> tuple[dict[int, list[str]] | None, str]:
    """Validate + parse a batch classification response.

    Returns ``(id -> labels, "")`` on success, or ``(None, reason)`` if the
    response doesn't structurally match what was asked for. Deliberately
    strict on *structure* (array of the right length, every requested id
    present exactly once) since a structural mismatch means we can't safely
    attribute any result to any input; loose on *content* (an item with a
    garbled or missing "emotions" field just yields an empty label list for
    that one id via `_normalize_labels`, rather than failing the batch).
    """
    if isinstance(payload, dict):
        # A model that ignores "array only" and wraps it -- tolerate the common shapes.
        for key in ("results", "items", "classifications", "predictions"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        return None, f"expected a JSON array of {len(expected_ids)} items, got {type(payload).__name__}"
    if len(payload) != len(expected_ids):
        return None, f"expected {len(expected_ids)} items, got {len(payload)}"

    expected_set = set(expected_ids)
    seen: set[int] = set()
    result: dict[int, list[str]] = {}
    for item in payload:
        if not isinstance(item, dict) or "id" not in item:
            return None, f"item missing 'id' field: {item!r}"[:300]
        try:
            item_id = int(item["id"])
        except (TypeError, ValueError):
            return None, f"non-integer id: {item.get('id')!r}"[:300]
        if item_id not in expected_set:
            return None, f"id {item_id} not in this batch's requested ids"
        if item_id in seen:
            return None, f"duplicate id {item_id} in response"
        seen.add(item_id)
        result[item_id] = _normalize_labels(item, emotions, max_labels)

    if seen != expected_set:
        return None, f"missing ids in response: {sorted(expected_set - seen)}"
    return result, ""


def _pred_path(fold_dir: Path) -> Path:
    return fold_dir / "predictions.csv"


def _predictions_error_rate(path: Path) -> float:
    """Fraction of rows in predictions.csv with a recorded error.

    A fold that finishes (every sentence got *some* response, even a failed
    one) still gets metrics.json written, so fold-level resume treats it as
    "done" forever -- even if every response was an API error (e.g. a run
    against a since-fixed bad model id or an expired key). Used to reopen
    such folds instead of permanently locking in garbage results.
    """
    if not path.exists():
        return 0.0
    try:
        df = pd.read_csv(path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, OSError):
        return 1.0
    if "error" not in df.columns or len(df) == 0:
        return 0.0
    errs = df["error"].astype(str).replace("nan", "")
    return float((errs.str.len() > 0).mean())


def _load_partial_predictions(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, OSError) as exc:
        # A crash mid-write (server crash, OOM-kill, ...) can leave a truncated
        # CSV. Don't brick resume over it: treat the fold as if no partial
        # predictions were saved and re-classify it from sentence 0. Costs a
        # re-run of this one fold, not the whole experiment.
        print(f"[resume] WARNING: {path} is unreadable ({exc!r}); restarting this fold's predictions")
        return {}
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
    # Flushed after every sentence, so this is the file most likely to be
    # mid-write if the process is killed. Write to a temp file and rename
    # (atomic on the same filesystem) instead of writing `path` in place, so
    # a crash never leaves a truncated/corrupt predictions.csv behind.
    tmp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(serializable).to_csv(tmp, index=False)
    tmp.replace(path)


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
    batch_size = max(1, int(cfg["decoding"].get("batch_size", 1)))
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
                error_rate = _predictions_error_rate(pred_file)
                if metrics and error_rate <= 0.5:
                    # strip internal _meta for summary aggregates if present
                    clean = {k: v for k, v in metrics.items() if k != "_meta"}
                    fold_metrics.append(clean)
                    print(f"[resume] skip fold {fold} (metrics present)")
                    continue
                if metrics:
                    print(
                        f"[resume] fold {fold}: metrics present but {error_rate:.0%} of "
                        f"predictions errored -- reopening to retry failed sentences "
                        f"instead of keeping stale results"
                    )

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

            pending: list[int] = []
            for i in range(len(val_df)):
                if i in rows_by_idx and not rows_by_idx[i].get("error"):
                    # Keep successful rows; retry rows that had parse/API errors
                    if rows_by_idx[i].get("pred") not in (None, "", "[]", []):
                        continue
                    # empty pred without error might be legitimate; skip if timestamp set
                    if rows_by_idx[i].get("timestamp") and not rows_by_idx[i].get("error"):
                        continue
                pending.append(i)

            for batch_start in range(0, len(pending), batch_size):
                batch_ids = pending[batch_start : batch_start + batch_size]
                sentences = {i: str(val_df.at[i, "sentence"]) for i in batch_ids}
                items = []
                for i in batch_ids:
                    review_context = (
                        str(val_df.at[i, review_col]) if review_col in val_df.columns else None
                    )
                    items.append({"id": i, "sentence": sentences[i], "review_context": review_context})

                messages = classification_messages_batch(
                    items,
                    emotions,
                    strategy,
                    few_shot_examples=few,
                    guidelines_path=guidelines_path,
                )
                unit_id = f"fold{fold}:ids{batch_ids[0]}-{batch_ids[-1]}"
                try:
                    result = client.chat(messages, temperature=temperature)
                    run.log_llm_call(
                        purpose="classify_batch",
                        result=result,
                        fold=fold,
                        unit_id=unit_id,
                        extra={"batch_size": len(batch_ids), "ids": batch_ids},
                    )
                    try:
                        payload = parse_json_payload(result.text)
                        id_to_labels, batch_err = _validate_batch_response(
                            payload, batch_ids, emotions, max_labels
                        )
                    except Exception as exc:  # noqa: BLE001
                        id_to_labels, batch_err = None, str(exc)

                    n = len(batch_ids)
                    if id_to_labels is None:
                        # Structural mismatch: can't safely attribute any result to any
                        # input, so the whole batch is retried next run (same philosophy
                        # as a per-sentence error, just at batch granularity).
                        for i in batch_ids:
                            rows_by_idx[i] = {
                                "idx": i, "sentence": sentences[i], "pred": [],
                                "error": f"batch validation failed: {batch_err}",
                                "raw": result.text[:500], "timestamp": utc_now(),
                                "prompt_tokens": "", "completion_tokens": "", "total_tokens": "",
                                "latency_ms": "", "model_id": result.model_id, "backend": result.backend,
                            }
                    else:
                        # Usage/latency are for the whole batch call; apportion evenly
                        # across its rows so a naive sum over predictions.csv still adds
                        # up to the true total. The authoritative, un-apportioned figure
                        # for this exact call lives in events.jsonl / usage_totals.json
                        # via log_llm_call above.
                        u = result.usage
                        share = lambda v: (v / n) if v is not None else ""  # noqa: E731
                        for i in batch_ids:
                            rows_by_idx[i] = {
                                "idx": i, "sentence": sentences[i], "pred": id_to_labels[i],
                                "error": "", "raw": "", "timestamp": utc_now(),
                                "prompt_tokens": share(u.prompt_tokens),
                                "completion_tokens": share(u.completion_tokens),
                                "total_tokens": share(u.total_tokens),
                                "latency_ms": round(result.latency_ms / n, 2),
                                "model_id": result.model_id, "backend": result.backend,
                            }
                except Exception as exc:  # noqa: BLE001
                    run.log_event("llm_error", fold=fold, unit_id=unit_id, error=str(exc))
                    for i in batch_ids:
                        rows_by_idx[i] = {
                            "idx": i, "sentence": sentences[i], "pred": [], "error": str(exc),
                            "raw": "", "timestamp": utc_now(),
                            "prompt_tokens": "", "completion_tokens": "", "total_tokens": "", "latency_ms": "",
                            "model_id": spec.get("model_id", ""), "backend": spec.get("backend", ""),
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
