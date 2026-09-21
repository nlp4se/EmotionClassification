"""Decoder-only emotion classification via TypeSafe AI's Jev model (System One).

A parallel, self-contained complement to :mod:`emotion_cls.decoding.classify`
(RQ2), evaluating the same three prompting strategies on the same
cross-validation folds -- but through TypeSafe's structured System One API
rather than a chat/completion model, so it does not go through
:class:`~emotion_cls.decoding.clients.LLMClient` / ``build_client`` at all.
Nothing in :mod:`clients`, :mod:`classify`, or :mod:`prompts` is imported for
mutation, only the shared guideline-text helpers.

Key differences from the RQ2 decoders, and why:

- **One Noul (yes/no, 0-1) question per emotion, in a single call per
  sentence**, rather than a free-form "reply with JSON" completion. This
  mirrors the multi-label decision the rest of the paper already makes
  (independent per-emotion evidence), and returns a continuous [0, 1] value
  per emotion, so the *same* ``threshold_topk`` decision rule used
  throughout RQ1/RQ2/RQ4 (theta=0.5, cap at 3, single-label fallback)
  applies unchanged -- no separate decision rule to justify.
- **No temperature dimension.** System One's Noul primitive does not expose
  a sampling-temperature parameter; there is nothing to sweep. Runs once per
  strategy, tagged ``t_na`` in the output path (vs. ``t0``/``t0.3``/``t0.7``
  for the other decoders) so this is visible rather than silently implied.
- **No batching.** The Noul question schema answers a fixed set of named
  questions once per call; it has no per-array-item indexing the way the
  chat-based batch prompt does, so each sentence is its own call. This
  matches the *default* ``batch_size=1`` the other decoders also use unless
  overridden, so it is not a methodological disadvantage relative to them.

Resumable the same way as RQ2: completed folds are skipped; within a fold,
finished sentence indices are skipped and predictions are flushed after
every call.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from emotion_cls.config import decoder_spec, load_env, require_api_key, resolve_path
from emotion_cls.data.dataset import label_matrix, load_ground_truth, multilabel_folds
from emotion_cls.decoding.prompts import normalize_strategy
from emotion_cls.experiment import ChatResult, ExperimentRun, TokenUsage, timed, utc_now
from emotion_cls.guidelines import (
    emotion_section,
    general_plutchik_definitions,
    load_guidelines_text,
    neutral_definition,
    strip_guideline_examples,
)
from emotion_cls.training.metrics import compute_metrics, threshold_topk

BACKEND = "typesafe"


def _emotion_instructions(
    emotion: str,
    strategy: str,
    guidelines_text: str,
    general_defs: str,
    few_shot_examples: list[str] | None,
) -> str:
    """Per-emotion Noul ``instructions`` text for one strategy.

    Reuses the exact same guideline-text helpers RQ2's prompt builder uses
    (:mod:`emotion_cls.guidelines`), so "zero-shot" / "few-shot guidelines"
    / "few-shot guidelines+dataset" mean the same thing here as they do for
    every other decoder -- only the delivery mechanism (Noul question vs.
    chat message) differs.
    """
    if emotion == "Neutral":
        body = neutral_definition()
    else:
        body = emotion_section(emotion, guidelines_text)
        if strategy == "zero_shot":
            body = strip_guideline_examples(body)

    parts = [general_defs, body]
    if strategy == "few_shot_guidelines_dataset" and few_shot_examples:
        examples_block = "\n".join(f"- {ex}" for ex in few_shot_examples)
        parts.append(
            "Additional labelled examples from the training fold "
            f"(illustration only, follow the definition above):\n{examples_block}"
        )
    return "\n\n".join(p for p in parts if p and p.strip())


def _build_client(spec: dict[str, Any]):
    from typesafe_sdk import TypeSafeClient

    api_key = require_api_key(spec.get("api_key_env", "TYPESAFE_API_KEY"))
    return TypeSafeClient(api_key=api_key, model=spec["model_id"])


def _classify_sentence(client, sentence: str, review_context: str | None, emotions: list[str],
                        strategy: str, guidelines_text: str, general_defs: str,
                        few: dict[str, list[str]] | None) -> tuple[dict[str, float], ChatResult]:
    from typesafe_sdk import Noul

    state: Any = sentence
    if review_context and review_context.strip() and review_context.strip() != sentence.strip():
        state = {"sentence": sentence, "review": review_context.strip()}

    questions = {
        e: Noul(
            instructions=_emotion_instructions(
                e, strategy, guidelines_text, general_defs,
                (few or {}).get(e),
            )
        )
        for e in emotions
    }

    t0 = time.perf_counter()
    response = client.system_one(state=state, questions=questions)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    probs = {e: float(response.nouls[e].noul) for e in emotions}
    usage = TokenUsage(
        prompt_tokens=response.usage.input_tokens,
        completion_tokens=response.usage.output_tokens,
        total_tokens=(
            (response.usage.input_tokens or 0) + (response.usage.output_tokens or 0)
            if response.usage.input_tokens is not None or response.usage.output_tokens is not None
            else None
        ),
    )
    result = ChatResult(
        text=json.dumps(probs),
        usage=usage,
        latency_ms=latency_ms,
        model_id=response.model,
        backend=BACKEND,
    )
    return probs, result


def _pred_path(fold_dir: Path) -> Path:
    return fold_dir / "predictions.csv"


def _load_partial_predictions(path: Path, emotions: list[str]) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, OSError):
        return {}
    if "idx" not in df.columns:
        return {}
    out: dict[int, dict[str, Any]] = {}
    for _, row in df.iterrows():
        idx = int(row["idx"])
        probs = {}
        for e in emotions:
            col = f"prob_{e}"
            if col in df.columns and pd.notna(row.get(col)):
                probs[e] = float(row[col])
        out[idx] = {
            "idx": idx,
            "sentence": row.get("sentence", ""),
            "probs": probs,
            "error": row.get("error", "") if pd.notna(row.get("error", "")) else "",
            "timestamp": row.get("timestamp", ""),
            "prompt_tokens": row.get("prompt_tokens", ""),
            "completion_tokens": row.get("completion_tokens", ""),
            "total_tokens": row.get("total_tokens", ""),
            "latency_ms": row.get("latency_ms", ""),
            "model_id": row.get("model_id", ""),
        }
    return out


def _flush_predictions(path: Path, rows_by_idx: dict[int, dict[str, Any]], emotions: list[str]) -> None:
    rows = []
    for i in sorted(rows_by_idx):
        r = rows_by_idx[i]
        row = {
            "idx": r["idx"], "sentence": r["sentence"], "error": r.get("error", ""),
            "timestamp": r.get("timestamp", ""),
            "prompt_tokens": r.get("prompt_tokens", ""), "completion_tokens": r.get("completion_tokens", ""),
            "total_tokens": r.get("total_tokens", ""), "latency_ms": r.get("latency_ms", ""),
            "model_id": r.get("model_id", ""),
        }
        for e in emotions:
            row[f"prob_{e}"] = r.get("probs", {}).get(e, "")
        rows.append(row)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(path)


def run_typesafe_classification(
    cfg: dict[str, Any],
    decoder_key: str = "jev-latest",
    *,
    dry_run: bool = False,
    resume: bool = True,
) -> Path:
    load_env()
    emotions = cfg["data"]["emotions"]
    df = load_ground_truth(cfg["data"]["ground_truth"], emotions=emotions)
    y = label_matrix(df, emotions)
    strategy = normalize_strategy(cfg["decoding"]["strategy"])
    max_labels = int(cfg["evaluation"]["max_labels"])
    n_folds = int(cfg["evaluation"]["n_folds"])
    seed = int(cfg["project"]["seed"])
    few_k = int(cfg["decoding"].get("few_shot_k", 5))
    guidelines_path = cfg["data"].get("guidelines")
    review_col = cfg["data"].get("review_column", "review")
    resume = bool(cfg.get("experiment", {}).get("resume", resume))

    spec = decoder_spec(cfg, decoder_key)
    out_dir = (
        resolve_path(cfg["project"]["output_dir"])
        / "decoder_classify"
        / decoder_key
        / strategy
        / "t_na"  # no temperature dimension for this backend -- see module docstring
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    run = ExperimentRun(
        out_dir,
        name=f"decoder_classify_typesafe:{decoder_key}:{strategy}",
        cfg=cfg,
        resume=resume,
        extra_meta={"decoder": decoder_key, "strategy": strategy, "model": spec, "temperature": None},
    )

    if dry_run:
        print(f"[dry-run] typesafe classify model={spec['model_id']} strategy={strategy}")
        run.finalize(status="dry_run")
        return out_dir

    client = _build_client(spec)
    guidelines_text = load_guidelines_text(guidelines_path)
    general_defs = general_plutchik_definitions(guidelines_text)
    fold_metrics: list[dict[str, Any]] = []

    for fold, (train_idx, val_idx) in enumerate(multilabel_folds(y, n_folds, seed), start=1):
        fold_dir = out_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        pred_file = _pred_path(fold_dir)

        if run.fold_done(fold):
            metrics = run.load_fold_metrics(fold)
            if metrics:
                clean = {k: v for k, v in metrics.items() if k != "_meta"}
                fold_metrics.append(clean)
                print(f"[resume] skip fold {fold} (metrics present)")
                continue

        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx].reset_index(drop=True)
        few = None
        if strategy == "few_shot_guidelines_dataset":
            few = {e: train_df.loc[train_df[e] == 1, "sentence"].tolist()[:few_k] for e in emotions}

        rows_by_idx = _load_partial_predictions(pred_file, emotions) if resume else {}
        if rows_by_idx:
            print(f"[resume] fold {fold}: {len(rows_by_idx)}/{len(val_df)} sentences already done")

        t_fold = timed()
        run.log_event("fold_started", fold=fold, n_val=len(val_df), n_done=len(rows_by_idx))

        pending = [
            i for i in range(len(val_df))
            if not (i in rows_by_idx and rows_by_idx[i].get("probs") and not rows_by_idx[i].get("error"))
        ]

        for i in pending:
            sentence = str(val_df.at[i, "sentence"])
            review_context = str(val_df.at[i, review_col]) if review_col in val_df.columns else None
            unit_id = f"fold{fold}:idx{i}"
            try:
                probs, result = _classify_sentence(
                    client, sentence, review_context, emotions, strategy,
                    guidelines_text, general_defs, few,
                )
                run.log_llm_call(purpose="classify_typesafe", result=result, fold=fold, unit_id=unit_id)
                rows_by_idx[i] = {
                    "idx": i, "sentence": sentence, "probs": probs, "error": "",
                    "timestamp": utc_now(),
                    "prompt_tokens": result.usage.prompt_tokens or "",
                    "completion_tokens": result.usage.completion_tokens or "",
                    "total_tokens": result.usage.total_tokens or "",
                    "latency_ms": round(result.latency_ms, 2),
                    "model_id": result.model_id,
                }
            except Exception as exc:  # noqa: BLE001
                run.log_event("llm_error", fold=fold, unit_id=unit_id, error=str(exc))
                rows_by_idx[i] = {
                    "idx": i, "sentence": sentence, "probs": {}, "error": str(exc),
                    "timestamp": utc_now(), "prompt_tokens": "", "completion_tokens": "",
                    "total_tokens": "", "latency_ms": "", "model_id": spec.get("model_id", ""),
                }
            _flush_predictions(pred_file, rows_by_idx, emotions)

        prob_matrix = np.zeros((len(val_df), len(emotions)), dtype=float)
        for i in range(len(val_df)):
            probs = rows_by_idx.get(i, {}).get("probs", {})
            for j, e in enumerate(emotions):
                prob_matrix[i, j] = probs.get(e, 0.0)
        np.save(fold_dir / "probs.npy", prob_matrix)
        preds = threshold_topk(prob_matrix, k=max_labels)

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
