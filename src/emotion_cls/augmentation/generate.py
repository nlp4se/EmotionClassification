"""Generate synthetic emotion reviews until a target count is reached (RQ4 / RQ3).

Resumable: existing output CSV rows are counted and generation continues until
the remaining quota is filled. Each successful batch is flushed to disk.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

from emotion_cls.config import decoder_spec, resolve_path
from emotion_cls.data.dataset import load_ground_truth, majority_count
from emotion_cls.decoding.clients import build_client, parse_json_payload
from emotion_cls.decoding.prompts import generation_messages, normalize_strategy
from emotion_cls.experiment import ExperimentRun, utc_now


def _empty_row(sentence: str, review: str, emotion: str, emotions: list[str]) -> dict[str, Any]:
    row = {
        "iteration": "",
        "app_name": "",
        "categoryId": "",
        "reviewId": "",
        "sentenceId": "",
        "at": "",
        "score": "",
        "feature": "",
        "review": review,
        "sentence": sentence,
        "generated_at": utc_now(),
    }
    for e in emotions + ["Reject"]:
        row[e] = 1 if e == emotion else 0
    return row


def generate_for_emotion(
    cfg: dict[str, Any],
    *,
    decoder_key: str,
    emotion: str,
    target_count: int | None = None,
    batch_size: int = 10,
    strategy: str | None = None,
    resume: bool = True,
) -> Path:
    """Generate until human+synthetic positives for `emotion` reach `target_count`.

    If target_count is None, use majority-class parity on the ground truth
    (among generation emotions).
    """
    emotions_all = cfg["data"]["emotions"]
    gen_emotions = cfg["data"]["generation_emotions"]
    gt = load_ground_truth(cfg["data"]["ground_truth"], emotions=emotions_all)
    human_pos = int(gt[emotion].sum())
    majority = majority_count(gt, gen_emotions)
    target = int(target_count if target_count is not None else majority)
    resume = bool(cfg.get("experiment", {}).get("resume", resume))

    strategy = normalize_strategy(strategy or cfg["decoding"]["strategy"])
    spec = decoder_spec(cfg, decoder_key)

    out_dir = resolve_path(cfg["data"]["synthetic_dir"]) / "generated" / decoder_key / strategy
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"generated_{decoder_key}_{emotion.lower()}_reviews_{strategy}.csv"

    run = ExperimentRun(
        out_dir / f"_runs_{emotion}",
        name=f"generate:{decoder_key}:{strategy}:{emotion}",
        cfg=cfg,
        resume=resume,
        extra_meta={
            "emotion": emotion,
            "decoder": decoder_key,
            "strategy": strategy,
            "target": target,
            "human_pos": human_pos,
            "output": str(out_path),
        },
    )

    client = build_client(spec)
    collected: list[dict[str, Any]] = []
    if resume and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            collected = list(reader)
        run.log_event("resume_loaded", n_existing=len(collected), path=str(out_path))

    need = max(0, target - human_pos - len(collected))
    print(
        f"{emotion}: human={human_pos} synthetic={len(collected)} "
        f"target={target} need={need} via {spec['backend']}:{spec['model_id']}"
    )
    run.log_event(
        "generate_status",
        emotion=emotion,
        human_pos=human_pos,
        synthetic=len(collected),
        target=target,
        need=need,
    )

    if need == 0:
        run.finalize(status="completed", reason="already_at_target")
        return out_path

    consecutive_errors = 0
    max_consecutive_errors = 8
    backoff = 1.0
    while need > 0:
        n = min(batch_size, need)
        messages = generation_messages(
            emotion, n, strategy, guidelines_path=cfg["data"].get("guidelines")
        )
        try:
            result = client.chat(
                messages, temperature=float(cfg.get("augmentation", {}).get("temperature", 0.8))
            )
            run.log_llm_call(
                purpose="generate",
                result=result,
                unit_id=f"{emotion}:batch:{len(collected)}",
                extra={"requested": n},
            )
            payload = parse_json_payload(result.text)
            reviews = payload.get("reviews", payload if isinstance(payload, list) else [])
            consecutive_errors = 0
            backoff = 1.0
        except Exception as exc:  # noqa: BLE001
            run.log_event("generate_batch_error", emotion=emotion, error=str(exc))
            msg = str(exc).lower()
            # Do not spin forever on auth / permission failures
            if any(x in msg for x in ("401", "403", "unauthorized", "invalid api key", "authentication")):
                run.finalize(status="failed", error=str(exc))
                raise RuntimeError(
                    f"Fatal auth/permission error while generating {emotion}: {exc}"
                ) from exc
            # Any other error (bad model id, persistent 4xx/5xx, malformed LLM
            # output, ...) used to retry immediately forever with no backoff
            # and no cap -- a single persistent failure (e.g. a wrong model
            # id) would spin here indefinitely, hammering the API. Back off
            # and give up after enough *consecutive* failures; a transient
            # blip that later succeeds resets the counter.
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                run.finalize(status="failed", error=str(exc), consecutive_errors=consecutive_errors)
                raise RuntimeError(
                    f"Giving up generating {emotion} after {consecutive_errors} consecutive "
                    f"non-auth failures (already retried with backoff): {exc}"
                ) from exc
            print(
                f"  batch error ({consecutive_errors}/{max_consecutive_errors}): {exc}; "
                f"retrying in {backoff:.0f}s..."
            )
            time.sleep(backoff)
            backoff = min(backoff * 2, 60.0)
            continue

        added = 0
        for item in reviews:
            if not isinstance(item, dict):
                continue
            sentence = str(item.get("sentence") or item.get("review") or "").strip()
            review = str(item.get("review") or sentence).strip()
            if not sentence:
                continue
            collected.append(_empty_row(sentence, review, emotion, emotions_all))
            added += 1
            need = max(0, target - human_pos - len(collected))
            if need == 0:
                break

        _write_csv(out_path, collected, emotions_all)
        run.log_event(
            "generate_flushed",
            emotion=emotion,
            added=added,
            synthetic=len(collected),
            need=need,
            path=str(out_path),
        )
        print(f"  stored {len(collected)} synthetic ({need} remaining)")

    run.finalize(status="completed", synthetic=len(collected), target=target)
    return out_path


def _write_csv(path: Path, rows: list[dict[str, Any]], emotions: list[str]) -> None:
    fieldnames = [
        "iteration",
        "app_name",
        "categoryId",
        "reviewId",
        "sentenceId",
        "at",
        "score",
        "feature",
        "review",
        "sentence",
        "generated_at",
        *emotions,
        "Reject",
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";", extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    tmp.replace(path)
