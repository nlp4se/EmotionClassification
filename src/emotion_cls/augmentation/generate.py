"""Generate synthetic emotion reviews until a target count is reached (RQ4 / RQ3)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from emotion_cls.config import decoder_spec, resolve_path
from emotion_cls.data.dataset import load_ground_truth, majority_count
from emotion_cls.decoding.clients import build_client, parse_json_payload
from emotion_cls.decoding.prompts import generation_messages, normalize_strategy


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
    need = max(0, target - human_pos)

    strategy = normalize_strategy(strategy or cfg["decoding"]["strategy"])
    spec = decoder_spec(cfg, decoder_key)

    out_dir = resolve_path(cfg["data"]["synthetic_dir"]) / "generated" / decoder_key / strategy
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"generated_{decoder_key}_{emotion.lower()}_reviews_{strategy}.csv"

    client = build_client(spec)
    collected: list[dict[str, Any]] = []
    # Resume if file exists
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            collected = list(reader)
        need = max(0, target - human_pos - len(collected))

    print(f"{emotion}: human={human_pos} target={target} need={need} via {spec['backend']}:{spec['model_id']}")

    while len([r for r in collected]) < (target - human_pos) and need > 0:
        n = min(batch_size, need)
        messages = generation_messages(emotion, n, strategy, guidelines_path=cfg["data"].get("guidelines"))
        text = client.chat(messages, temperature=float(cfg.get("augmentation", {}).get("temperature", 0.8)))
        payload = parse_json_payload(text)
        reviews = payload.get("reviews", payload if isinstance(payload, list) else [])
        for item in reviews:
            sentence = str(item.get("sentence") or item.get("review") or "").strip()
            review = str(item.get("review") or sentence).strip()
            if not sentence:
                continue
            collected.append(_empty_row(sentence, review, emotion, emotions_all))
            need = max(0, target - human_pos - len(collected))
            if need == 0:
                break

        _write_csv(out_path, collected, emotions_all)
        print(f"  stored {len(collected)} synthetic ({need} remaining)")

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
        *emotions,
        "Reject",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        w.writeheader()
        for row in rows:
            w.writerow(row)
