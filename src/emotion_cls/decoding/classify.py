"""Decoder-only zero/few-shot emotion classification (RQ2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from emotion_cls.config import decoder_spec, resolve_path
from emotion_cls.data.dataset import label_matrix, load_ground_truth, multilabel_folds
from emotion_cls.decoding.clients import build_client, parse_json_payload
from emotion_cls.decoding.prompts import classification_messages, normalize_strategy
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


def run_decoder_classification(cfg: dict[str, Any], decoder_key: str, *, dry_run: bool = False) -> Path:
    emotions = cfg["data"]["emotions"]
    df = load_ground_truth(cfg["data"]["ground_truth"], emotions=emotions)
    y = label_matrix(df, emotions)
    strategy = normalize_strategy(cfg["decoding"]["strategy"])
    max_labels = int(cfg["evaluation"]["max_labels"])
    n_folds = int(cfg["evaluation"]["n_folds"])
    seed = int(cfg["project"]["seed"])
    few_k = int(cfg["decoding"].get("few_shot_k", 5))
    temperature = float(cfg["decoding"].get("temperature", 0.0))

    spec = decoder_spec(cfg, decoder_key)
    out_dir = resolve_path(cfg["project"]["output_dir"]) / "decoder_classify" / decoder_key / strategy
    out_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"[dry-run] decoder classify model={spec['model_id']} backend={spec['backend']} strategy={strategy}")
        return out_dir

    client = build_client(spec)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(multilabel_folds(y, n_folds, seed), start=1):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx].reset_index(drop=True)
        few = None
        if strategy == "few_shot_guidelines_dataset":
            few = {}
            for e in emotions:
                if e == "Neutral":
                    continue
                ex = train_df.loc[train_df[e] == 1, "sentence"].tolist()[:few_k]
                few[e] = ex

        preds = np.zeros((len(val_df), len(emotions)), dtype=int)
        rows = []
        for i, sentence in enumerate(val_df["sentence"].tolist()):
            messages = classification_messages(sentence, emotions, strategy, few_shot_examples=few)
            text = client.chat(messages, temperature=temperature)
            try:
                payload = parse_json_payload(text)
                labels = _normalize_labels(payload, emotions, max_labels)
            except Exception as exc:  # noqa: BLE001
                labels = []
                rows.append({"sentence": sentence, "pred": [], "error": str(exc), "raw": text[:500]})
            else:
                rows.append({"sentence": sentence, "pred": labels, "error": "", "raw": ""})
            for lab in labels:
                preds[i, emotions.index(lab)] = 1

        metrics = compute_metrics(label_matrix(val_df, emotions), preds, emotions)
        fold_metrics.append(metrics)
        pd.DataFrame(rows).to_csv(out_dir / f"fold_{fold}_predictions.csv", index=False)
        with open(out_dir / f"fold_{fold}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    with open(out_dir / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(fold_metrics, f, indent=2)
    return out_dir
