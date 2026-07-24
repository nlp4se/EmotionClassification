"""Experiment run tracking: timestamps, resume markers, JSONL event logs, token usage."""

from __future__ import annotations

import json
import platform
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from emotion_cls.config import REPO_ROOT


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TokenUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
        if self.extra:
            d["extra"] = self.extra
        return d

    def add(self, other: "TokenUsage") -> "TokenUsage":
        def _sum(a: int | None, b: int | None) -> int | None:
            if a is None and b is None:
                return None
            return int(a or 0) + int(b or 0)

        return TokenUsage(
            prompt_tokens=_sum(self.prompt_tokens, other.prompt_tokens),
            completion_tokens=_sum(self.completion_tokens, other.completion_tokens),
            total_tokens=_sum(self.total_tokens, other.total_tokens),
        )


@dataclass
class ChatResult:
    text: str
    usage: TokenUsage
    latency_ms: float
    model_id: str
    backend: str
    raw_response: dict[str, Any] | None = None


def _sanitize_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Drop private / non-serialisable keys and avoid leaking env secrets."""
    out = {}
    for k, v in cfg.items():
        if k.startswith("_"):
            continue
        out[k] = deepcopy(v)
    return out


class ExperimentRun:
    """Stateful run directory with resumable progress and append-only event log.

    Layout under ``out_dir``::

        run_meta.json       # config snapshot, started_at, run_id
        events.jsonl        # append-only detailed events
        progress.json       # completed units + token totals (rewritten often)
        usage_totals.json   # rolling token / latency aggregates
    """

    def __init__(
        self,
        out_dir: Path,
        *,
        name: str,
        cfg: dict[str, Any],
        resume: bool = True,
        extra_meta: dict[str, Any] | None = None,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.resume = resume
        self.events_path = self.out_dir / "events.jsonl"
        self.progress_path = self.out_dir / "progress.json"
        self.meta_path = self.out_dir / "run_meta.json"
        self.usage_path = self.out_dir / "usage_totals.json"

        self.progress: dict[str, Any] = {
            "completed_folds": [],
            "completed_units": [],
            "status": "running",
            "updated_at": utc_now(),
        }
        self.usage_totals = TokenUsage()
        self.total_latency_ms = 0.0
        self.n_llm_calls = 0

        if resume and self.progress_path.exists():
            try:
                self.progress = json.loads(self.progress_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        if resume and self.usage_path.exists():
            try:
                raw = json.loads(self.usage_path.read_text(encoding="utf-8"))
                self.usage_totals = TokenUsage(
                    prompt_tokens=raw.get("prompt_tokens"),
                    completion_tokens=raw.get("completion_tokens"),
                    total_tokens=raw.get("total_tokens"),
                )
                self.total_latency_ms = float(raw.get("total_latency_ms", 0.0))
                self.n_llm_calls = int(raw.get("n_llm_calls", 0))
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        if self.meta_path.exists() and resume:
            try:
                meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
                self.run_id = meta.get("run_id", str(uuid.uuid4()))
            except json.JSONDecodeError:
                self.run_id = str(uuid.uuid4())
            self.log_event("run_resumed", name=name)
        else:
            self.run_id = str(uuid.uuid4())
            meta = {
                "run_id": self.run_id,
                "name": name,
                "started_at": utc_now(),
                "repo_root": str(REPO_ROOT),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "config": _sanitize_cfg(cfg),
                "extra": extra_meta or {},
            }
            self._atomic_write_json(self.meta_path, meta)
            self.log_event("run_started", name=name)

    # ------------------------------------------------------------------ resume
    def fold_done(self, fold: int) -> bool:
        if not self.resume:
            return False
        marker = self.out_dir / f"fold_{fold}" / "metrics.json"
        if marker.exists():
            return True
        return fold in set(self.progress.get("completed_folds", []))

    def load_fold_metrics(self, fold: int) -> dict[str, Any] | None:
        path = self.out_dir / f"fold_{fold}" / "metrics.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def unit_done(self, unit_id: str) -> bool:
        if not self.resume:
            return False
        return unit_id in set(self.progress.get("completed_units", []))

    def mark_unit_done(self, unit_id: str, **payload: Any) -> None:
        units = list(self.progress.get("completed_units", []))
        if unit_id not in units:
            units.append(unit_id)
        self.progress["completed_units"] = units
        self.progress["updated_at"] = utc_now()
        self._flush_progress()
        self.log_event("unit_done", unit_id=unit_id, **payload)

    def mark_fold_done(self, fold: int, metrics: dict[str, Any], *, elapsed_s: float | None = None) -> None:
        fold_dir = self.out_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        payload = dict(metrics)
        payload["_meta"] = {
            "fold": fold,
            "finished_at": utc_now(),
            "elapsed_s": elapsed_s,
        }
        self._atomic_write_json(fold_dir / "metrics.json", payload)
        folds = list(self.progress.get("completed_folds", []))
        if fold not in folds:
            folds.append(fold)
        self.progress["completed_folds"] = sorted(folds)
        self.progress["updated_at"] = utc_now()
        self._flush_progress()
        self.log_event("fold_done", fold=fold, elapsed_s=elapsed_s, metrics_keys=list(metrics.keys()))

    # ------------------------------------------------------------------ logging
    def log_event(self, event_type: str, **payload: Any) -> None:
        record = {
            "ts": utc_now(),
            "run_id": self.run_id,
            "event": event_type,
            **payload,
        }
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def log_llm_call(
        self,
        *,
        purpose: str,
        result: ChatResult,
        fold: int | None = None,
        unit_id: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.usage_totals = self.usage_totals.add(result.usage)
        self.total_latency_ms += float(result.latency_ms)
        self.n_llm_calls += 1
        self._flush_usage()
        self.log_event(
            "llm_call",
            purpose=purpose,
            fold=fold,
            unit_id=unit_id,
            backend=result.backend,
            model_id=result.model_id,
            latency_ms=result.latency_ms,
            usage=result.usage.as_dict(),
            **(extra or {}),
        )

    def finalize(self, status: str = "completed", **payload: Any) -> None:
        self.progress["status"] = status
        self.progress["finished_at"] = utc_now()
        self.progress["updated_at"] = utc_now()
        self._flush_progress()
        self._flush_usage()
        meta = {}
        if self.meta_path.exists():
            try:
                meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                meta = {}
        meta["finished_at"] = utc_now()
        meta["status"] = status
        meta["usage_totals"] = {
            **self.usage_totals.as_dict(),
            "total_latency_ms": self.total_latency_ms,
            "n_llm_calls": self.n_llm_calls,
        }
        meta.update(payload)
        self._atomic_write_json(self.meta_path, meta)
        self.log_event("run_finished", status=status, **payload)

    def _flush_progress(self) -> None:
        self._atomic_write_json(self.progress_path, self.progress)

    def _flush_usage(self) -> None:
        self._atomic_write_json(
            self.usage_path,
            {
                **self.usage_totals.as_dict(),
                "total_latency_ms": self.total_latency_ms,
                "n_llm_calls": self.n_llm_calls,
                "updated_at": utc_now(),
            },
        )

    @staticmethod
    def _atomic_write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        tmp.replace(path)


def timed() -> float:
    return time.perf_counter()
