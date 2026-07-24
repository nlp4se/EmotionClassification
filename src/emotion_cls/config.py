"""Paths, YAML config loading, and API-key helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_env(dotenv_path: Path | None = None) -> None:
    """Load `.env` from the repo root (or an explicit path).

    Uses ``override=True`` so a corrected key in `.env` wins over a stale
    value already in the process environment (common with long-running
    nohup jobs that called ``load_env`` before `.env` was fixed).
    """
    path = dotenv_path or (REPO_ROOT / ".env")
    load_dotenv(path, override=True)
    _ensure_writable_hf_cache()


def _ensure_writable_hf_cache() -> None:
    """Prefer a writable Hugging Face cache (shared /data caches are often read-only)."""
    repo_cache = REPO_ROOT / ".cache" / "huggingface"
    env_vals = [
        os.environ.get(k, "")
        for k in (
            "HUGGINGFACE_HUB_CACHE",
            "HF_HUB_CACHE",
            "HF_HOME",
            "TRANSFORMERS_CACHE",
            "SENTENCE_TRANSFORMERS_HOME",
        )
    ]
    shared_prefix = "/data/caches/huggingface"
    use_repo = any(v.startswith(shared_prefix) for v in env_vals if v)
    if not use_repo:
        shared = Path(shared_prefix) / "hub"
        if shared.exists():
            # Hub dir may be listable but not creatable for this user
            try:
                probe = shared / f".write_probe_{os.getuid()}"
                probe.write_text("ok")
                probe.unlink(missing_ok=True)
            except OSError:
                use_repo = True
    if not use_repo:
        return

    hub = repo_cache / "hub"
    transformers = repo_cache / "transformers"
    st = repo_cache / "sentence_transformers"
    for d in (hub, transformers, st):
        d.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(repo_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub)
    os.environ["HF_HUB_CACHE"] = str(hub)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers)
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(st)


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(resolve_path(path), encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_default_config(
    config_path: str | Path = "configs/default.yaml",
    models_path: str | Path = "configs/models.yaml",
) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    cfg["_models"] = load_yaml(models_path)
    cfg["_repo_root"] = str(REPO_ROOT)
    return cfg


def require_api_key(env_var: str, *, allow_empty: bool = False) -> str:
    """Return an API key from the environment after loading `.env`."""
    load_env()
    key = os.environ.get(env_var, "").strip()
    if key or allow_empty:
        return key
    raise RuntimeError(
        f"Missing API key: set {env_var} in the environment or in {REPO_ROOT / '.env'}. "
        f"See .env.example."
    )


def encoder_hub_id(cfg: dict[str, Any], encoder_key: str) -> str:
    encoders = cfg.get("_models", {}).get("encoders", {})
    if encoder_key not in encoders:
        # Allow raw Hugging Face ids
        return encoder_key
    return encoders[encoder_key]["hub_id"]


def decoder_spec(cfg: dict[str, Any], decoder_key: str) -> dict[str, Any]:
    decoders = cfg.get("_models", {}).get("decoders", {})
    pilots = cfg.get("_models", {}).get("augmentation_generators", {})
    if decoder_key in decoders:
        return dict(decoders[decoder_key])
    if decoder_key in pilots:
        return dict(pilots[decoder_key])
    raise KeyError(
        f"Unknown decoder '{decoder_key}'. "
        f"Known: {sorted(list(decoders) + list(pilots))}"
    )
