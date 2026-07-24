"""Paths, YAML config loading, and API-key helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_env(dotenv_path: Path | None = None) -> None:
    """Load `.env` from the repo root (or an explicit path)."""
    path = dotenv_path or (REPO_ROOT / ".env")
    load_dotenv(path, override=False)


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
    pilots = cfg.get("_models", {}).get("augmentation_pilot", {})
    if decoder_key in decoders:
        return dict(decoders[decoder_key])
    if decoder_key in pilots:
        return dict(pilots[decoder_key])
    raise KeyError(
        f"Unknown decoder '{decoder_key}'. "
        f"Known: {sorted(list(decoders) + list(pilots))}"
    )
