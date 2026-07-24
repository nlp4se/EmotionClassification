"""Pull / unload Ollama models to keep disk and VRAM under control."""

from __future__ import annotations

import os
import subprocess
from typing import Any


def _host() -> str:
    return (os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")


def _run(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=check,
        text=True,
        capture_output=True,
    )


def ollama_list_names() -> set[str]:
    """Return installed model names (e.g. ``gemma3:4b``, ``mistral:latest``)."""
    try:
        proc = _run(["ollama", "list"], check=False)
    except FileNotFoundError as exc:
        raise RuntimeError("ollama CLI not found on PATH") from exc
    if proc.returncode != 0:
        return set()
    names: set[str] = set()
    for i, line in enumerate(proc.stdout.splitlines()):
        if i == 0 and line.lower().startswith("name"):
            continue
        parts = line.split()
        if parts:
            names.add(parts[0])
    return names


def ollama_has_model(model_id: str) -> bool:
    installed = ollama_list_names()
    if model_id in installed:
        return True
    # ``mistral`` may appear as ``mistral:latest``
    if ":" not in model_id and f"{model_id}:latest" in installed:
        return True
    return False


def ensure_ollama_model(model_id: str, *, pull_if_missing: bool = True) -> None:
    """Ensure ``model_id`` is installed locally; optionally ``ollama pull``."""
    if ollama_has_model(model_id):
        print(f"[ollama] already present: {model_id}")
        return
    if not pull_if_missing:
        raise RuntimeError(
            f"Ollama model '{model_id}' is not installed and pull_if_missing=False"
        )
    print(f"[ollama] pulling {model_id} ...")
    proc = _run(["ollama", "pull", model_id], check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ollama pull {model_id} failed:\n{proc.stdout}\n{proc.stderr}"
        )
    print(f"[ollama] pull done: {model_id}")


def remove_ollama_model(model_id: str, *, stop_first: bool = True) -> None:
    """Unload from VRAM (best-effort) and delete the on-disk model."""
    if stop_first:
        # Free GPU memory; ignore errors if the model was not loaded
        _run(["ollama", "stop", model_id], check=False)
    if not ollama_has_model(model_id):
        print(f"[ollama] nothing to remove: {model_id}")
        return
    print(f"[ollama] removing {model_id} ...")
    proc = _run(["ollama", "rm", model_id], check=False)
    if proc.returncode != 0:
        # Try :latest alias
        alt = f"{model_id}:latest" if ":" not in model_id else model_id
        proc2 = _run(["ollama", "rm", alt], check=False)
        if proc2.returncode != 0:
            raise RuntimeError(
                f"ollama rm {model_id} failed:\n{proc.stdout}\n{proc.stderr}\n"
                f"{proc2.stdout}\n{proc2.stderr}"
            )
    print(f"[ollama] removed: {model_id}")


def maybe_ensure_from_spec(spec: dict[str, Any], cfg: dict[str, Any]) -> None:
    if spec.get("backend") != "ollama":
        return
    exp = cfg.get("experiment", {})
    if not bool(exp.get("ollama_pull_if_missing", True)):
        return
    ensure_ollama_model(spec["model_id"], pull_if_missing=True)


def maybe_remove_from_spec(spec: dict[str, Any], cfg: dict[str, Any], *, force: bool = False) -> None:
    if spec.get("backend") != "ollama":
        return
    exp = cfg.get("experiment", {})
    if force or bool(exp.get("ollama_remove_after_use", False)):
        remove_ollama_model(spec["model_id"])
