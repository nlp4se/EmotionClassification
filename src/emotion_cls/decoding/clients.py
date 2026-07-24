"""LLM backends: Ollama (local) + proprietary HTTP APIs.

All clients return :class:`ChatResult` with text, latency, and token usage when
the provider exposes it.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import requests

from emotion_cls.config import load_env, require_api_key
from emotion_cls.experiment import ChatResult, TokenUsage


def _usage_openai_like(data: dict[str, Any]) -> TokenUsage:
    u = data.get("usage") or {}
    prompt = u.get("prompt_tokens")
    completion = u.get("completion_tokens")
    total = u.get("total_tokens")
    if total is None and prompt is not None and completion is not None:
        total = int(prompt) + int(completion)
    return TokenUsage(
        prompt_tokens=int(prompt) if prompt is not None else None,
        completion_tokens=int(completion) if completion is not None else None,
        total_tokens=int(total) if total is not None else None,
        extra={k: v for k, v in u.items() if k not in {"prompt_tokens", "completion_tokens", "total_tokens"}},
    )


class LLMClient:
    backend: str = "base"
    model_id: str = ""

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> ChatResult:
        raise NotImplementedError


class OllamaClient(LLMClient):
    backend = "ollama"

    def __init__(self, model_id: str, base_url: str | None = None):
        load_env()
        self.model_id = model_id
        self.base_url = (base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> ChatResult:
        payload = {
            "model": self.model_id,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        t0 = time.perf_counter()
        r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        prompt = data.get("prompt_eval_count")
        completion = data.get("eval_count")
        total = None
        if prompt is not None or completion is not None:
            total = int(prompt or 0) + int(completion or 0)
        usage = TokenUsage(
            prompt_tokens=int(prompt) if prompt is not None else None,
            completion_tokens=int(completion) if completion is not None else None,
            total_tokens=total,
            extra={
                k: data.get(k)
                for k in ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration")
                if k in data
            },
        )
        return ChatResult(
            text=data["message"]["content"],
            usage=usage,
            latency_ms=latency_ms,
            model_id=self.model_id,
            backend=self.backend,
            raw_response={k: v for k, v in data.items() if k != "message"},
        )


class OpenAIClient(LLMClient):
    backend = "openai"

    def __init__(self, model_id: str, api_key_env: str = "OPENAI_API_KEY"):
        self.model_id = model_id
        self.api_key = require_api_key(api_key_env)
        self.url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> ChatResult:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model_id, "messages": messages, "temperature": temperature}
        t0 = time.perf_counter()
        r = requests.post(self.url, headers=headers, json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return ChatResult(
            text=data["choices"][0]["message"]["content"],
            usage=_usage_openai_like(data),
            latency_ms=latency_ms,
            model_id=self.model_id,
            backend=self.backend,
        )


class AnthropicClient(LLMClient):
    backend = "anthropic"

    def __init__(self, model_id: str, api_key_env: str = "ANTHROPIC_API_KEY"):
        self.model_id = model_id
        load_env()
        key = os.environ.get(api_key_env) or os.environ.get("CLAUDE_API_KEY")
        if not key:
            raise RuntimeError(f"Missing API key: set {api_key_env} or CLAUDE_API_KEY in .env")
        self.api_key = key
        self.url = "https://api.anthropic.com/v1/messages"

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> ChatResult:
        system = ""
        user_msgs = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                user_msgs.append(m)
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": 4096,
            "temperature": temperature,
            "messages": user_msgs,
        }
        if system:
            payload["system"] = system
        t0 = time.perf_counter()
        r = requests.post(self.url, headers=headers, json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        u = data.get("usage") or {}
        prompt = u.get("input_tokens")
        completion = u.get("output_tokens")
        total = None
        if prompt is not None or completion is not None:
            total = int(prompt or 0) + int(completion or 0)
        usage = TokenUsage(
            prompt_tokens=int(prompt) if prompt is not None else None,
            completion_tokens=int(completion) if completion is not None else None,
            total_tokens=total,
            extra={k: v for k, v in u.items() if k not in {"input_tokens", "output_tokens"}},
        )
        return ChatResult(
            text=data["content"][0]["text"],
            usage=usage,
            latency_ms=latency_ms,
            model_id=self.model_id,
            backend=self.backend,
        )


class GeminiClient(LLMClient):
    backend = "gemini"

    def __init__(self, model_id: str, api_key_env: str = "GEMINI_API_KEY"):
        self.model_id = model_id
        self.api_key = require_api_key(api_key_env)

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> ChatResult:
        parts = []
        for m in messages:
            parts.append(f"{m['role'].upper()}: {m['content']}")
        prompt = "\n\n".join(parts)
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_id}:generateContent"
        )
        t0 = time.perf_counter()
        r = requests.post(
            url,
            params={"key": self.api_key},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temperature},
            },
            timeout=600,
        )
        r.raise_for_status()
        data = r.json()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        meta = data.get("usageMetadata") or {}
        prompt_t = meta.get("promptTokenCount")
        completion_t = meta.get("candidatesTokenCount")
        total_t = meta.get("totalTokenCount")
        usage = TokenUsage(
            prompt_tokens=int(prompt_t) if prompt_t is not None else None,
            completion_tokens=int(completion_t) if completion_t is not None else None,
            total_tokens=int(total_t) if total_t is not None else None,
            extra={k: v for k, v in meta.items() if k not in {"promptTokenCount", "candidatesTokenCount", "totalTokenCount"}},
        )
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return ChatResult(
            text=text,
            usage=usage,
            latency_ms=latency_ms,
            model_id=self.model_id,
            backend=self.backend,
        )


class MistralClient(LLMClient):
    backend = "mistral"

    def __init__(self, model_id: str, api_key_env: str = "MISTRAL_API_KEY"):
        self.model_id = model_id
        self.api_key = require_api_key(api_key_env)
        self.url = "https://api.mistral.ai/v1/chat/completions"

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> ChatResult:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model_id, "messages": messages, "temperature": temperature}
        t0 = time.perf_counter()
        r = requests.post(self.url, headers=headers, json=payload, timeout=600)
        r.raise_for_status()
        data = r.json()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return ChatResult(
            text=data["choices"][0]["message"]["content"],
            usage=_usage_openai_like(data),
            latency_ms=latency_ms,
            model_id=self.model_id,
            backend=self.backend,
        )


def build_client(spec: dict[str, Any]) -> LLMClient:
    backend = spec["backend"]
    model_id = spec["model_id"]
    if backend == "ollama":
        return OllamaClient(model_id)
    if backend == "openai":
        return OpenAIClient(model_id, spec.get("api_key_env", "OPENAI_API_KEY"))
    if backend == "anthropic":
        return AnthropicClient(model_id, spec.get("api_key_env", "ANTHROPIC_API_KEY"))
    if backend == "gemini":
        return GeminiClient(model_id, spec.get("api_key_env", "GEMINI_API_KEY"))
    if backend == "mistral":
        return MistralClient(model_id, spec.get("api_key_env", "MISTRAL_API_KEY"))
    raise ValueError(f"Unknown backend: {backend}")


def parse_json_payload(text: str) -> Any:
    """Extract JSON from a model response (raw or fenced)."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise
