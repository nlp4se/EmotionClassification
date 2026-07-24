"""LLM backends: Ollama (local) + proprietary HTTP APIs."""

from __future__ import annotations

import json
import os
from typing import Any

import requests

from emotion_cls.config import load_env, require_api_key


class LLMClient:
    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> str:
        raise NotImplementedError


class OllamaClient(LLMClient):
    def __init__(self, model_id: str, base_url: str | None = None):
        load_env()
        self.model_id = model_id
        self.base_url = (base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> str:
        payload = {
            "model": self.model_id,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=600)
        r.raise_for_status()
        return r.json()["message"]["content"]


class OpenAIClient(LLMClient):
    def __init__(self, model_id: str, api_key_env: str = "OPENAI_API_KEY"):
        self.model_id = model_id
        self.api_key = require_api_key(api_key_env)
        self.url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model_id, "messages": messages, "temperature": temperature}
        r = requests.post(self.url, headers=headers, json=payload, timeout=600)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


class AnthropicClient(LLMClient):
    def __init__(self, model_id: str, api_key_env: str = "ANTHROPIC_API_KEY"):
        self.model_id = model_id
        # Accept legacy CLAUDE_API_KEY as alias
        load_env()
        key = os.environ.get(api_key_env) or os.environ.get("CLAUDE_API_KEY")
        if not key:
            raise RuntimeError(f"Missing API key: set {api_key_env} or CLAUDE_API_KEY in .env")
        self.api_key = key
        self.url = "https://api.anthropic.com/v1/messages"

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> str:
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
        r = requests.post(self.url, headers=headers, json=payload, timeout=600)
        r.raise_for_status()
        return r.json()["content"][0]["text"]


class GeminiClient(LLMClient):
    def __init__(self, model_id: str, api_key_env: str = "GEMINI_API_KEY"):
        self.model_id = model_id
        self.api_key = require_api_key(api_key_env)

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> str:
        # Flatten to a single user prompt for simplicity / compatibility
        parts = []
        for m in messages:
            parts.append(f"{m['role'].upper()}: {m['content']}")
        prompt = "\n\n".join(parts)
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_id}:generateContent"
        )
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
        return data["candidates"][0]["content"]["parts"][0]["text"]


class MistralClient(LLMClient):
    def __init__(self, model_id: str, api_key_env: str = "MISTRAL_API_KEY"):
        self.model_id = model_id
        self.api_key = require_api_key(api_key_env)
        self.url = "https://api.mistral.ai/v1/chat/completions"

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model_id, "messages": messages, "temperature": temperature}
        r = requests.post(self.url, headers=headers, json=payload, timeout=600)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


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
