"""Minimal client for any OpenAI-compatible chat-completions API (OpenAI, vLLM,
Ollama, OpenRouter, Together, Anthropic's OpenAI-compatible endpoint, ...).
Standard library only, so the agent has no SDK dependency."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Completion:
    text: str
    model: str
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LLM(Protocol):
    model: str

    def complete(self, messages: list[dict]) -> Completion: ...


@dataclass
class OpenAICompatLLM:
    model: str
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    temperature: float = 0.7
    max_tokens: int = 8192
    timeout_s: float = 300.0
    retries: int = 3
    extra_body: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls, model: str | None = None, base_url: str | None = None,
                 **kw) -> OpenAICompatLLM:
        model = model or os.environ.get("KP_MODEL")
        if not model:
            raise ValueError("no model configured: pass --model or set KP_MODEL")
        base_url = (base_url or os.environ.get("KP_BASE_URL")
                    or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1")
        api_key = os.environ.get("KP_API_KEY") or os.environ.get("OPENAI_API_KEY")
        return cls(model=model, base_url=base_url.rstrip("/"), api_key=api_key, **kw)

    def complete(self, messages: list[dict]) -> Completion:
        body = {"model": self.model, "messages": messages, "temperature": self.temperature,
                "max_tokens": self.max_tokens, **self.extra_body}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions",
                                     data=json.dumps(body).encode(), headers=headers,
                                     method="POST")
        last_err: Exception | None = None
        for attempt in range(self.retries):
            t0 = time.monotonic()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    payload = json.loads(resp.read())
                usage = payload.get("usage") or {}
                return Completion(
                    text=payload["choices"][0]["message"].get("content") or "",
                    model=payload.get("model", self.model),
                    latency_s=time.monotonic() - t0,
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                )
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:500]
                last_err = RuntimeError(f"HTTP {e.code} from {self.base_url}: {detail}")
                if e.code not in (408, 409, 429) and e.code < 500:
                    break  # client error: retrying will not help
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
                last_err = e
            time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM request failed: {last_err}")


@dataclass
class ScriptedLLM:
    """Replays canned responses in order. Used by tests and dry runs."""

    responses: Iterable[str]
    model: str = "scripted"

    def __post_init__(self):
        self._it = iter(self.responses)
        self.calls: list[list[dict]] = []

    def complete(self, messages: list[dict]) -> Completion:
        self.calls.append(messages)
        return Completion(text=next(self._it), model=self.model, latency_s=0.0)
