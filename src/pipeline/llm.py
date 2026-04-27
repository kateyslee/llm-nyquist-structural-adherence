"""LLM backends: Gemini (default) and OpenAI-compatible (Qwen via Ollama/vLLM)."""

from __future__ import annotations

import random
import re
import time
import warnings
from dataclasses import dataclass
from typing import Protocol

from google.api_core import exceptions as google_api_exceptions
from openai import OpenAI

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    import google.generativeai as genai

from ..config import gemini_api_key, gemini_model, ollama_openai_defaults, openai_compatible_config


def strip_markdown_fence(text: str) -> str:
    t = text.strip()
    if not t.startswith("```"):
        return t
    lines = t.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _retry_after_seconds(exc: BaseException) -> float | None:
    m = re.search(r"retry in ([\d.]+)\s*s", str(exc), re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


class LLMClient(Protocol):
    def generate(self, system: str, user: str) -> str: ...


@dataclass
class GeminiClient:
    model_name: str | None = None
    max_quota_retries: int = 6

    def __post_init__(self) -> None:
        key = gemini_api_key()
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY for Gemini.")
        genai.configure(api_key=key)
        self.model_name = self.model_name or gemini_model()

    def generate(self, system: str, user: str) -> str:
        try:
            model = genai.GenerativeModel(self.model_name, system_instruction=system)
        except TypeError:
            user = f"{system}\n\n---\n\n{user}"
            model = genai.GenerativeModel(self.model_name)

        last_exc: BaseException | None = None
        for attempt in range(self.max_quota_retries):
            try:
                resp = model.generate_content(user)
                if not resp.candidates:
                    fb = getattr(resp, "prompt_feedback", None)
                    block = getattr(fb, "block_reason", None) if fb is not None else None
                    raise ValueError(
                        "Model returned no candidates (empty or blocked). "
                        f"prompt_feedback={fb!r} block_reason={block!r}"
                    )
                parts = getattr(resp.candidates[0].content, "parts", None) or []
                texts = [getattr(p, "text", "") for p in parts]
                return strip_markdown_fence("".join(texts))
            except google_api_exceptions.ResourceExhausted as e:
                last_exc = e
                if attempt >= self.max_quota_retries - 1:
                    raise
                wait = _retry_after_seconds(e) or min(90.0, (2**attempt) + random.uniform(0, 1.5))
                time.sleep(wait)
            except google_api_exceptions.GoogleAPIError:
                raise

        assert last_exc is not None
        raise last_exc


@dataclass
class OpenAICompatClient:
    base_url: str
    api_key: str
    model: str

    def __post_init__(self) -> None:
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def generate(self, system: str, user: str) -> str:
        r = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.7,
        )
        content = (r.choices[0].message.content or "").strip()
        return strip_markdown_fence(content)


def build_default_client(prefer: str = "gemini") -> LLMClient:
    prefer = prefer.lower().replace("-", "_")
    if prefer == "gemini":
        if not gemini_api_key():
            raise RuntimeError(
                "Gemini backend requires GEMINI_API_KEY or GOOGLE_API_KEY in .env (or use --backend ollama_qwen)."
            )
        return GeminiClient()
    if prefer == "ollama_qwen":
        base, key, model = ollama_openai_defaults()
        return OpenAICompatClient(base_url=base, api_key=key, model=model)
    if prefer == "openai_compat":
        base, key, model = openai_compatible_config()
        if not (base and key and model):
            raise RuntimeError(
                "Set OPENAI_API_BASE, OPENAI_API_KEY, OPENAI_MODEL for OpenAI-compatible backend."
            )
        return OpenAICompatClient(base_url=base, api_key=key, model=model)
    raise RuntimeError(
        f"Unknown backend {prefer!r}. Use: gemini, ollama_qwen, openai_compat "
        "(hyphens ok: ollama-qwen)."
    )
