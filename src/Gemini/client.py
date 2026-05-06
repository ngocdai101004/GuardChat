"""Gemini 2.5 Flash API client wrapper for GuardChat Task 2.

Thin wrapper around the official ``google-genai`` SDK. Handles:

    * lazy import (so users who only need the local Llama / SafeGuider
      rewriters don't need to install ``google-genai``),
    * API-key resolution from explicit argument / ``GEMINI_API_KEY`` /
      ``GOOGLE_API_KEY`` env vars,
    * relaxed safety thresholds (``BLOCK_NONE`` for the four
      configurable categories) so the model is allowed to read
      adversarial GuardChat prompts and produce a sanitised rewrite,
      rather than refusing the request outright,
    * retry-with-exponential-backoff for transient API failures
      (HTTP 429 / 500 / 503).

Why ``BLOCK_NONE``?
-------------------
GuardChat test prompts are **adversarial by construction**. With the
default safety thresholds, Gemini refuses many of them and returns an
empty / blocked response, which makes the rewriter useless. Setting
``BLOCK_NONE`` lets the model see the input and produce a sanitised
rewrite per the system instruction. The *output* is the rewrite itself,
which the system instruction explicitly constrains to be safe.

Library versions
----------------
* ``google-genai >= 0.3``  (the new Gemini SDK, ``from google import genai``).
* ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY`` env var with a valid key
  obtained from https://aistudio.google.com/.

If you have only the older ``google-generativeai`` package installed,
upgrade with::

    pip install -U "google-genai>=0.3"

The two SDKs cannot coexist cleanly on the same import path; we use
the new one because it ships first-class Gemini 2.5 support.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_MODEL_NAME = "gemini-2.5-flash"


# Categories that the Gemini Developer API allows callers to override.
# Note: ``HARM_CATEGORY_CIVIC_INTEGRITY`` and
# ``HARM_CATEGORY_DANGEROUS_CONTENT`` exist on Vertex AI; the public
# Developer API exposes the four below. We list the broad set and let
# the SDK silently ignore unknown keys.
_OVERRIDABLE_HARM_CATEGORIES: List[str] = [
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
]


def _import_genai():
    """Lazily import the new Gemini SDK with a friendly error."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise RuntimeError(
            "Gemini support needs the `google-genai` package "
            "(`pip install \"google-genai>=0.3\"`). The older "
            "`google-generativeai` is not used here."
        ) from e
    return genai, types


def _resolve_api_key(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        v = os.environ.get(var)
        if v:
            return v
    raise RuntimeError(
        "No Gemini API key found. Pass --api-key or set GEMINI_API_KEY / "
        "GOOGLE_API_KEY in the environment. Get a key at "
        "https://aistudio.google.com/."
    )


@dataclass
class GenerationConfig:
    """Sampling settings for a single rewrite call.

    Greedy decoding (``temperature=0``) is the default - we want a
    deterministic, structured rewrite, not creative variance.
    """

    max_output_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0


@dataclass
class GeminiClientConfig:
    model_name: str = DEFAULT_MODEL_NAME
    api_key: Optional[str] = None
    relax_safety: bool = True
    retries: int = 3
    backoff_seconds: float = 2.0
    request_timeout: Optional[float] = 60.0
    generation: GenerationConfig = field(default_factory=GenerationConfig)


@dataclass
class GeminiResponse:
    """Decoded result of a single ``generate_content`` call."""

    text: str
    blocked: bool                       # True if Gemini's safety filter killed it.
    block_reason: Optional[str] = None  # e.g. "PROHIBITED_CONTENT", "SAFETY"
    finish_reason: Optional[str] = None
    raw: Any = None                     # original SDK response object (debug only)


class GeminiClient:
    """Tiny wrapper that exposes a single :meth:`generate` call."""

    def __init__(self, config: GeminiClientConfig = GeminiClientConfig()) -> None:
        self.config = config
        genai, types = _import_genai()
        self._genai = genai
        self._types = types

        api_key = _resolve_api_key(config.api_key)
        self._client = genai.Client(api_key=api_key)

        # Safety overrides: BLOCK_NONE lets the model see adversarial
        # GuardChat inputs and emit a sanitised rewrite per the system
        # instruction. We build the structured ``SafetySetting`` list
        # eagerly so it can be reused across all generate calls.
        self._safety_settings = (
            self._build_safety_settings() if config.relax_safety else None
        )

    # ----------------------- Setup helpers -------------------------- #

    def _build_safety_settings(self):
        types = self._types
        out = []
        for cat in _OVERRIDABLE_HARM_CATEGORIES:
            try:
                out.append(types.SafetySetting(
                    category=cat,
                    threshold="BLOCK_NONE",
                ))
            except (TypeError, ValueError):
                # SDK rejected the category - silently skip and keep
                # going, matching the SDK's "ignore unknown" semantics.
                continue
        return out

    def _build_generate_config(self, system_instruction: str):
        types = self._types
        gen = self.config.generation
        kwargs: Dict[str, Any] = {
            "system_instruction": system_instruction,
            "temperature": gen.temperature,
            "top_p": gen.top_p,
            "max_output_tokens": gen.max_output_tokens,
        }
        if self._safety_settings:
            kwargs["safety_settings"] = self._safety_settings
        return types.GenerateContentConfig(**kwargs)

    # ---------------------------- Generate -------------------------- #

    def generate(
        self,
        user_prompt: str,
        system_instruction: str,
    ) -> GeminiResponse:
        """Run one ``generate_content`` call with retry-on-transient-error.

        Retries trigger on HTTP 429 / 5xx / network exceptions; safety
        blocks are returned as a populated :class:`GeminiResponse` and
        not retried (re-running with the same input would not change
        the verdict).
        """
        cfg = self._build_generate_config(system_instruction)
        last_exc: Optional[Exception] = None
        for attempt in range(max(1, self.config.retries)):
            try:
                response = self._client.models.generate_content(
                    model=self.config.model_name,
                    contents=user_prompt,
                    config=cfg,
                )
                return self._decode(response)
            except Exception as e:  # noqa: BLE001 - SDK raises a wide tree
                last_exc = e
                msg = str(e).lower()
                if attempt + 1 >= self.config.retries:
                    break
                if self._is_transient(msg):
                    sleep_s = self.config.backoff_seconds * (2 ** attempt)
                    time.sleep(sleep_s)
                    continue
                # Non-transient: bail immediately.
                break

        if last_exc is not None:
            raise last_exc
        # Should be unreachable, but keep mypy happy:
        raise RuntimeError("Gemini generate_content returned no result.")

    # ----------------------- Response decoding ---------------------- #

    @staticmethod
    def _is_transient(msg: str) -> bool:
        """Return True for retry-eligible error messages."""
        markers = (
            "429", "rate limit", "rate-limit",
            "500", "502", "503", "504",
            "deadline", "timeout", "temporarily unavailable",
        )
        return any(m in msg for m in markers)

    def _decode(self, response) -> GeminiResponse:
        """Best-effort extraction of text + safety verdict from any
        google-genai response shape we have observed.
        """
        text = ""
        blocked = False
        block_reason: Optional[str] = None
        finish_reason: Optional[str] = None

        # The SDK exposes ``.text`` on simple successes.
        try:
            text = response.text or ""
        except Exception:
            text = ""

        candidates = getattr(response, "candidates", None) or []
        if candidates:
            cand = candidates[0]
            finish_reason = self._stringify_enum(getattr(cand, "finish_reason", None))
            # Gather text from candidate content parts as a fallback.
            if not text:
                content = getattr(cand, "content", None)
                parts = getattr(content, "parts", None) or []
                fragments = []
                for p in parts:
                    t = getattr(p, "text", None)
                    if t:
                        fragments.append(t)
                text = "".join(fragments)

            # finish_reason in {"SAFETY","PROHIBITED_CONTENT","SPII",...}
            if finish_reason and finish_reason.upper() in {
                "SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII",
            }:
                blocked = True
                block_reason = finish_reason

        # Prompt-level blocks land on ``response.prompt_feedback``.
        pf = getattr(response, "prompt_feedback", None)
        if pf is not None:
            br = getattr(pf, "block_reason", None)
            if br:
                blocked = True
                block_reason = block_reason or self._stringify_enum(br)

        return GeminiResponse(
            text=text or "",
            blocked=blocked,
            block_reason=block_reason,
            finish_reason=finish_reason,
            raw=response,
        )

    @staticmethod
    def _stringify_enum(v) -> Optional[str]:
        if v is None:
            return None
        # SDK uses google.genai.types enums; .name is the canonical form.
        return getattr(v, "name", None) or str(v)


__all__ = [
    "DEFAULT_MODEL_NAME",
    "GenerationConfig",
    "GeminiClientConfig",
    "GeminiClient",
    "GeminiResponse",
]
