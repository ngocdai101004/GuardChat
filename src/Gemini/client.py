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


# The paper names *Gemini 2.5 Flash* as the proprietary Task-2 rewriter,
# but ``gemini-2.5-flash`` (and ``-flash-lite``, and the whole 2.0 line)
# now answers 404 "no longer available to new users" for API keys issued
# after its retirement - the id is still listed by ``models.list()``,
# which is why the failure only shows up at generate time. The default
# is therefore the current Flash tier, pinned to a concrete version
# rather than the drifting ``gemini-flash-latest`` alias so a benchmark
# run stays reproducible. Override with ``--model`` if your key still
# has 2.5 access; whatever is used is recorded in the output's ``meta``.
DEFAULT_MODEL_NAME = "gemini-3.5-flash"


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


API_KEY_ENV_KEYS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")


def _resolve_api_key(explicit: Optional[str]) -> str:
    """Explicit argument, then the environment, then the repo-root ``.env``.

    Same resolution order the gated HuggingFace repos use, so every
    credential in this benchmark lives in one git-ignored file.
    """
    from src.utils.hf_token import resolve_env_value

    key = resolve_env_value(API_KEY_ENV_KEYS, explicit=explicit)
    if key:
        return key
    raise RuntimeError(
        "No Gemini API key found. Pass --api-key, set GEMINI_API_KEY / "
        "GOOGLE_API_KEY in the environment, or add GEMINI_API_KEY=... to "
        "the repo-root .env file. Get a key at https://aistudio.google.com/."
    )


@dataclass
class GenerationConfig:
    """Sampling settings for a single rewrite call.

    Greedy decoding (``temperature=0``) is the default - we want a
    deterministic, structured rewrite, not creative variance.

    ``thinking_budget=0`` disables Gemini 2.5's reasoning tokens. This is
    not an optimisation, it is a correctness fix: reasoning tokens are
    drawn from the same ``max_output_tokens`` pool as the answer, so with
    thinking left on its default the model can spend the whole budget
    deliberating and return an empty response with
    ``finish_reason=MAX_TOKENS``. Sanitising a prompt is a rewriting
    task, not a reasoning one. Set it to ``None`` to leave the model
    default in place.
    """

    max_output_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0
    thinking_budget: Optional[int] = 0


@dataclass
class GeminiClientConfig:
    """Per-run client settings.

    ``retries`` is the total number of attempts per sample, not the
    number of extra ones: ``retries=3`` means three calls at most.

    ``retry_blocked`` also re-sends a request the safety filter killed,
    and a request that came back empty. At ``temperature=0`` a re-send
    is usually deterministic and blocks the same way, so this rarely
    rescues a sample - but Google's filter is not perfectly stable, the
    cost of finding out is three calls on a handful of rows, and a
    sample that survives three blocks is a much stronger claim to put in
    the paper than one that survived a single call.
    """

    model_name: str = DEFAULT_MODEL_NAME
    api_key: Optional[str] = None
    relax_safety: bool = True
    retries: int = 3
    retry_blocked: bool = True
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
    truncated: bool = False             # hit max_output_tokens mid-answer
    attempts: int = 1                   # calls made, including retries
    raw: Any = None                     # original SDK response object (debug only)


class GeminiClient:
    """Tiny wrapper that exposes a single :meth:`generate` call."""

    def __init__(self, config: GeminiClientConfig = GeminiClientConfig()) -> None:
        self.config = config
        genai, types = _import_genai()
        self._genai = genai
        self._types = types

        api_key = _resolve_api_key(config.api_key)
        self._client = genai.Client(
            api_key=api_key,
            **self._client_kwargs(config.request_timeout),
        )

        # Safety overrides: BLOCK_NONE lets the model see adversarial
        # GuardChat inputs and emit a sanitised rewrite per the system
        # instruction. We build the structured ``SafetySetting`` list
        # eagerly so it can be reused across all generate calls.
        self._safety_settings = (
            self._build_safety_settings() if config.relax_safety else None
        )

    # ----------------------- Setup helpers -------------------------- #

    def _client_kwargs(self, request_timeout: Optional[float]) -> Dict[str, Any]:
        """Per-request HTTP timeout, in the shape the SDK wants (ms)."""
        if not request_timeout:
            return {}
        try:
            return {"http_options": self._types.HttpOptions(
                timeout=int(float(request_timeout) * 1000)
            )}
        except (AttributeError, TypeError, ValueError):
            # Older SDKs without HttpOptions - fall back to the default.
            return {}

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
        if gen.thinking_budget is not None:
            try:
                kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=int(gen.thinking_budget)
                )
            except (AttributeError, TypeError, ValueError):
                # Model or SDK without a configurable thinking budget.
                pass
        return types.GenerateContentConfig(**kwargs)

    # --------------------------- Preflight -------------------------- #

    def preflight(self) -> None:
        """One trivial call, so a bad model id fails now rather than 1,000
        times.

        A retired model id is not a transient error and is not retried,
        so without this the whole run completes "successfully" with every
        record carrying ``status='error'``. Raises :class:`RuntimeError`
        with the ids the key can actually use.
        """
        try:
            self._client.models.generate_content(
                model=self.config.model_name,
                contents="Reply with OK.",
            )
        except Exception as e:  # noqa: BLE001 - re-raised with context
            msg = str(e)
            if "404" not in msg and "NOT_FOUND" not in msg:
                raise RuntimeError(
                    f"Gemini preflight failed for {self.config.model_name!r}: {e}"
                ) from e
            raise RuntimeError(
                f"Gemini rejected the model id {self.config.model_name!r}:\n"
                f"  {msg}\n"
                f"Available to this key: {', '.join(self.available_models()) or '(none)'}\n"
                f"Pass a different --model."
            ) from e

    def available_models(self, contains: str = "flash") -> List[str]:
        """Model ids this key may call, filtered by substring.

        Best-effort: ``models.list`` advertises ids that ``generate_content``
        then refuses, so this is a hint for the error message, not a
        guarantee.
        """
        try:
            return [
                m.name.split("/")[-1]
                for m in self._client.models.list()
                if contains in (m.name or "")
                and "generateContent" in (getattr(m, "supported_actions", None) or [])
            ]
        except Exception:  # noqa: BLE001 - diagnostics only
            return []

    # ---------------------------- Generate -------------------------- #

    def generate(
        self,
        user_prompt: str,
        system_instruction: str,
    ) -> GeminiResponse:
        """Run ``generate_content``, retrying up to ``config.retries`` times.

        Three things are retried, each with exponential backoff:

        * transient exceptions - HTTP 429 / 5xx, quota, timeouts, network;
        * safety blocks, when ``config.retry_blocked`` is set;
        * responses that came back with no text at all.

        A sample that is still blocked after the last attempt is returned
        as a populated :class:`GeminiResponse` with ``blocked=True`` and
        the attempt count, not raised - the caller records it as a real
        result rather than an infrastructure failure. Anything that keeps
        raising is re-raised so the caller can classify the exception.

        Non-transient exceptions (a rejected model id, a bad key) are
        *not* retried: they would fail identically every time and the
        run should stop, not burn its budget.
        """
        cfg = self._build_generate_config(system_instruction)
        max_attempts = max(1, self.config.retries)
        last_exc: Optional[Exception] = None
        last_response: Optional[GeminiResponse] = None

        for attempt in range(max_attempts):
            is_last = (attempt + 1 >= max_attempts)
            try:
                response = self._client.models.generate_content(
                    model=self.config.model_name,
                    contents=user_prompt,
                    config=cfg,
                )
            except Exception as e:  # noqa: BLE001 - SDK raises a wide tree
                last_exc = e
                if is_last or not self._is_transient(str(e).lower()):
                    break
                time.sleep(self.config.backoff_seconds * (2 ** attempt))
                continue

            decoded = self._decode(response)
            decoded.attempts = attempt + 1
            last_response = decoded
            last_exc = None

            if not self._should_retry_response(decoded) or is_last:
                return decoded
            time.sleep(self.config.backoff_seconds * (2 ** attempt))

        if last_response is not None:
            last_response.attempts = max_attempts
            return last_response
        if last_exc is not None:
            raise last_exc
        # Should be unreachable, but keep mypy happy:
        raise RuntimeError("Gemini generate_content returned no result.")

    def _should_retry_response(self, resp: "GeminiResponse") -> bool:
        """True when a *successful* call produced nothing usable.

        A truncated answer is deliberately not retried - the same
        ``max_output_tokens`` would truncate it again. Raise the budget
        instead.
        """
        if resp.blocked:
            return bool(self.config.retry_blocked)
        return not (resp.text or "").strip() and not resp.truncated

    # ----------------------- Response decoding ---------------------- #

    @staticmethod
    def _is_transient(msg: str) -> bool:
        """Return True for retry-eligible error messages."""
        markers = (
            "429", "rate limit", "rate-limit", "quota", "resource_exhausted",
            "500", "502", "503", "504", "internal error",
            "deadline", "timeout", "timed out", "connection",
            "unavailable", "overloaded",
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
        truncated = False

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
            reason = (finish_reason or "").upper()
            if reason in {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII",
                          "RECITATION", "IMAGE_SAFETY"}:
                blocked = True
                block_reason = finish_reason
            elif reason == "MAX_TOKENS":
                # A rewrite that stops mid-sentence is not a usable
                # P_safe. Surface it rather than letting a half-prompt
                # through as if it were complete.
                truncated = True

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
            truncated=truncated,
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
