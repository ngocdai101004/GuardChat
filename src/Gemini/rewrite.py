"""Task 2 rewrite pipeline using the Gemini 2.5 Flash API.

Mirrors the public surface of :mod:`src.SafeGuider.rewrite` and
:mod:`src.Llama.rewrite` so the benchmark aggregator can compose
Table 2 across the local Llama rewriter and the proprietary Gemini
rewriter without branching.

API only - there is no model checkpoint on disk. The system instruction
is taken from the shared :data:`src.utils.REWRITE_SYSTEM_PROMPT`, the
same prompt the local Llama rewriter uses.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from src.utils import (
    GuardChatSample,
    REWRITE_SYSTEM_PROMPT,
    build_rewrite_user_message,
    cleanup_rewrite_response,
)

from .client import (
    DEFAULT_MODEL_NAME,
    GeminiClient,
    GeminiClientConfig,
    GenerationConfig,
)


@dataclass
class RewriteResult:
    """One rewrite output, ready for serialisation.

    Schema is aligned with :class:`src.Llama.rewrite.RewriteResult` plus
    two Gemini-specific diagnostics (``blocked`` / ``block_reason``)
    used to detect samples where the model refused to generate.
    """

    sample_id: str
    original_prompt: str
    rewritten_prompt: str
    was_modified: bool
    raw_response: str
    elapsed_sec: float
    label_names: List[str]
    source: Optional[str] = None
    blocked: bool = False
    block_reason: Optional[str] = None
    finish_reason: Optional[str] = None
    model_name: str = DEFAULT_MODEL_NAME

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "original_prompt": self.original_prompt,
            "rewritten_prompt": self.rewritten_prompt,
            "was_modified": self.was_modified,
            "raw_response": self.raw_response,
            "elapsed_sec": self.elapsed_sec,
            "label_names": self.label_names,
            "source": self.source,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "finish_reason": self.finish_reason,
            "model_name": self.model_name,
        }


class RewritePipeline:
    """Gemini 2.5 Flash rewriter wired to the GuardChat schema."""

    def __init__(
        self,
        client: GeminiClient,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.client = client
        self.system_prompt = system_prompt or REWRITE_SYSTEM_PROMPT

    @classmethod
    def from_api_key(
        cls,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        max_output_tokens: int = 256,
        temperature: float = 0.0,
        relax_safety: bool = True,
        retries: int = 3,
        backoff_seconds: float = 2.0,
        request_timeout: Optional[float] = 60.0,
        system_prompt: Optional[str] = None,
    ) -> "RewritePipeline":
        """Build a :class:`RewritePipeline` from explicit settings.

        ``api_key`` defaults to ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``
        env vars; raise a friendly error if neither is set.
        """
        cfg = GeminiClientConfig(
            model_name=model_name,
            api_key=api_key,
            relax_safety=relax_safety,
            retries=retries,
            backoff_seconds=backoff_seconds,
            request_timeout=request_timeout,
            generation=GenerationConfig(
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            ),
        )
        client = GeminiClient(cfg)
        return cls(client=client, system_prompt=system_prompt)

    # --------------------------- Inference -------------------------- #

    def rewrite_prompt(
        self,
        prompt: str,
        sample_id: str = "0",
        label_names: Optional[List[str]] = None,
        source: Optional[str] = None,
    ) -> RewriteResult:
        t0 = time.time()
        user_msg = build_rewrite_user_message(prompt)
        try:
            resp = self.client.generate(user_msg, self.system_prompt)
            raw_text = resp.text
            blocked = resp.blocked
            block_reason = resp.block_reason
            finish_reason = resp.finish_reason
        except Exception as e:  # noqa: BLE001 - bubble up as a record entry
            raw_text = ""
            blocked = True
            block_reason = f"exception: {type(e).__name__}: {e}"
            finish_reason = None

        cleaned = cleanup_rewrite_response(raw_text)
        if not cleaned:
            # Either Gemini blocked the call or the response was empty.
            # Fall back to a generic safe alternative so downstream T2I
            # evaluation never receives an empty prompt.
            cleaned = "a serene landscape"

        return RewriteResult(
            sample_id=str(sample_id),
            original_prompt=str(prompt or ""),
            rewritten_prompt=cleaned,
            was_modified=(cleaned.strip().lower() != str(prompt or "").strip().lower()),
            raw_response=raw_text,
            elapsed_sec=round(time.time() - t0, 4),
            label_names=list(label_names or []),
            source=source,
            blocked=bool(blocked),
            block_reason=block_reason,
            finish_reason=finish_reason,
            model_name=self.client.config.model_name,
        )

    def rewrite_samples(
        self,
        samples: Sequence[GuardChatSample],
    ) -> List[RewriteResult]:
        out: List[RewriteResult] = []
        for s in samples:
            out.append(self.rewrite_prompt(
                prompt=s.enhanced_prompt,
                sample_id=str(s.sample_id),
                label_names=s.label_names,
                source=s.source,
            ))
        return out


__all__ = [
    "RewritePipeline",
    "RewriteResult",
]
