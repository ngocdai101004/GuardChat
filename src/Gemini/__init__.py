"""Gemini 2.5 Flash baseline for GuardChat Task 2 (NSFW concept removal).

API-only baseline backed by ``google-genai``. No model weights are
downloaded - the only requirement is a valid Gemini API key in the
``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) environment variable.

Rewrites both GuardChat input representations - the enhanced prompt and
the multi-turn conversation - and serialises to the shared
:class:`src.utils.RewriteRecord` schema, so the benchmark aggregator can
compose Table 2 across this and the local Llama rewriter without
branching.
"""

from .client import (
    DEFAULT_MODEL_NAME,
    GeminiClient,
    GeminiClientConfig,
    GeminiResponse,
    GenerationConfig,
)
from .rewrite import DEFAULT_MAX_OUTPUT_TOKENS, RewritePipeline

__all__ = [
    "DEFAULT_MODEL_NAME",
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "GeminiClient",
    "GeminiClientConfig",
    "GeminiResponse",
    "GenerationConfig",
    "RewritePipeline",
]
