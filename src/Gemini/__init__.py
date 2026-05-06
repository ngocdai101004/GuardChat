"""Gemini 2.5 Flash baseline for GuardChat Task 2 (NSFW concept removal).

API-only baseline backed by ``google-genai``. No model weights are
downloaded - the only requirement is a valid Gemini API key in the
``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) environment variable.

Public surface mirrors :mod:`src.Llama.rewrite` so the benchmark
aggregator can compose Table 2 across the local Llama rewriter and
the proprietary Gemini rewriter without branching.
"""

from .client import (
    DEFAULT_MODEL_NAME,
    GeminiClient,
    GeminiClientConfig,
    GeminiResponse,
    GenerationConfig,
)
from .rewrite import RewritePipeline, RewriteResult

__all__ = [
    "DEFAULT_MODEL_NAME",
    "GeminiClient",
    "GeminiClientConfig",
    "GeminiResponse",
    "GenerationConfig",
    "RewritePipeline",
    "RewriteResult",
]
