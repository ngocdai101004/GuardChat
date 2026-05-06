"""Llama-3.1-8B-Instruct baseline for GuardChat Task 2 (NSFW concept removal).

Loads ``meta-llama/Llama-3.1-8B-Instruct`` from a local snapshot
(populated by :mod:`src.Llama.download_weights`) and exposes a
:class:`RewritePipeline` with the same surface as
:mod:`src.SafeGuider.rewrite`.

This is **inference only** - the rewriter is configured entirely by
the shared system prompt in :mod:`src.utils.rewrite_prompt`.
"""

from .model import (
    DEFAULT_LOCAL_DIR,
    DEFAULT_MODEL_NAME,
    GenerationConfig,
    LlamaConfig,
    LlamaModel,
)
from .rewrite import RewritePipeline, RewriteResult

__all__ = [
    "DEFAULT_MODEL_NAME",
    "DEFAULT_LOCAL_DIR",
    "GenerationConfig",
    "LlamaConfig",
    "LlamaModel",
    "RewritePipeline",
    "RewriteResult",
]
