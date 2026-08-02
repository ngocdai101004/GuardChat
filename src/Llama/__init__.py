"""Llama-3.1-8B-Instruct baseline for GuardChat Task 2 (NSFW concept removal).

Loads ``meta-llama/Llama-3.1-8B-Instruct`` from a local snapshot
(populated by :mod:`src.Llama.download_weights`) and rewrites both
GuardChat input representations - the enhanced prompt and the multi-turn
conversation.

Inference only. The rewriter is configured entirely by the shared system
prompts in :mod:`src.utils.rewrite_prompt`, the same ones the Gemini
baseline uses, and serialises to the shared
:class:`src.utils.RewriteRecord` schema - so Table 2 compares the two
models rather than two task framings.
"""

from .model import (
    DEFAULT_LOCAL_DIR,
    DEFAULT_MODEL_NAME,
    MAX_NEW_TOKENS_CAP,
    GenerationConfig,
    LlamaConfig,
    LlamaModel,
)
from .rewrite import RewritePipeline

__all__ = [
    "DEFAULT_MODEL_NAME",
    "DEFAULT_LOCAL_DIR",
    "MAX_NEW_TOKENS_CAP",
    "GenerationConfig",
    "LlamaConfig",
    "LlamaModel",
    "RewritePipeline",
]
