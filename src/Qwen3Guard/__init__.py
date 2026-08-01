"""Qwen3Guard-Gen-8B baseline for GuardChat Task 1 (zero-shot).

Loads ``Qwen/Qwen3Guard-Gen-8B`` from the module's own ``weights/``
folder (populated on first use, or by
:mod:`src.Qwen3Guard.download_weights`) and exposes a
:class:`RecognitionPipeline` with the same surface as the other Task-1
baselines (:mod:`src.ShieldGemma`, :mod:`src.LlamaGuard`,
:mod:`src.SafeGuider`, :mod:`src.BiLSTM`, :mod:`src.BERT`).

This is **inference only** - there is no trainer. Two taxonomy modes are
supported, named to match the sibling baselines:

* ``mode='guardchat'`` (default) - GuardChat's six categories replace the
  shipped nine in the prompt, so the class counts line up one-to-one and
  ``shocking`` is reachable.
* ``mode='native'`` - Qwen3Guard's own nine categories, mapped back to
  GuardChat's six; ``shocking`` is unreachable and two categories
  (``Jailbreak``, ``Politically Sensitive Topics``) stay unmapped by
  design.

Unlike the sibling models, Qwen3Guard grades three severity levels
(``Safe`` / ``Controversial`` / ``Unsafe``); ``controversial_as_unsafe``
decides how the middle one collapses into GuardChat's binary verdict.
"""

from .model import (
    ASSISTANT_PROMPT_TEMPLATE,
    DEFAULT_LOCAL_DIR,
    DEFAULT_MODEL_NAME,
    GenerationConfig,
    Qwen3GuardConfig,
    Qwen3GuardModel,
    USER_PROMPT_TEMPLATE,
    build_custom_prompt,
    render_conversation,
)
from .recognition import CONV_FORMATS, RecognitionPipeline, RecognitionPrediction
from .taxonomy import (
    GUARDCHAT_CATEGORIES,
    GUARDCHAT_TO_GUARDCHAT,
    GUARDCHAT_TO_NATIVE,
    MODES,
    NATIVE_CATEGORIES,
    NATIVE_TO_GUARDCHAT,
    SEVERITIES,
    SEVERITY_LABELS,
    UNMAPPED_NATIVE_CATEGORIES,
    canonicalise_category,
    categories_for_mode,
    categories_to_guardchat_vector,
    category_block,
    category_map_for_mode,
    normalise_mode,
    parse_qwen3guard_response,
    severity_is_unsafe,
    unreachable_categories,
)

__all__ = [
    "DEFAULT_MODEL_NAME",
    "DEFAULT_LOCAL_DIR",
    "GenerationConfig",
    "Qwen3GuardConfig",
    "Qwen3GuardModel",
    "USER_PROMPT_TEMPLATE",
    "ASSISTANT_PROMPT_TEMPLATE",
    "build_custom_prompt",
    "render_conversation",
    "CONV_FORMATS",
    "RecognitionPipeline",
    "RecognitionPrediction",
    "MODES",
    "SEVERITIES",
    "SEVERITY_LABELS",
    "NATIVE_CATEGORIES",
    "NATIVE_TO_GUARDCHAT",
    "UNMAPPED_NATIVE_CATEGORIES",
    "GUARDCHAT_CATEGORIES",
    "GUARDCHAT_TO_GUARDCHAT",
    "GUARDCHAT_TO_NATIVE",
    "canonicalise_category",
    "categories_for_mode",
    "categories_to_guardchat_vector",
    "category_block",
    "category_map_for_mode",
    "normalise_mode",
    "parse_qwen3guard_response",
    "severity_is_unsafe",
    "unreachable_categories",
]
