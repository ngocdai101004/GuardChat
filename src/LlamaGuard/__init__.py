"""Llama-Guard-3-8B baseline for GuardChat Task 1 (zero-shot).

Loads ``meta-llama/Llama-Guard-3-8B`` from the module's own ``weights/``
folder (populated on first use, or by
:mod:`src.LlamaGuard.download_weights`) and exposes a
:class:`RecognitionPipeline` with the same surface as the other Task-1
baselines (:mod:`src.ShieldGemma`, :mod:`src.SafeGuider`,
:mod:`src.BiLSTM`, :mod:`src.BERT`).

This is **inference only** - there is no trainer. Two taxonomy modes are
supported, named to match :mod:`src.ShieldGemma`:

* ``mode='guardchat'`` (default) - pass GuardChat's six categories
  straight into the chat template, so the class counts line up
  one-to-one and ``shocking`` is reachable.
* ``mode='native'`` - use Llama-Guard's own S1-S14 hazard taxonomy and
  map S-codes back to GuardChat's six categories; ``shocking`` is
  unreachable in this mode.
"""

from .model import (
    DEFAULT_LOCAL_DIR,
    DEFAULT_MODEL_NAME,
    GenerationConfig,
    LlamaGuardConfig,
    LlamaGuardModel,
)
from .recognition import CONV_FORMATS, RecognitionPipeline, RecognitionPrediction
from .taxonomy import (
    CUSTOM_SCODE_TO_GUARDCHAT,
    GUARDCHAT_CUSTOM_CATEGORIES,
    GUARDCHAT_TO_SCODES,
    LLAMAGUARD3_CATEGORIES,
    MODES,
    SCODE_TO_GUARDCHAT,
    normalise_mode,
    parse_llamaguard_response,
    scode_map_for_mode,
    scodes_to_guardchat_vector,
    unreachable_categories,
)

__all__ = [
    "DEFAULT_MODEL_NAME",
    "DEFAULT_LOCAL_DIR",
    "GenerationConfig",
    "LlamaGuardConfig",
    "LlamaGuardModel",
    "CONV_FORMATS",
    "RecognitionPipeline",
    "RecognitionPrediction",
    "MODES",
    "LLAMAGUARD3_CATEGORIES",
    "SCODE_TO_GUARDCHAT",
    "GUARDCHAT_TO_SCODES",
    "GUARDCHAT_CUSTOM_CATEGORIES",
    "CUSTOM_SCODE_TO_GUARDCHAT",
    "normalise_mode",
    "parse_llamaguard_response",
    "scode_map_for_mode",
    "scodes_to_guardchat_vector",
    "unreachable_categories",
]
