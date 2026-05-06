"""Llama-Guard-3-8B baseline for GuardChat Task 1 (zero-shot).

Loads ``meta-llama/Llama-Guard-3-8B`` from a local snapshot (populated
by :mod:`src.LlamaGuard.download_weights`) and exposes a
:class:`RecognitionPipeline` with the same surface as the supervised
baselines (:mod:`src.SafeGuider`, :mod:`src.BiLSTM`, :mod:`src.BERT`).

This is **inference only** - there is no trainer. Two taxonomy modes
are supported:

* ``mode='native'`` (default) - use Llama-Guard's S1-S14 hazard
  taxonomy and map S-codes back to GuardChat's six categories.
* ``mode='custom'`` - pass GuardChat's six categories straight into the
  chat template; the model reasons zero-shot over them.
"""

from .model import (
    DEFAULT_LOCAL_DIR,
    DEFAULT_MODEL_NAME,
    GenerationConfig,
    LlamaGuardConfig,
    LlamaGuardModel,
)
from .recognition import RecognitionPipeline, RecognitionPrediction
from .taxonomy import (
    CUSTOM_SCODE_TO_GUARDCHAT,
    GUARDCHAT_CUSTOM_CATEGORIES,
    GUARDCHAT_TO_SCODES,
    LLAMAGUARD3_CATEGORIES,
    SCODE_TO_GUARDCHAT,
    parse_llamaguard_response,
    scodes_to_guardchat_vector,
)

__all__ = [
    "DEFAULT_MODEL_NAME",
    "DEFAULT_LOCAL_DIR",
    "GenerationConfig",
    "LlamaGuardConfig",
    "LlamaGuardModel",
    "RecognitionPipeline",
    "RecognitionPrediction",
    "LLAMAGUARD3_CATEGORIES",
    "SCODE_TO_GUARDCHAT",
    "GUARDCHAT_TO_SCODES",
    "GUARDCHAT_CUSTOM_CATEGORIES",
    "CUSTOM_SCODE_TO_GUARDCHAT",
    "parse_llamaguard_response",
    "scodes_to_guardchat_vector",
]
