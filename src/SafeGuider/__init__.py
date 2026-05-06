"""SafeGuider evaluation code for GuardChat.

Wraps the vendored SafeGuider (vendors/SafeGuider/) and adapts it to the
two GuardChat benchmark tasks:

    Task 1 - Multi-label unsafe text recognition
        Inputs: enhanced single-turn prompt OR concatenated multi-turn
                conversation. Output: 6-way multi-label vector over
                {sexual, illegal, shocking, violence, self-harm, harassment}.
                Metrics: Macro-F1, Recall, ASR.

    Task 2 - NSFW concept removal via prompt rewriting
        Inputs: enhanced toxic prompt. Output: rewritten prompt produced by
                SafetyAwareBeamSearch over CLIP-EOS safety scores.
                Metrics: CLIP cosine similarity (this repo) and downstream
                Safe Generation Rate via T2I models (external).

The vendored SafeGuider folder uses bare imports (e.g. `from classifier
import ThreeLayerClassifier`), so we add it to ``sys.path`` once at import
time. This keeps the third-party copy untouched while still letting us
import its pieces as a library.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_VENDOR_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "vendors", "SafeGuider"))

if not os.path.isdir(_VENDOR_DIR):
    raise ImportError(
        f"Vendored SafeGuider not found at {_VENDOR_DIR!r}. "
        f"Make sure vendors/SafeGuider/ is present at the repo root."
    )

if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)

# Re-export the bits we use most often from vendored SafeGuider.
from classifier import ThreeLayerClassifier  # noqa: E402
from encoder import CLIPEncoder  # noqa: E402
from beam_search import (  # noqa: E402
    SafetyAwareBeamSearch,
    BeamSearchResult,
    DEFAULT_BEAM_WIDTH,
    DEFAULT_MAX_DEPTH,
    DEFAULT_SAFETY_THRESHOLD,
    DEFAULT_SIMILARITY_FLOOR,
)

from .classifier import MultiLabelClassifier
from .data import (
    CATEGORIES,
    GuardChatSample,
    load_guardchat,
    load_safe_prompts,
    label_vector_from_labels,
    flatten_conversation,
)
from .metrics import (
    macro_f1,
    per_class_f1,
    recall_score,
    attack_success_rate,
    clip_cosine_similarity,
)

VENDOR_DIR = _VENDOR_DIR

__all__ = [
    "VENDOR_DIR",
    # Vendored re-exports
    "ThreeLayerClassifier",
    "CLIPEncoder",
    "SafetyAwareBeamSearch",
    "BeamSearchResult",
    "DEFAULT_BEAM_WIDTH",
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_SAFETY_THRESHOLD",
    "DEFAULT_SIMILARITY_FLOOR",
    # GuardChat additions
    "MultiLabelClassifier",
    "CATEGORIES",
    "GuardChatSample",
    "load_guardchat",
    "load_safe_prompts",
    "label_vector_from_labels",
    "flatten_conversation",
    "macro_f1",
    "per_class_f1",
    "recall_score",
    "attack_success_rate",
    "clip_cosine_similarity",
]
