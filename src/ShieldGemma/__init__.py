"""ShieldGemma-2B baseline for GuardChat Task 1 (zero-shot).

Loads ``google/shieldgemma-2b`` (from the Hub or a local snapshot
populated by :mod:`src.ShieldGemma.download_weights`) and exposes a
:class:`RecognitionPipeline` with the same surface as the other Task-1
baselines (:mod:`src.LlamaGuard`, :mod:`src.SafeGuider`,
:mod:`src.BiLSTM`, :mod:`src.BERT`).

This is **inference only** - there is no trainer. ShieldGemma judges one
safety policy at a time, so a multi-label vector is built by scoring each
policy independently and thresholding ``P(Yes)``. Two policy sets are
available:

* ``mode='guardchat'`` (default) - six policies, one per GuardChat
  category, so the class counts line up one-to-one.
* ``mode='native'`` - ShieldGemma's four published policies plus a lossy
  many-to-one mapping; ``shocking`` is unreachable in this mode.
"""

from .hf_token import resolve_hf_token
from .model import (
    DEFAULT_LOCAL_DIR,
    DEFAULT_MODEL_NAME,
    PROMPT_TEMPLATE,
    ShieldGemmaConfig,
    ShieldGemmaModel,
    build_scoring_prompt,
    ensure_local_snapshot,
    resolve_device,
)
from .recognition import (
    TEXT_KINDS,
    RecognitionPipeline,
    RecognitionPrediction,
    normalise_kind,
    text_for_kind,
)
from .taxonomy import (
    GUARDCHAT_POLICIES,
    GUARDCHAT_POLICY_TO_GUARDCHAT,
    MODES,
    NATIVE_POLICIES,
    NATIVE_POLICY_TO_GUARDCHAT,
    policies_for_mode,
    policy_map_for_mode,
    scores_to_guardchat_vector,
    unreachable_categories,
)

__all__ = [
    "DEFAULT_MODEL_NAME",
    "DEFAULT_LOCAL_DIR",
    "PROMPT_TEMPLATE",
    "ShieldGemmaConfig",
    "ShieldGemmaModel",
    "build_scoring_prompt",
    "ensure_local_snapshot",
    "resolve_device",
    "resolve_hf_token",
    "TEXT_KINDS",
    "RecognitionPipeline",
    "RecognitionPrediction",
    "normalise_kind",
    "text_for_kind",
    "MODES",
    "NATIVE_POLICIES",
    "NATIVE_POLICY_TO_GUARDCHAT",
    "GUARDCHAT_POLICIES",
    "GUARDCHAT_POLICY_TO_GUARDCHAT",
    "policies_for_mode",
    "policy_map_for_mode",
    "scores_to_guardchat_vector",
    "unreachable_categories",
]
