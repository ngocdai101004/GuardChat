"""Compatibility shim - GuardChat data utilities now live in ``src.utils.data``.

Kept here so existing imports such as ``from src.SafeGuider.data import
GuardChatSample`` keep working. New code should import directly from
:mod:`src.utils.data`.
"""

from src.utils.data import (  # noqa: F401
    CATEGORIES,
    NUM_CATEGORIES,
    GuardChatSample,
    flatten_conversation,
    label_vector_from_labels,
    load_guardchat,
    load_safe_prompts,
    split_texts_and_labels,
)
