"""Shared GuardChat utilities used by all baselines.

Centralises:

    * The canonical six-category schema for unsafe text recognition.
    * The :class:`GuardChatSample` data model and JSON / JSONL loaders.
    * Task 1 / Task 2 evaluation metrics (Macro-F1, Recall, ASR,
      CLIP cosine similarity).

Every baseline (SafeGuider, BiLSTM, BERT, zero-shot LLMs) imports from
here so that label ordering, evaluation outputs, and CLI schemas stay
in sync.
"""

from .data import (
    CATEGORIES,
    NUM_CATEGORIES,
    GuardChatSample,
    flatten_conversation,
    label_vector_from_labels,
    load_guardchat,
    load_safe_prompts,
    split_texts_and_labels,
)
from .metrics import (
    attack_success_rate,
    binary_from_multilabel,
    clip_cosine_similarity,
    macro_f1,
    per_class_f1,
    recall_score,
    summarise_recognition,
)
from .rewrite_prompt import (
    SYSTEM_PROMPT as REWRITE_SYSTEM_PROMPT,
    USER_TEMPLATE as REWRITE_USER_TEMPLATE,
    build_chat_messages as build_rewrite_messages,
    build_user_message as build_rewrite_user_message,
    cleanup_response as cleanup_rewrite_response,
)

__all__ = [
    "CATEGORIES",
    "NUM_CATEGORIES",
    "GuardChatSample",
    "flatten_conversation",
    "label_vector_from_labels",
    "load_guardchat",
    "load_safe_prompts",
    "split_texts_and_labels",
    "attack_success_rate",
    "binary_from_multilabel",
    "clip_cosine_similarity",
    "macro_f1",
    "per_class_f1",
    "recall_score",
    "summarise_recognition",
    "REWRITE_SYSTEM_PROMPT",
    "REWRITE_USER_TEMPLATE",
    "build_rewrite_messages",
    "build_rewrite_user_message",
    "cleanup_rewrite_response",
]
