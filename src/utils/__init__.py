"""Shared GuardChat utilities used by all baselines.

Centralises:

    * The canonical six-category schema for unsafe text recognition.
    * The :class:`GuardChatSample` data model and JSON / JSONL loaders,
      plus the three Task-1 input representations (``prompt`` /
      ``raw_prompt`` / ``conversation``).
    * Task 1 / Task 2 evaluation metrics (Macro-F1, Recall, ASR,
      CLIP cosine similarity).
    * HuggingFace plumbing for the zero-shot baselines: token
      resolution, snapshot download into a module-local ``weights/``
      folder, device / dtype selection.
    * The resumable Task-1 evaluation loop shared by the zero-shot CLIs.

Every baseline (SafeGuider, BiLSTM, BERT, zero-shot LLMs) imports from
here so that label ordering, evaluation outputs, and CLI schemas stay
in sync.
"""

from .data import (
    CATEGORIES,
    DEFAULT_HF_REPO,
    DEFAULT_HF_SPLIT,
    NUM_CATEGORIES,
    TEXT_KINDS,
    GuardChatSample,
    flatten_conversation,
    gold_category,
    label_vector_from_labels,
    load_guardchat,
    load_safe_prompts,
    normalise_text_kind,
    split_texts_and_labels,
    text_for_kind,
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
from .hf_model import (
    DEFAULT_IGNORE_PATTERNS,
    DTYPE_CHOICES,
    bnb_config,
    default_dtype_for,
    ensure_local_snapshot,
    from_pretrained_with_dtype,
    is_hub_id,
    resolve_device,
    resolve_torch_dtype,
    resolve_weights_path,
    snapshot_is_present,
)
from .hf_token import ENV_KEYS as HF_TOKEN_ENV_KEYS, resolve_hf_token
from .rewrite_prompt import (
    SYSTEM_PROMPT as REWRITE_SYSTEM_PROMPT,
    USER_TEMPLATE as REWRITE_USER_TEMPLATE,
    build_chat_messages as build_rewrite_messages,
    build_user_message as build_rewrite_user_message,
    cleanup_response as cleanup_rewrite_response,
)
from .task1_eval import (
    checkpoint_path,
    evaluate_kind,
    load_checkpoint,
    output_path,
    print_metrics,
    save_kind,
)

__all__ = [
    "CATEGORIES",
    "DEFAULT_HF_REPO",
    "DEFAULT_HF_SPLIT",
    "NUM_CATEGORIES",
    "TEXT_KINDS",
    "GuardChatSample",
    "flatten_conversation",
    "gold_category",
    "label_vector_from_labels",
    "load_guardchat",
    "load_safe_prompts",
    "normalise_text_kind",
    "split_texts_and_labels",
    "text_for_kind",
    "attack_success_rate",
    "binary_from_multilabel",
    "clip_cosine_similarity",
    "macro_f1",
    "per_class_f1",
    "recall_score",
    "summarise_recognition",
    "DEFAULT_IGNORE_PATTERNS",
    "DTYPE_CHOICES",
    "bnb_config",
    "default_dtype_for",
    "ensure_local_snapshot",
    "from_pretrained_with_dtype",
    "is_hub_id",
    "resolve_device",
    "resolve_torch_dtype",
    "resolve_weights_path",
    "snapshot_is_present",
    "HF_TOKEN_ENV_KEYS",
    "resolve_hf_token",
    "REWRITE_SYSTEM_PROMPT",
    "REWRITE_USER_TEMPLATE",
    "build_rewrite_messages",
    "build_rewrite_user_message",
    "cleanup_rewrite_response",
    "checkpoint_path",
    "evaluate_kind",
    "load_checkpoint",
    "output_path",
    "print_metrics",
    "save_kind",
]
