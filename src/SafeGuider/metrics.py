"""Compatibility shim - metrics now live in ``src.utils.metrics``."""

from src.utils.metrics import (  # noqa: F401
    attack_success_rate,
    binary_from_multilabel,
    clip_cosine_similarity,
    macro_f1,
    per_class_f1,
    recall_score,
    summarise_recognition,
)
