"""Qwen2.5-7B-Instruct baseline for GuardChat Task 1 (zero-shot).

Loads ``Qwen/Qwen2.5-7B-Instruct`` from a local snapshot (populated by
:mod:`src.Qwen.download_weights`) and exposes a
:class:`RecognitionPipeline` with the same surface as the supervised
baselines (:mod:`src.SafeGuider`, :mod:`src.BiLSTM`, :mod:`src.BERT`)
and the other zero-shot baseline (:mod:`src.LlamaGuard`).

This is **inference only**. The classifier is configured entirely
through :data:`src.Qwen.prompts.SYSTEM_PROMPT`, which:

* lists the six GuardChat NSFW categories with definitions paraphrased
  from Appendix A of the paper,
* asks the model to flag every category that applies, taking the
  *overall* multi-turn intent into account, and
* pins the response to a strict 6-key JSON object so parsing is
  deterministic.

A robust :func:`prompts.parse_response` tolerates the most common Qwen
deviations (markdown fences, key spelling variants, trailing prose).
"""

from .model import (
    DEFAULT_LOCAL_DIR,
    DEFAULT_MODEL_NAME,
    GenerationConfig,
    QwenConfig,
    QwenModel,
)
from .prompts import (
    CATEGORY_DEFINITIONS,
    DEFAULT_JSON_SKELETON,
    SYSTEM_PROMPT,
    USER_TEMPLATE_CONVERSATION,
    USER_TEMPLATE_SINGLE,
    build_chat_messages,
    build_user_message,
    parse_response,
)
from .recognition import RecognitionPipeline, RecognitionPrediction

__all__ = [
    "DEFAULT_MODEL_NAME",
    "DEFAULT_LOCAL_DIR",
    "GenerationConfig",
    "QwenConfig",
    "QwenModel",
    "RecognitionPipeline",
    "RecognitionPrediction",
    "CATEGORY_DEFINITIONS",
    "SYSTEM_PROMPT",
    "USER_TEMPLATE_SINGLE",
    "USER_TEMPLATE_CONVERSATION",
    "DEFAULT_JSON_SKELETON",
    "build_chat_messages",
    "build_user_message",
    "parse_response",
]
