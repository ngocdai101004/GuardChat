"""Llama-Guard-3-8B model wrapper for GuardChat Task 1 (zero-shot).

This module loads ``meta-llama/Llama-Guard-3-8B`` from a *local* folder
populated by :mod:`src.LlamaGuard.download_weights` (default
``src/LlamaGuard/weights/Llama-Guard-3-8B``) and exposes a single
:meth:`LlamaGuardModel.moderate` call that:

    1. wraps the input prompt / conversation in the model's chat
       template (optionally overriding the default S1-S14 hazard
       taxonomy with GuardChat's six categories),
    2. runs a short causal generation (``max_new_tokens=20`` is enough
       for the ``safe`` / ``unsafe\\nS3`` verdict),
    3. returns the raw decoded string for the caller to parse via
       :func:`taxonomy.parse_llamaguard_response`.

Loading
-------
Llama-Guard-3-8B is gated on HuggingFace. Before the first run, accept
the licence at https://huggingface.co/meta-llama/Llama-Guard-3-8B and
authenticate (``huggingface-cli login`` or ``HF_TOKEN`` env var). Then
populate the local cache using ``download_weights.py``.

Memory / dtype
--------------
The 8B parameter model needs ~16 GB in bf16 / fp16. Use ``dtype='nf4'``
or ``dtype='int8'`` (requires ``bitsandbytes`` >= 0.43) to fit on
smaller GPUs:

================  ============  =====================
``dtype``          GPU footprint  notes
================  ============  =====================
``bfloat16``      ~16 GB        default, recommended on H100/A100
``float16``       ~16 GB        if bf16 unsupported
``float32``       ~32 GB        rarely useful
``int8``          ~9 GB         needs bitsandbytes
``nf4``           ~5 GB         needs bitsandbytes; 4-bit NF4
================  ============  =====================

Tested with: ``torch>=2.1``, ``transformers>=4.43`` (Llama 3.1 support),
``accelerate>=0.26``, ``huggingface_hub>=0.20``,
``bitsandbytes>=0.43`` (only when ``dtype in {'int8', 'nf4'}``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from .taxonomy import (
    GUARDCHAT_CUSTOM_CATEGORIES,
    parse_llamaguard_response,
)


DEFAULT_MODEL_NAME = "meta-llama/Llama-Guard-3-8B"
DEFAULT_LOCAL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "weights", "Llama-Guard-3-8B"
)


# Dtypes are exposed as strings on the CLI; resolve them lazily so that
# we don't trigger an import of bitsandbytes when the user picks bf16.
_DTYPE_NAMES = {
    "bfloat16", "bf16",
    "float16", "fp16",
    "float32", "fp32",
    "int8", "8bit",
    "nf4", "4bit",
}


def _resolve_torch_dtype(name: str) -> torch.dtype:
    n = name.lower()
    if n in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if n in {"float16", "fp16"}:
        return torch.float16
    if n in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(
        f"Unsupported torch dtype {name!r}. Use one of "
        f"{sorted(_DTYPE_NAMES)}."
    )


def _bnb_config(name: str):
    """Build a ``BitsAndBytesConfig`` for int8 / nf4 quantisation."""
    n = name.lower()
    if n not in {"int8", "8bit", "nf4", "4bit"}:
        return None
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as e:  # pragma: no cover - hard runtime dep
        raise RuntimeError(
            "Quantised loading needs transformers >= 4.43 with the "
            "BitsAndBytesConfig API."
        ) from e
    if n in {"int8", "8bit"}:
        return BitsAndBytesConfig(load_in_8bit=True)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


@dataclass
class GenerationConfig:
    """Sampling settings for the safety verdict.

    Greedy decoding is the canonical Meta reference behaviour: the
    "safe" / "unsafe + S-codes" verdict is short and deterministic.
    """

    max_new_tokens: int = 20
    do_sample: bool = False
    temperature: float = 0.0


@dataclass
class LlamaGuardConfig:
    model_path: str = DEFAULT_LOCAL_DIR
    dtype: str = "bfloat16"
    device: Optional[str] = None
    custom_categories: Optional[Mapping[str, str]] = None
    excluded_category_keys: Optional[Sequence[str]] = None
    generation: GenerationConfig = field(default_factory=GenerationConfig)


class LlamaGuardModel:
    """Loaded Llama-Guard-3-8B + helpers to run a single moderation call."""

    def __init__(self, config: LlamaGuardConfig = LlamaGuardConfig()) -> None:
        self.config = config
        self._load()

    # --------------------------- Loading ----------------------------- #

    def _load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = self.config.model_path
        if not (os.path.isdir(path) or "/" in path and not os.path.isabs(path)):
            # Allow either a HF id (with "/") or a local directory.
            raise FileNotFoundError(
                f"Llama-Guard weights not found at {path!r}. "
                f"Either pass a HuggingFace model id or run "
                f"`python -m src.LlamaGuard.download_weights` to populate "
                f"the local cache."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            path, clean_up_tokenization_spaces=True,
        )

        kwargs: Dict[str, Any] = {}
        bnb = _bnb_config(self.config.dtype)
        if bnb is not None:
            kwargs["quantization_config"] = bnb
            # device_map="auto" cooperates with bitsandbytes' weight
            # placement; explicit ``.to(device)`` is unnecessary.
            kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = _resolve_torch_dtype(self.config.dtype)
            if self.config.device is not None:
                kwargs["device_map"] = {"": self.config.device}
            else:
                kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
        self.model.eval()
        # Cache the parameter device for tensor placement at call time.
        self.device = next(self.model.parameters()).device

    # ----------------- Chat-template / generation ------------------- #

    def _apply_chat_template(self, chat: Sequence[Dict[str, str]]) -> torch.Tensor:
        """Format a chat for Llama-Guard, optionally overriding categories.

        ``custom_categories`` and ``excluded_category_keys`` are forwarded
        verbatim to ``tokenizer.apply_chat_template``. The HuggingFace
        Llama-Guard 3 tokenizer accepts both kwargs - older versions
        only honour one of them, so we silently drop kwargs the
        installed tokenizer does not understand.
        """
        kwargs: Dict[str, Any] = {"return_tensors": "pt"}
        if self.config.custom_categories is not None:
            kwargs["categories"] = dict(self.config.custom_categories)
        if self.config.excluded_category_keys:
            kwargs["excluded_category_keys"] = list(self.config.excluded_category_keys)

        try:
            input_ids = self.tokenizer.apply_chat_template(list(chat), **kwargs)
        except (TypeError, ValueError):
            # Drop unrecognised kwargs and retry with the minimum.
            input_ids = self.tokenizer.apply_chat_template(
                list(chat), return_tensors="pt",
            )
        return input_ids.to(self.device)

    @torch.no_grad()
    def moderate(self, chat: Sequence[Dict[str, str]]) -> str:
        """Run the safety verdict for a single chat turn list.

        ``chat`` follows the standard ``[{"role": "user", "content": ...},
        ...]`` schema. We classify the *user* turns by default, which is
        the right granularity for Task 1 (input filtering before T2I).
        """
        if not chat:
            return "safe"

        input_ids = self._apply_chat_template(chat)
        gen = self.config.generation
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": gen.max_new_tokens,
            "do_sample": gen.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id or 0,
        }
        if gen.do_sample:
            gen_kwargs["temperature"] = gen.temperature

        output = self.model.generate(input_ids=input_ids, **gen_kwargs)
        new_tokens = output[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    # --------------------------- Convenience ------------------------ #

    def classify_chat(self, chat: Sequence[Dict[str, str]]):
        """Run :meth:`moderate` and parse the response in one step.

        Returns ``(is_unsafe, scodes, raw_response)``.
        """
        raw = self.moderate(chat)
        is_unsafe, codes = parse_llamaguard_response(raw)
        return is_unsafe, codes, raw


__all__ = [
    "DEFAULT_MODEL_NAME",
    "DEFAULT_LOCAL_DIR",
    "GenerationConfig",
    "LlamaGuardConfig",
    "LlamaGuardModel",
]
