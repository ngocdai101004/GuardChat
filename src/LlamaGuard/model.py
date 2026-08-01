"""Llama-Guard-3-8B model wrapper for GuardChat Task 1 (zero-shot).

Loads ``meta-llama/Llama-Guard-3-8B`` from the module's own ``weights/``
folder - populated on first use, or ahead of time by
:mod:`src.LlamaGuard.download_weights` - and exposes a single
:meth:`LlamaGuardModel.moderate` call that:

    1. wraps the input prompt / conversation in the model's chat
       template (optionally overriding the default S1-S14 hazard
       taxonomy with GuardChat's six categories),
    2. runs a short causal generation (``max_new_tokens=20`` is enough
       for the ``safe`` / ``unsafe\\nS3`` verdict),
    3. returns the raw decoded string for the caller to parse via
       :func:`taxonomy.parse_llamaguard_response`.

Access
------
Llama-Guard-3-8B is gated on HuggingFace. Accept the licence at
https://huggingface.co/meta-llama/Llama-Guard-3-8B, then expose a token
via ``HF_TOKEN`` (see :mod:`src.utils.hf_token`).

Memory / dtype
--------------
The 8B parameter model needs ~16 GB in bf16 / fp16. Use ``dtype='nf4'``
or ``dtype='int8'`` (requires ``bitsandbytes`` >= 0.43) to fit on
smaller GPUs:

==================  =============  ========================================
``dtype``            footprint      notes
==================  =============  ========================================
``bfloat16``        ~16 GB         default on an accelerator; recommended
``float16``         ~16 GB         if bf16 unsupported
``float32``         ~32 GB         default on CPU; rarely useful
``int8``            ~9 GB          needs bitsandbytes
``nf4``             ~5 GB          needs bitsandbytes; 4-bit NF4
==================  =============  ========================================

Tested with: ``torch>=2.1``, ``transformers>=4.43`` (Llama 3.1 support),
``accelerate>=0.26``, ``huggingface_hub>=0.23``,
``bitsandbytes>=0.43`` (only when ``dtype in {'int8', 'nf4'}``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

import torch

from src.utils import (
    bnb_config,
    default_dtype_for,
    from_pretrained_with_dtype,
    resolve_device,
    resolve_weights_path,
)

from .taxonomy import parse_llamaguard_response


DEFAULT_MODEL_NAME = "meta-llama/Llama-Guard-3-8B"
DEFAULT_LOCAL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "weights", "Llama-Guard-3-8B"
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
    dtype: str = "auto"
    device: Optional[str] = None
    token: Optional[str] = None
    custom_categories: Optional[Mapping[str, str]] = None
    excluded_category_keys: Optional[Sequence[str]] = None
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    # Fetch the snapshot into ``model_path`` when that folder is missing,
    # instead of leaving ~16 GB in the shared ~/.cache/huggingface tree.
    auto_download: bool = True
    repo_id: str = DEFAULT_MODEL_NAME


class LlamaGuardModel:
    """Loaded Llama-Guard-3-8B + helpers to run a single moderation call."""

    def __init__(self, config: LlamaGuardConfig = LlamaGuardConfig()) -> None:
        self.config = config
        self._load()

    # --------------------------- Loading ----------------------------- #

    def _load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        token = self.config.token
        auth: Dict[str, Any] = {"token": token} if token else {}

        path = resolve_weights_path(
            self.config.model_path,
            repo_id=self.config.repo_id,
            token=token,
            auto_download=self.config.auto_download,
            log_prefix="LlamaGuard",
        )

        self.device = resolve_device(self.config.device)
        dtype_name = self.config.dtype
        if not dtype_name or dtype_name == "auto":
            dtype_name = default_dtype_for(self.device)
        self.dtype_name = dtype_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            path, clean_up_tokenization_spaces=True, **auth
        )

        bnb = bnb_config(dtype_name)
        if bnb is not None:
            # device_map="auto" cooperates with bitsandbytes' weight
            # placement; an explicit ``.to(device)`` is unnecessary.
            self.model = AutoModelForCausalLM.from_pretrained(
                path, quantization_config=bnb, device_map="auto", **auth
            )
        else:
            self.model = from_pretrained_with_dtype(
                AutoModelForCausalLM, path, dtype_name,
                device_map={"": str(self.device)}, **auth
            )
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
