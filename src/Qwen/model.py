"""Qwen2.5-7B-Instruct model wrapper for GuardChat Task 1 (zero-shot).

Loads ``Qwen/Qwen2.5-7B-Instruct`` from a local snapshot folder
(populated by :mod:`src.Qwen.download_weights`, default
``src/Qwen/weights/Qwen2.5-7B-Instruct``) and exposes a single
:meth:`QwenModel.generate_classification` call that:

    1. wraps the input in :data:`prompts.SYSTEM_PROMPT` plus a
       category-aware user message (see :mod:`prompts`),
    2. runs a short causal generation (``max_new_tokens=64`` is enough
       for the strict 6-key JSON object we ask the model to emit),
    3. returns the decoded string for the caller to parse via
       :func:`prompts.parse_response`.

Memory / dtype
--------------
The 7B parameter model needs ~14 GB in bf16/fp16. Use ``dtype='nf4'``
or ``dtype='int8'`` (requires ``bitsandbytes`` >= 0.43) to fit on
smaller GPUs:

================  ============  =====================
``dtype``          GPU footprint  notes
================  ============  =====================
``bfloat16``      ~14 GB        default, recommended
``float16``       ~14 GB        if bf16 unsupported
``float32``       ~28 GB        rarely useful
``int8``          ~8 GB         needs bitsandbytes
``nf4``           ~4 GB         needs bitsandbytes; 4-bit NF4
================  ============  =====================

Tested with: ``torch>=2.1``, ``transformers>=4.43`` (Qwen2 native
support landed in 4.37; we pin to 4.43 to share the runtime baseline
with Llama-Guard), ``accelerate>=0.26``, ``huggingface_hub>=0.20``,
``bitsandbytes>=0.43`` (only when ``dtype in {'int8','nf4'}``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_LOCAL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "weights", "Qwen2.5-7B-Instruct"
)


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
        f"Unsupported torch dtype {name!r}. Use one of {sorted(_DTYPE_NAMES)}."
    )


def _bnb_config(name: str):
    n = name.lower()
    if n not in {"int8", "8bit", "nf4", "4bit"}:
        return None
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as e:  # pragma: no cover
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
    """Sampling settings for the classification verdict.

    Greedy decoding is the right default - we want the model to
    deterministically emit the same JSON skeleton every time. Sampling
    introduces unnecessary variance for a structured output task.
    """

    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0


@dataclass
class QwenConfig:
    model_path: str = DEFAULT_LOCAL_DIR
    dtype: str = "bfloat16"
    device: Optional[str] = None
    trust_remote_code: bool = False
    generation: GenerationConfig = field(default_factory=GenerationConfig)


class QwenModel:
    """Loaded Qwen2.5-7B-Instruct + helpers for zero-shot classification."""

    def __init__(self, config: QwenConfig = QwenConfig()) -> None:
        self.config = config
        self._load()

    # --------------------------- Loading ----------------------------- #

    def _load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = self.config.model_path
        if not (os.path.isdir(path) or "/" in path and not os.path.isabs(path)):
            raise FileNotFoundError(
                f"Qwen weights not found at {path!r}. "
                f"Either pass a HuggingFace model id or run "
                f"`python -m src.Qwen.download_weights` to populate the "
                f"local cache."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            path,
            clean_up_tokenization_spaces=True,
            trust_remote_code=self.config.trust_remote_code,
        )

        kwargs: Dict[str, Any] = {
            "trust_remote_code": self.config.trust_remote_code,
        }
        bnb = _bnb_config(self.config.dtype)
        if bnb is not None:
            kwargs["quantization_config"] = bnb
            kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = _resolve_torch_dtype(self.config.dtype)
            if self.config.device is not None:
                kwargs["device_map"] = {"": self.config.device}
            else:
                kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
        self.model.eval()
        self.device = next(self.model.parameters()).device

    # ----------------- Chat-template / generation ------------------- #

    def _apply_chat_template(self, messages: Sequence[Dict[str, str]]) -> torch.Tensor:
        """Format messages for Qwen's chat template and return ``input_ids``."""
        text = self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
        )
        # We tokenise as a second step (rather than ``tokenize=True``)
        # so we can attach the attention mask cleanly for generate().
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        return encoded.input_ids.to(self.device), encoded.attention_mask.to(self.device)

    @torch.no_grad()
    def generate_classification(self, messages: Sequence[Dict[str, str]]) -> str:
        """Run a single classification turn and return the model's text reply."""
        if not messages:
            return ""

        input_ids, attention_mask = self._apply_chat_template(messages)
        gen = self.config.generation

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": gen.max_new_tokens,
            "do_sample": gen.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "attention_mask": attention_mask,
        }
        if gen.do_sample:
            gen_kwargs["temperature"] = gen.temperature
            gen_kwargs["top_p"] = gen.top_p

        output = self.model.generate(input_ids=input_ids, **gen_kwargs)
        new_tokens = output[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


__all__ = [
    "DEFAULT_MODEL_NAME",
    "DEFAULT_LOCAL_DIR",
    "GenerationConfig",
    "QwenConfig",
    "QwenModel",
]
