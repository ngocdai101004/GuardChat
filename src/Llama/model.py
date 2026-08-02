"""Llama-3.1-8B-Instruct wrapper for GuardChat Task 2 rewriting.

Loads ``meta-llama/Llama-3.1-8B-Instruct`` from a local folder (default
``src/Llama/weights/Llama-3.1-8B-Instruct``, populated by
:mod:`src.Llama.download_weights`) and exposes one call,
:meth:`LlamaModel.generate_batch`, that turns a list of chat message
lists into a list of decoded replies.

This is the open-source counterpart to the Gemini Task-2 rewriter. It
sees the *same* system prompts and the same ``[Tn]`` turn contract
(:mod:`src.utils.rewrite_prompt`) and serialises to the same record
schema, so the two rows of Table 2 differ only in the model.

Batching
--------
1,000 samples x 2 representations is 2,000 generations, and a
conversation rewrite has to re-emit ~1,500 tokens. Generating one at a
time leaves the GPU almost idle, so requests are batched with **left**
padding - the only correct padding side for decoder-only generation,
since right padding would put pad tokens between the prompt and the
first generated token.

The token budget is adaptive: a rewrite is roughly as long as its
input, so ``max_new_tokens`` is derived per batch from the longest
source text rather than pinned at a worst-case constant. That keeps
short prompts from paying for a 2,500-token generation window.

Memory / dtype
--------------
=============  =============  ===================================
``dtype``      GPU footprint  notes
=============  =============  ===================================
``bfloat16``   ~16 GB         default on an accelerator
``float16``    ~16 GB         if bf16 is unsupported
``float32``    ~32 GB         the CPU default; rarely useful
``int8``       ~9 GB          needs bitsandbytes (CUDA)
``nf4``        ~5 GB          needs bitsandbytes (CUDA); 4-bit NF4
=============  =============  ===================================

Add roughly ``batch_size x sequence_length`` of KV cache on top. Lower
``--batch-size`` before lowering precision.

Tested with ``torch>=2.1``, ``transformers>=4.43`` (Llama 3.1 support),
``accelerate>=0.26``, ``huggingface_hub>=0.23``, and
``bitsandbytes>=0.43`` only when ``dtype in {'int8','nf4'}``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from src.utils import (
    bnb_config,
    default_dtype_for,
    from_pretrained_with_dtype,
    resolve_device,
    resolve_weights_path,
)


DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_LOCAL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "weights", "Llama-3.1-8B-Instruct"
)

# Ceiling on the generation window, per input representation. The
# adaptive estimate below almost always lands well under these; they
# exist so one pathological input cannot stall a batch.
MAX_NEW_TOKENS_CAP: Dict[str, int] = {
    "prompt": 1024,
    "conversation": 2560,
}

# A sanitised rewrite is about as long as its source, plus slack for the
# [Tn] markers and for substitutions that run longer than what they
# replace ("a corpse" -> "a weathered mannequin").
_LENGTH_SLACK = 1.4
_LENGTH_FLOOR = 96


@dataclass
class GenerationConfig:
    """Decoding settings.

    Greedy is the default: a sanitised prompt should be reproducible, and
    it is what the Gemini rewriter uses (``temperature=0``).

    ``sample_temperature`` is only used on *retries*. Re-running a greedy
    generation is bit-for-bit identical, so a retry after an empty or
    unparseable answer is pointless unless the decode is perturbed - see
    :meth:`src.Llama.rewrite.RewritePipeline.rewrite_samples`.
    """

    max_new_tokens: Optional[int] = None    # None = adaptive, capped per kind
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    sample_temperature: float = 0.7
    sample_top_p: float = 0.9


@dataclass
class LlamaConfig:
    model_path: str = DEFAULT_LOCAL_DIR
    dtype: str = "auto"
    device: Optional[str] = None
    token: Optional[str] = None
    batch_size: int = 8
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    # Fetch the snapshot into ``model_path`` when it is missing, instead
    # of leaving ~16 GB in the shared ~/.cache/huggingface tree.
    auto_download: bool = True
    repo_id: str = DEFAULT_MODEL_NAME


class LlamaModel:
    """Loaded Llama-3.1-8B-Instruct plus batched chat generation."""

    def __init__(self, config: LlamaConfig = LlamaConfig()) -> None:
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
            log_prefix="Llama",
        )

        self.device = resolve_device(self.config.device)
        dtype_name = self.config.dtype
        if not dtype_name or dtype_name == "auto":
            dtype_name = default_dtype_for(self.device)
        self.dtype_name = dtype_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            path, clean_up_tokenization_spaces=True, **auth
        )
        # Decoder-only generation must pad on the left, or the model
        # continues from pad tokens rather than from the prompt.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            # Llama 3.1 ships no pad token. Its reserved right-pad id is
            # the intended filler; fall back to EOS on older tokenizers.
            pad = self.tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>")
            if pad is None or pad < 0:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.pad_token_id = pad

        bnb = bnb_config(dtype_name)
        if bnb is not None:
            # device_map="auto" cooperates with bitsandbytes' placement;
            # an explicit .to(device) is unnecessary and would error.
            self.model = AutoModelForCausalLM.from_pretrained(
                path, quantization_config=bnb, device_map="auto", **auth
            )
        else:
            self.model = from_pretrained_with_dtype(
                AutoModelForCausalLM, path, dtype_name,
                device_map={"": str(self.device)}, **auth
            )
        self.model.eval()
        self.device = next(self.model.parameters()).device

    # ------------------------ Prompt rendering ----------------------- #

    def render_chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Apply the shipped Llama 3.1 chat template.

        Unlike the guard models in Task 1, nothing is hand-rolled here:
        the template takes no taxonomy argument, so there is nothing for
        it to silently ignore. Its ``Today Date`` line defaults to a
        fixed string rather than the wall clock, so the rendering is
        stable across runs.
        """
        return self.tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True,
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(str(text), add_special_tokens=False).input_ids)

    def budget_for(self, source_lengths: Sequence[int], kind: str) -> int:
        """``max_new_tokens`` for a batch, from its longest source text."""
        override = self.config.generation.max_new_tokens
        cap = MAX_NEW_TOKENS_CAP.get(kind, MAX_NEW_TOKENS_CAP["prompt"])
        if override:
            return int(override)
        longest = max(source_lengths) if source_lengths else 0
        return int(min(cap, int(longest * _LENGTH_SLACK) + _LENGTH_FLOOR))

    # --------------------------- Generation -------------------------- #

    @torch.no_grad()
    def generate_batch(
        self,
        batch: Sequence[Sequence[Mapping[str, str]]],
        max_new_tokens: int,
        sample: bool = False,
    ) -> List[str]:
        """Generate one reply per chat in ``batch``.

        ``sample=True`` switches to the retry decode (see
        :class:`GenerationConfig`); the default is greedy.
        """
        if not batch:
            return []

        texts = [self.render_chat(m) for m in batch]
        encoded = self.tokenizer(
            texts, return_tensors="pt", padding=True, add_special_tokens=False,
        ).to(self.device)

        gen = self.config.generation
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "pad_token_id": self.tokenizer.pad_token_id,
            "do_sample": bool(sample or gen.do_sample),
        }
        if gen_kwargs["do_sample"]:
            gen_kwargs["temperature"] = (
                gen.sample_temperature if sample else gen.temperature
            )
            gen_kwargs["top_p"] = gen.sample_top_p if sample else gen.top_p

        output = self.model.generate(**encoded, **gen_kwargs)

        # Left padding means every row's prompt ends at the same index,
        # so the continuation starts at the shared input width.
        start = encoded["input_ids"].shape[-1]
        return [
            self.tokenizer.decode(row[start:], skip_special_tokens=True)
            for row in output
        ]


__all__ = [
    "DEFAULT_MODEL_NAME",
    "DEFAULT_LOCAL_DIR",
    "MAX_NEW_TOKENS_CAP",
    "GenerationConfig",
    "LlamaConfig",
    "LlamaModel",
]
