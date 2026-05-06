"""Task 2 rewrite pipeline using Llama-3.1-8B-Instruct.

Mirrors the public surface of :mod:`src.SafeGuider.rewrite` so that
benchmarking code can swap baselines transparently:

    pipe = RewritePipeline.from_pretrained(weights_dir)
    results = pipe.rewrite_samples(samples)

The model is **inference only** - there is no training. Rewriting is
configured entirely by the shared system prompt in
:mod:`src.utils.rewrite_prompt`.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from src.utils import (
    GuardChatSample,
    build_rewrite_messages,
    cleanup_rewrite_response,
)

from .model import (
    DEFAULT_LOCAL_DIR,
    DEFAULT_MODEL_NAME,
    GenerationConfig,
    LlamaConfig,
    LlamaModel,
)


@dataclass
class RewriteResult:
    """One rewrite output, ready for serialisation.

    Schema is aligned with :class:`src.SafeGuider.rewrite.RewriteResult`
    so a shared aggregator can compose Table 2 across baselines.
    Fields specific to SafeGuider's beam search (``removed_tokens``,
    ``original_safety``, ``modified_safety``, ``safeguider_similarity``)
    are intentionally absent here - they have no analogue in an
    LLM-based rewriter.
    """

    sample_id: str
    original_prompt: str
    rewritten_prompt: str
    was_modified: bool
    raw_response: str
    elapsed_sec: float
    label_names: List[str]
    source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "original_prompt": self.original_prompt,
            "rewritten_prompt": self.rewritten_prompt,
            "was_modified": self.was_modified,
            "raw_response": self.raw_response,
            "elapsed_sec": self.elapsed_sec,
            "label_names": self.label_names,
            "source": self.source,
        }


class RewritePipeline:
    """Llama-3.1-8B-Instruct rewriter wired to the GuardChat schema."""

    def __init__(
        self,
        model: LlamaModel,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.model = model
        # ``None`` means "use the shared default in src.utils.rewrite_prompt".
        self.system_prompt = system_prompt

    @classmethod
    def from_pretrained(
        cls,
        weights: str = DEFAULT_LOCAL_DIR,
        device: Optional[str] = None,
        dtype: str = "bfloat16",
        max_new_tokens: int = 200,
        do_sample: bool = False,
        system_prompt: Optional[str] = None,
    ) -> "RewritePipeline":
        cfg = LlamaConfig(
            model_path=weights,
            dtype=dtype,
            device=device,
            generation=GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
            ),
        )
        model = LlamaModel(cfg)
        return cls(model=model, system_prompt=system_prompt)

    # --------------------------- Inference -------------------------- #

    def rewrite_prompt(
        self,
        prompt: str,
        sample_id: str = "0",
        label_names: Optional[List[str]] = None,
        source: Optional[str] = None,
    ) -> RewriteResult:
        t0 = time.time()
        messages = build_rewrite_messages(prompt, system_prompt=self.system_prompt)
        raw = self.model.rewrite(messages)
        cleaned = cleanup_rewrite_response(raw)
        # If cleanup returned an empty string, fall back to a generic
        # safe alternative so downstream T2I pipelines never receive ""
        if not cleaned:
            cleaned = "a serene landscape"
        return RewriteResult(
            sample_id=str(sample_id),
            original_prompt=str(prompt or ""),
            rewritten_prompt=cleaned,
            was_modified=(cleaned.strip().lower() != str(prompt or "").strip().lower()),
            raw_response=raw,
            elapsed_sec=round(time.time() - t0, 4),
            label_names=list(label_names or []),
            source=source,
        )

    def rewrite_samples(
        self,
        samples: Sequence[GuardChatSample],
    ) -> List[RewriteResult]:
        out: List[RewriteResult] = []
        for s in samples:
            out.append(self.rewrite_prompt(
                prompt=s.enhanced_prompt,
                sample_id=str(s.sample_id),
                label_names=s.label_names,
                source=s.source,
            ))
        return out


__all__ = [
    "RewritePipeline",
    "RewriteResult",
]
