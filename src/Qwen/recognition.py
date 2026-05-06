"""Task 1 zero-shot pipeline using Qwen2.5-7B-Instruct.

Mirrors the public surface of :mod:`src.SafeGuider.recognition` /
:mod:`src.BiLSTM.recognition` / :mod:`src.BERT.recognition` /
:mod:`src.LlamaGuard.recognition` so benchmarks can swap baselines
without touching call sites:

    pipe = RecognitionPipeline.from_pretrained(weights_dir)
    preds = pipe.predict_samples(samples, kind="conversation")

The model is **zero-shot** - there is no ``RecognitionTrainer``. The
classifier is configured purely through the system prompt in
:mod:`src.Qwen.prompts`, which lists each GuardChat category, asks the
model to mark all that apply, and pins the response format to a strict
6-key JSON object. Robustness against minor format drift (markdown
fences, key spelling variants) is handled by
:func:`prompts.parse_response`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.utils import (
    CATEGORIES,
    NUM_CATEGORIES,
    GuardChatSample,
)

from .model import (
    DEFAULT_LOCAL_DIR,
    DEFAULT_MODEL_NAME,
    QwenConfig,
    QwenModel,
)
from .prompts import (
    SYSTEM_PROMPT,
    build_chat_messages,
    parse_response,
)


# -------------------------- Prediction record ----------------------- #

@dataclass
class RecognitionPrediction:
    sample_id: str
    text: str
    multi_label: List[int]
    binary_pred: int
    label_names: List[str]
    raw_response: str
    parse_ok: bool
    label_vector_true: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "sample_id": self.sample_id,
            "text": self.text,
            # Qwen produces hard 0/1 decisions, not calibrated
            # probabilities, so ``probs`` is omitted (matches
            # LlamaGuard's record schema).
            "multi_label": {c: int(v) for c, v in zip(CATEGORIES, self.multi_label)},
            "predicted_categories": self.label_names,
            "binary_pred": int(self.binary_pred),
            "raw_response": self.raw_response,
            "parse_ok": bool(self.parse_ok),
        }
        if self.label_vector_true is not None:
            out["label_vector_true"] = {
                c: int(v) for c, v in zip(CATEGORIES, self.label_vector_true)
            }
        return out


# ----------------------------- Pipeline ----------------------------- #

def _sample_text(sample: GuardChatSample, kind: str) -> str:
    """Pick the input string the prompt template will see.

    For ``kind='single'`` we feed the enhanced prompt verbatim. For
    ``kind='conversation'`` we feed the flattened multi-turn dialogue
    (``role: content`` per line) so the system prompt can reason over
    the entire conversation as a single classification target rather
    than continuing it.
    """
    if kind == "single":
        return sample.enhanced_prompt or ""
    if kind == "conversation":
        text = sample.conversation_text()
        return text if text else (sample.enhanced_prompt or "")
    raise ValueError(f"kind must be 'single' or 'conversation', got {kind!r}")


class RecognitionPipeline:
    """Zero-shot Qwen recogniser with the GuardChat output schema."""

    def __init__(
        self,
        model: QwenModel,
        system_prompt: str = SYSTEM_PROMPT,
        threshold: float = 0.5,  # accepted for API symmetry; unused.
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.threshold = float(threshold)

    @classmethod
    def from_pretrained(
        cls,
        weights: str = DEFAULT_LOCAL_DIR,
        device: Optional[str] = None,
        dtype: str = "bfloat16",
        max_new_tokens: int = 64,
        do_sample: bool = False,
        system_prompt: Optional[str] = None,
        threshold: float = 0.5,
    ) -> "RecognitionPipeline":
        """Load Qwen2.5-7B-Instruct from a local snapshot folder (or HF id)."""
        from .model import GenerationConfig
        cfg = QwenConfig(
            model_path=weights,
            dtype=dtype,
            device=device,
            generation=GenerationConfig(
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
            ),
        )
        model = QwenModel(cfg)
        return cls(
            model=model,
            system_prompt=system_prompt or SYSTEM_PROMPT,
            threshold=threshold,
        )

    # -------------------------- Inference ------------------------- #

    def _classify_one(
        self,
        sample: GuardChatSample,
        kind: str,
    ) -> Tuple[List[int], int, str, bool, str]:
        """Return ``(multi_label, binary, raw, parse_ok, text_for_record)``."""
        text = _sample_text(sample, kind=kind)
        messages = build_chat_messages(text, kind=kind, system_prompt=self.system_prompt)
        raw = self.model.generate_classification(messages)
        vec, ok, _ = parse_response(raw, on_error="all_zeros")
        binary = int(any(v == 1 for v in vec))
        return vec, binary, raw, ok, text

    def predict_samples(
        self,
        samples: Sequence[GuardChatSample],
        kind: str = "single",
        batch_size: int = 1,   # accepted for API symmetry; ignored.
    ) -> List[RecognitionPrediction]:
        results: List[RecognitionPrediction] = []
        for s in samples:
            vec, binary, raw, ok, text = self._classify_one(s, kind=kind)
            results.append(RecognitionPrediction(
                sample_id=s.sample_id,
                text=text,
                multi_label=vec,
                binary_pred=binary,
                label_names=[c for c, v in zip(CATEGORIES, vec) if v == 1],
                raw_response=raw,
                parse_ok=ok,
                label_vector_true=list(s.label_vector),
            ))
        return results


__all__ = [
    "RecognitionPipeline",
    "RecognitionPrediction",
]
