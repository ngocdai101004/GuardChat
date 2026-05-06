"""Task 1 zero-shot pipeline using Llama-Guard-3-8B.

Mirrors the public surface of :mod:`src.SafeGuider.recognition` /
:mod:`src.BiLSTM.recognition` / :mod:`src.BERT.recognition` so
benchmarking code can swap the backbone transparently:

    pipe = RecognitionPipeline.from_pretrained(weights_dir, mode="native")
    preds = pipe.predict_samples(samples, kind="conversation")

The model is **zero-shot** - there is no ``RecognitionTrainer``. Two
inference modes are provided:

* ``mode='native'`` (default): use Llama-Guard-3's S1-S14 hazard
  taxonomy, then map each S-code to a GuardChat category via
  :data:`taxonomy.SCODE_TO_GUARDCHAT`. The Llama-Guard taxonomy has no
  analogue for ``shocking`` so this mode never fires that class.

* ``mode='custom'``: pass GuardChat's six categories as a custom
  taxonomy (S1=Sexual, S2=Illegal, S3=Shocking, ..., S6=Harassment) into
  the chat template. The model then reasons zero-shot over the
  GuardChat schema directly. Useful for completeness on ``shocking``,
  but typically weaker on the natively-trained classes.
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
    LlamaGuardConfig,
    LlamaGuardModel,
)
from .taxonomy import (
    CUSTOM_SCODE_TO_GUARDCHAT,
    GUARDCHAT_CUSTOM_CATEGORIES,
    SCODE_TO_GUARDCHAT,
    parse_llamaguard_response,
    scodes_to_guardchat_vector,
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
    scodes: List[str]
    label_vector_true: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "sample_id": self.sample_id,
            "text": self.text,
            # ``probs`` is intentionally absent: Llama-Guard emits
            # discrete S-codes, not category probabilities.
            "multi_label": {c: int(v) for c, v in zip(CATEGORIES, self.multi_label)},
            "predicted_categories": self.label_names,
            "binary_pred": int(self.binary_pred),
            "raw_response": self.raw_response,
            "scodes": list(self.scodes),
        }
        if self.label_vector_true is not None:
            out["label_vector_true"] = {
                c: int(v) for c, v in zip(CATEGORIES, self.label_vector_true)
            }
        return out


# ----------------------------- Pipeline ----------------------------- #

def _sample_to_chat(sample: GuardChatSample, kind: str) -> List[Dict[str, str]]:
    """Convert a :class:`GuardChatSample` into the chat list Llama-Guard expects.

    ``kind='single'`` wraps the enhanced prompt in one user turn.
    ``kind='conversation'`` forwards the multi-turn dialogue verbatim,
    falling back to the single-turn prompt when no conversation is
    attached. Llama-Guard moderates the most recent user message in the
    context of preceding turns - this is the closest match to the
    paper's ``X_conv`` representation while still respecting the
    model's training distribution (real chat-style messages).
    """
    if kind == "single":
        text = sample.enhanced_prompt or ""
        return [{"role": "user", "content": text}]
    if kind == "conversation":
        chat: List[Dict[str, str]] = []
        for turn in sample.conversation:
            role = str(turn.get("role", "user")).strip() or "user"
            content = str(turn.get("content", "")).strip()
            if not content:
                continue
            chat.append({"role": role, "content": content})
        if not chat:
            chat = [{"role": "user", "content": sample.enhanced_prompt or ""}]
        return chat
    raise ValueError(f"kind must be 'single' or 'conversation', got {kind!r}")


class RecognitionPipeline:
    """Zero-shot Llama-Guard recognizer with the GuardChat output schema."""

    def __init__(
        self,
        model: LlamaGuardModel,
        mode: str = "native",
        threshold: float = 0.5,  # kept for API symmetry; unused by Llama-Guard.
    ) -> None:
        if mode not in {"native", "custom"}:
            raise ValueError(f"mode must be 'native' or 'custom', got {mode!r}")
        self.model = model
        self.mode = mode
        self.threshold = float(threshold)
        self._scode_map = (
            CUSTOM_SCODE_TO_GUARDCHAT if mode == "custom" else SCODE_TO_GUARDCHAT
        )

    @classmethod
    def from_pretrained(
        cls,
        weights: str = DEFAULT_LOCAL_DIR,
        mode: str = "native",
        device: Optional[str] = None,
        dtype: str = "bfloat16",
        threshold: float = 0.5,
    ) -> "RecognitionPipeline":
        """Load Llama-Guard from a local snapshot folder (or HF id)."""
        cfg = LlamaGuardConfig(
            model_path=weights,
            dtype=dtype,
            device=device,
            custom_categories=(
                dict(GUARDCHAT_CUSTOM_CATEGORIES) if mode == "custom" else None
            ),
        )
        model = LlamaGuardModel(cfg)
        return cls(model=model, mode=mode, threshold=threshold)

    # -------------------------- Inference ------------------------- #

    def _classify_one(
        self,
        sample: GuardChatSample,
        kind: str,
    ) -> Tuple[List[int], int, List[str], str, str]:
        """Return ``(multi_label, binary, scodes, raw, text_for_record)``."""
        chat = _sample_to_chat(sample, kind=kind)
        is_unsafe, scodes, raw = self.model.classify_chat(chat)
        if is_unsafe:
            multi = scodes_to_guardchat_vector(scodes, scode_map=self._scode_map)
        else:
            multi = [0] * NUM_CATEGORIES

        binary = int(any(v == 1 for v in multi))
        # Always carry a textual representation so downstream JSON has a
        # human-readable record of what the model actually saw.
        text_for_record = (
            sample.enhanced_prompt if kind == "single"
            else sample.conversation_text() or sample.enhanced_prompt
        )
        return multi, binary, scodes, raw, text_for_record

    def predict_samples(
        self,
        samples: Sequence[GuardChatSample],
        kind: str = "single",
        batch_size: int = 1,   # accepted for API symmetry; ignored.
    ) -> List[RecognitionPrediction]:
        results: List[RecognitionPrediction] = []
        for s in samples:
            multi, binary, scodes, raw, text = self._classify_one(s, kind=kind)
            results.append(RecognitionPrediction(
                sample_id=s.sample_id,
                text=text,
                multi_label=multi,
                binary_pred=binary,
                label_names=[c for c, v in zip(CATEGORIES, multi) if v == 1],
                raw_response=raw,
                scodes=scodes,
                label_vector_true=list(s.label_vector),
            ))
        return results


__all__ = [
    "RecognitionPipeline",
    "RecognitionPrediction",
]
