"""Task 1 zero-shot pipeline using Llama-Guard-3-8B.

Mirrors the public surface of :mod:`src.ShieldGemma.recognition` and the
supervised baselines so benchmarking code can swap the backbone
transparently::

    pipe = RecognitionPipeline.from_pretrained(weights, mode="guardchat")
    preds = pipe.predict_samples(samples, kind="conversation")

The model is **zero-shot** - there is no ``RecognitionTrainer``.

Input representations (``kind``)
--------------------------------
* ``prompt`` - the enhanced adversarial prompt ($X_{single}$), wrapped
  in a single user turn.
* ``raw_prompt`` - the original un-enhanced seed prompt from the source
  dataset. The gap against ``prompt`` isolates how much of the attack
  comes from the enhancement step.
* ``conversation`` - the multi-turn dialogue ($X_{conv}$). Two encodings
  are available, see ``conv_format`` below.

Taxonomy modes (``mode``)
-------------------------
* ``guardchat`` (default) - pass GuardChat's six categories as a custom
  taxonomy (S1=Sexual ... S6=Harassment) into the chat template. The
  model reasons zero-shot over the GuardChat schema directly, so the
  class counts match one-to-one and ``shocking`` is reachable.
* ``native`` - use Llama-Guard-3's own S1-S14 hazard taxonomy, then map
  each S-code to a GuardChat category via
  :data:`taxonomy.SCODE_TO_GUARDCHAT`. This is the model exactly as
  released, but the taxonomy has no analogue for ``shocking``, so that
  category can never fire.

Each prediction keeps the raw generated verdict and the S-codes, so the
S-code -> category mapping can be revisited offline without re-running
the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.utils import (
    CATEGORIES,
    NUM_CATEGORIES,
    TEXT_KINDS,
    GuardChatSample,
    gold_category,
    normalise_text_kind,
    resolve_hf_token,
    text_for_kind,
)

from .model import (
    DEFAULT_LOCAL_DIR,
    DEFAULT_MODEL_NAME,
    LlamaGuardConfig,
    LlamaGuardModel,
)
from .taxonomy import (
    GUARDCHAT_CUSTOM_CATEGORIES,
    MODES,
    normalise_mode,
    scode_map_for_mode,
    scodes_to_guardchat_vector,
    unreachable_categories,
)


# How the multi-turn dialogue is handed to the model:
#
#   turns   forward the real turn list, letting Llama-Guard moderate the
#           last user message in the context of the preceding ones. This
#           is the model's training distribution and its default here.
#   concat  flatten the dialogue into one user message, matching how the
#           supervised baselines and ShieldGemma see $X_{conv}$. Use it
#           when Table 1 needs an apples-to-apples input across rows.
CONV_FORMATS = ("turns", "concat")


# -------------------------- Prediction record ----------------------- #

@dataclass
class RecognitionPrediction:
    sample_id: str
    text_kind: str
    text: str
    multi_label: List[int]
    binary_pred: int
    label_names: List[str]
    raw_response: str
    scodes: List[str]
    label_vector_true: Optional[List[int]] = None
    gold_category: Optional[str] = None
    skipped: bool = False

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "sample_id": self.sample_id,
            "text_kind": self.text_kind,
            "text": self.text,
            # ``probs`` is intentionally absent: Llama-Guard emits
            # discrete S-codes, not category probabilities. ``scodes`` +
            # ``raw_response`` are what make the mapping re-derivable.
            "scodes": list(self.scodes),
            "raw_response": self.raw_response,
            "multi_label": {c: int(v) for c, v in zip(CATEGORIES, self.multi_label)},
            "predicted_categories": list(self.label_names),
            "binary_pred": int(self.binary_pred),
        }
        if self.skipped:
            # Empty input (e.g. a missing raw_prompt): scored as safe
            # without a generation call, flagged so it can be excluded.
            out["skipped_empty_input"] = True
        if self.label_vector_true is not None:
            out["label_vector_true"] = {
                c: int(v) for c, v in zip(CATEGORIES, self.label_vector_true)
            }
            out["true_categories"] = [
                c for c, v in zip(CATEGORIES, self.label_vector_true) if v == 1
            ]
        if self.gold_category is not None:
            out["gold_category"] = self.gold_category
        return out


# ----------------------------- Pipeline ----------------------------- #

class RecognitionPipeline:
    """Zero-shot Llama-Guard recognizer with the GuardChat output schema."""

    def __init__(
        self,
        model: LlamaGuardModel,
        mode: str = "guardchat",
        threshold: float = 0.5,  # kept for API symmetry; unused by Llama-Guard.
        role_prefix: bool = True,
        conv_format: str = "turns",
    ) -> None:
        if conv_format not in CONV_FORMATS:
            raise ValueError(
                f"conv_format must be one of {CONV_FORMATS}, got {conv_format!r}"
            )
        self.model = model
        self.mode = normalise_mode(mode)
        self.threshold = float(threshold)
        self.role_prefix = bool(role_prefix)
        self.conv_format = conv_format
        self._scode_map = scode_map_for_mode(self.mode)

    @classmethod
    def from_pretrained(
        cls,
        weights: str = DEFAULT_LOCAL_DIR,
        mode: str = "guardchat",
        device: Optional[str] = None,
        dtype: str = "auto",
        threshold: float = 0.5,
        token: Optional[str] = None,
        role_prefix: bool = True,
        conv_format: str = "turns",
        auto_download: bool = True,
    ) -> "RecognitionPipeline":
        """Load Llama-Guard from a local snapshot folder (or HF id)."""
        mode = normalise_mode(mode)
        cfg = LlamaGuardConfig(
            model_path=weights,
            dtype=dtype,
            device=device,
            token=resolve_hf_token(token),
            custom_categories=(
                dict(GUARDCHAT_CUSTOM_CATEGORIES) if mode == "guardchat" else None
            ),
            auto_download=auto_download,
        )
        model = LlamaGuardModel(cfg)
        return cls(model=model, mode=mode, threshold=threshold,
                   role_prefix=role_prefix, conv_format=conv_format)

    @property
    def unreachable_categories(self) -> List[str]:
        """GuardChat categories the active taxonomy can never fire."""
        return unreachable_categories(self.mode)

    # -------------------------- Inference ------------------------- #

    def _build_chat(self, sample: GuardChatSample, kind: str, text: str):
        """Turn one sample into the chat list Llama-Guard expects."""
        if kind == "conversation" and self.conv_format == "turns":
            chat: List[Dict[str, str]] = []
            for turn in sample.conversation:
                role = str(turn.get("role", "user")).strip() or "user"
                content = str(turn.get("content", "")).strip()
                if not content:
                    continue
                chat.append({"role": role, "content": content})
            if chat:
                return chat
            # Fall through to the flattened text for dialogue-less rows.
        return [{"role": "user", "content": text}]

    def _predict_one(self, sample: GuardChatSample, kind: str) -> RecognitionPrediction:
        k = normalise_text_kind(kind)
        text = text_for_kind(sample, k, role_prefix=self.role_prefix)

        if not text.strip():
            multi = [0] * NUM_CATEGORIES
            scodes: List[str] = []
            raw = ""
            skipped = True
        else:
            chat = self._build_chat(sample, k, text)
            is_unsafe, scodes, raw = self.model.classify_chat(chat)
            multi = (
                scodes_to_guardchat_vector(scodes, scode_map=self._scode_map)
                if is_unsafe else [0] * NUM_CATEGORIES
            )
            skipped = False

        return RecognitionPrediction(
            sample_id=sample.sample_id,
            text_kind=k,
            text=text,
            multi_label=multi,
            binary_pred=int(any(v == 1 for v in multi)),
            label_names=[c for c, v in zip(CATEGORIES, multi) if v == 1],
            raw_response=raw,
            scodes=list(scodes),
            label_vector_true=list(sample.label_vector),
            gold_category=gold_category(sample),
            skipped=skipped,
        )

    def predict_samples(
        self,
        samples: Sequence[GuardChatSample],
        kind: str = "prompt",
        on_prediction: Optional[Callable[[RecognitionPrediction], None]] = None,
        progress: bool = True,
        batch_size: int = 1,   # accepted for API symmetry; ignored.
    ) -> List[RecognitionPrediction]:
        """Score every sample under one input representation.

        ``on_prediction`` is invoked after each sample - the CLI uses it
        to append to a resumable checkpoint file.
        """
        iterator: Any = samples
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(samples, desc=f"LlamaGuard[{normalise_text_kind(kind)}]")
            except ImportError:
                iterator = samples

        results: List[RecognitionPrediction] = []
        for s in iterator:
            pred = self._predict_one(s, kind=kind)
            results.append(pred)
            if on_prediction is not None:
                on_prediction(pred)
        return results


__all__ = [
    "CONV_FORMATS",
    "MODES",
    "TEXT_KINDS",
    "RecognitionPipeline",
    "RecognitionPrediction",
    "DEFAULT_MODEL_NAME",
]
