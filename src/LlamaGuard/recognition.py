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
* ``conversation`` - the multi-turn dialogue ($X_{conv}$), flattened
  into one message by default; see ``conv_format`` below.

Taxonomy modes (``mode``)
-------------------------
* ``guardchat`` (default) - swap the S1-S14 block for GuardChat's six
  categories (S1=Sexual ... S6=Harassment). The shipped chat template
  hardcodes its own list, so the prompt is rendered by
  :func:`model.build_custom_prompt` instead. The model then reasons
  zero-shot over the GuardChat schema, class counts match one-to-one,
  and ``shocking`` is reachable.
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

from dataclasses import dataclass, field
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
    scode_label_table,
    scode_map_for_mode,
    scodes_to_guardchat_vector,
    unreachable_categories,
)


# How the multi-turn dialogue is handed to the model:
#
#   concat  flatten the dialogue into one user message, matching how the
#           supervised baselines and ShieldGemma see $X_{conv}$. Default,
#           so Table 1 compares like with like.
#   turns   forward the real turn list. Llama-Guard's chat template
#           hard-requires strictly alternating user/assistant roles and
#           raises otherwise, so consecutive same-role turns are merged
#           into one message first. GuardChat dialogues are entirely
#           user-side, which means this collapses to a single user
#           message - i.e. the same content as `concat`, minus the
#           'user: ' role prefixes.
CONV_FORMATS = ("concat", "turns")


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
    scode_labels: List[str] = field(default_factory=list)
    unmapped_scodes: List[str] = field(default_factory=list)
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
            # What the model actually answered, before any GuardChat
            # remapping: the raw codes, their names under the taxonomy
            # the model was shown, and the undecoded verdict string.
            "scodes": list(self.scodes),
            "scode_labels": list(self.scode_labels),
            "raw_response": self.raw_response,
            "multi_label": {c: int(v) for c, v in zip(CATEGORIES, self.multi_label)},
            "predicted_categories": list(self.label_names),
            "binary_pred": int(self.binary_pred),
        }
        if self.unmapped_scodes:
            # The model said "unsafe" but with a code the active taxonomy
            # cannot place - e.g. a leftover S7-S14 while running in
            # guardchat mode. Such a row silently ends up all-zero, i.e.
            # counted as safe, so record it rather than lose it.
            out["unmapped_scodes"] = list(self.unmapped_scodes)
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
        conv_format: str = "concat",
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
        self._scode_labels = scode_label_table(self.mode)

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
        conv_format: str = "concat",
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
        """Turn one sample into the chat list Llama-Guard expects.

        The chat template refuses anything but strictly alternating
        user/assistant turns, so consecutive same-role messages are
        merged rather than passed through - otherwise every GuardChat
        row (all user turns) would raise a Jinja ``TemplateError``.
        """
        if kind == "conversation" and self.conv_format == "turns":
            chat: List[Dict[str, str]] = []
            for turn in sample.conversation:
                role = str(turn.get("role", "user")).strip() or "user"
                content = str(turn.get("content", "")).strip()
                if not content:
                    continue
                if chat and chat[-1]["role"] == role:
                    chat[-1]["content"] += "\n" + content
                else:
                    chat.append({"role": role, "content": content})
            if chat and chat[0]["role"] != "user":
                # The template assumes turn 0 is the user; prepend an
                # empty one rather than let it mislabel every speaker.
                chat.insert(0, {"role": "user", "content": ""})
            if chat:
                return chat
            # Fall through to the flattened text for dialogue-less rows.
        return [{"role": "user", "content": text}]

    def _predict_one(self, sample: GuardChatSample, kind: str) -> RecognitionPrediction:
        k = normalise_text_kind(kind)
        text = text_for_kind(sample, k, role_prefix=self.role_prefix)

        unmapped: List[str] = []
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
            if is_unsafe:
                unmapped = [c for c in scodes if c.upper() not in self._scode_map]
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
            scode_labels=[self._scode_labels.get(c.upper(), "?") for c in scodes],
            unmapped_scodes=unmapped,
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
