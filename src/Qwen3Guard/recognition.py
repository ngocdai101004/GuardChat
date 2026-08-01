"""Task 1 zero-shot pipeline using Qwen3Guard-Gen-8B.

Mirrors the public surface of :mod:`src.LlamaGuard.recognition` and
:mod:`src.ShieldGemma.recognition` so benchmarking code can swap the
backbone transparently::

    pipe = RecognitionPipeline.from_pretrained(weights, mode="guardchat")
    preds = pipe.predict_samples(samples, kind="conversation")

The model is **zero-shot** - there is no ``RecognitionTrainer``.

Input representations (``kind``)
--------------------------------
* ``prompt`` - the enhanced adversarial prompt ($X_{single}$), wrapped in
  a single user turn.
* ``raw_prompt`` - the original un-enhanced seed prompt from the source
  dataset. The gap against ``prompt`` isolates how much of the attack
  comes from the enhancement step.
* ``conversation`` - the multi-turn dialogue ($X_{conv}$), flattened into
  one message by default; see ``conv_format`` below.

Taxonomy modes (``mode``)
-------------------------
* ``guardchat`` (default) - swap the shipped nine-category block for
  GuardChat's six. The chat template hardcodes its own list, so the
  prompt is rendered by :func:`model.build_custom_prompt` instead. Class
  counts then match one-to-one and ``shocking`` is reachable.
* ``native`` - Qwen3Guard's own nine categories, mapped back to GuardChat
  via :data:`taxonomy.NATIVE_TO_GUARDCHAT`. This is the model exactly as
  released, but two of its categories (``Politically Sensitive Topics``,
  ``Jailbreak``) have no GuardChat analogue and ``shocking`` can never
  fire.

Severity
--------
Qwen3Guard grades ``Safe`` / ``Controversial`` / ``Unsafe``.
``controversial_as_unsafe`` (default ``True``) decides how the middle
level collapses into GuardChat's binary verdict. Every prediction stores
the raw severity string, so the other reading is recomputable offline
without re-running the model - as is the whole category mapping, since
the verbatim category names and the undecoded response are stored too.
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
    Qwen3GuardConfig,
    Qwen3GuardModel,
)
from .taxonomy import (
    GUARDCHAT_CATEGORIES,
    MODES,
    SEVERITY_LABELS,
    categories_to_guardchat_vector,
    category_map_for_mode,
    normalise_mode,
    severity_is_unsafe,
    unreachable_categories,
)


# How the multi-turn dialogue is handed to the model:
#
#   concat  flatten the dialogue into one user message, matching how the
#           supervised baselines and ShieldGemma see $X_{conv}$. Default,
#           so Table 1 compares like with like.
#   turns   forward the real turn list. Unlike Llama-Guard's template,
#           Qwen3Guard's tolerates consecutive same-role turns - it simply
#           emits another "USER: ..." block - so no role merging is
#           needed. GuardChat dialogues are entirely user-side, which
#           makes this the same content as `concat` with the model's own
#           "USER: " prefixes instead of ours.
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
    severity: str
    raw_categories: List[str] = field(default_factory=list)
    unmapped_categories: List[str] = field(default_factory=list)
    verdict_unsafe: bool = False
    controversial_as_unsafe: bool = True
    label_vector_true: Optional[List[int]] = None
    gold_category: Optional[str] = None
    skipped: bool = False

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "sample_id": self.sample_id,
            "text_kind": self.text_kind,
            "text": self.text,
            # ---- what the model actually answered ----
            # ``probs`` is intentionally absent: Qwen3Guard-Gen writes a
            # discrete verdict, not category probabilities. The severity,
            # the verbatim category names and the undecoded response are
            # what make every downstream number re-derivable.
            "severity": SEVERITY_LABELS.get(self.severity, self.severity),
            "raw_categories": list(self.raw_categories),
            "raw_response": self.raw_response,
            # How the three-level severity was collapsed to binary.
            "verdict_unsafe": int(self.verdict_unsafe),
            "controversial_as_unsafe": bool(self.controversial_as_unsafe),
            # ---- mapped onto the six GuardChat categories ----
            "multi_label": {c: int(v) for c, v in zip(CATEGORIES, self.multi_label)},
            "predicted_categories": list(self.label_names),
            "binary_pred": int(self.binary_pred),
        }
        if self.unmapped_categories:
            # The model flagged the sample but named a category the active
            # taxonomy cannot place - `Jailbreak` above all, which is the
            # form of the attack rather than the harm it depicts. Such a
            # row collapses to an all-zero vector, i.e. it is scored as
            # safe, so record it rather than lose it.
            out["unmapped_categories"] = list(self.unmapped_categories)
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
    """Zero-shot Qwen3Guard recognizer with the GuardChat output schema."""

    def __init__(
        self,
        model: Qwen3GuardModel,
        mode: str = "guardchat",
        threshold: float = 0.5,  # kept for API symmetry; unused by Qwen3Guard.
        role_prefix: bool = True,
        conv_format: str = "concat",
        controversial_as_unsafe: bool = True,
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
        self.controversial_as_unsafe = bool(controversial_as_unsafe)
        self._category_map = category_map_for_mode(self.mode)

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
        controversial_as_unsafe: bool = True,
        auto_download: bool = True,
    ) -> "RecognitionPipeline":
        """Load Qwen3Guard from a local snapshot folder (or HF id)."""
        mode = normalise_mode(mode)
        cfg = Qwen3GuardConfig(
            model_path=weights,
            dtype=dtype,
            device=device,
            token=resolve_hf_token(token),
            custom_categories=(
                dict(GUARDCHAT_CATEGORIES) if mode == "guardchat" else None
            ),
            auto_download=auto_download,
        )
        model = Qwen3GuardModel(cfg)
        return cls(model=model, mode=mode, threshold=threshold,
                   role_prefix=role_prefix, conv_format=conv_format,
                   controversial_as_unsafe=controversial_as_unsafe)

    @property
    def unreachable_categories(self) -> List[str]:
        """GuardChat categories the active taxonomy can never fire."""
        return unreachable_categories(self.mode)

    # -------------------------- Inference ------------------------- #

    def _build_chat(self, sample: GuardChatSample, kind: str, text: str):
        """Turn one sample into the chat list Qwen3Guard expects.

        The template grades whichever role speaks last, so the chat must
        end on a user turn for Task 1 (input filtering). GuardChat
        dialogues are entirely user-side, so this holds naturally; a
        trailing assistant turn would switch the model to
        response-moderation, which is a different task.
        """
        if kind == "conversation" and self.conv_format == "turns":
            chat: List[Dict[str, str]] = []
            for turn in sample.conversation:
                role = str(turn.get("role", "user")).strip() or "user"
                content = str(turn.get("content", "")).strip()
                if not content:
                    continue
                chat.append({"role": role, "content": content})
            if chat and chat[-1]["role"] == "user":
                return chat
            # Empty dialogue, or one ending on an assistant turn: fall
            # back to the flattened text so we keep grading the input.
        return [{"role": "user", "content": text}]

    def _predict_one(self, sample: GuardChatSample, kind: str) -> RecognitionPrediction:
        k = normalise_text_kind(kind)
        text = text_for_kind(sample, k, role_prefix=self.role_prefix)

        unmapped: List[str] = []
        names: List[str] = []
        if not text.strip():
            multi = [0] * NUM_CATEGORIES
            severity = "safe"
            raw = ""
            is_unsafe = False
            skipped = True
        else:
            chat = self._build_chat(sample, k, text)
            severity, names, raw = self.model.classify_chat(chat, mode=self.mode)
            is_unsafe = severity_is_unsafe(
                severity, controversial_as_unsafe=self.controversial_as_unsafe
            )
            multi = (
                categories_to_guardchat_vector(names, category_map=self._category_map)
                if is_unsafe else [0] * NUM_CATEGORIES
            )
            if is_unsafe:
                unmapped = [n for n in names if n not in self._category_map]
            skipped = False

        return RecognitionPrediction(
            sample_id=sample.sample_id,
            text_kind=k,
            text=text,
            multi_label=multi,
            binary_pred=int(any(v == 1 for v in multi)),
            label_names=[c for c, v in zip(CATEGORIES, multi) if v == 1],
            raw_response=raw,
            severity=severity,
            raw_categories=list(names),
            unmapped_categories=unmapped,
            verdict_unsafe=bool(is_unsafe),
            controversial_as_unsafe=self.controversial_as_unsafe,
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
                iterator = tqdm(samples, desc=f"Qwen3Guard[{normalise_text_kind(kind)}]")
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
