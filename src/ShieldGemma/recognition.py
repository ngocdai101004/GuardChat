"""Task 1 zero-shot pipeline using ShieldGemma-2B.

Mirrors the public surface of :mod:`src.LlamaGuard.recognition` so the
benchmark can swap backbones transparently::

    pipe = RecognitionPipeline.from_pretrained(weights, mode="guardchat")
    preds = pipe.predict_samples(samples, kind="conversation")

The model is **zero-shot** - there is no ``RecognitionTrainer``.

Three input representations are supported (``kind``):

* ``prompt`` - the enhanced adversarial prompt ($X_{single}$ in the
  paper; the ``prompt`` column of the GuardChat test set).
* ``raw_prompt`` - the original un-enhanced seed prompt harvested from
  the source dataset (``raw_prompt`` column). Same sample, far less
  adversarial rewriting; the gap between the two runs isolates how much
  of the attack comes from the enhancement step.
* ``conversation`` - the multi-turn dialogue concatenated into one
  string ($X_{conv}$).

Each prediction keeps the raw per-policy ``P(Yes)`` probabilities, so
thresholds and policy-to-category mappings can be revisited later
without re-running the model.
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
    ShieldGemmaConfig,
    ShieldGemmaModel,
    build_scoring_prompt,
)
from .taxonomy import (
    MODES,
    policies_for_mode,
    policy_map_for_mode,
    scores_to_guardchat_vector,
    unreachable_categories,
)


# -------------------------- Prediction record ----------------------- #

@dataclass
class RecognitionPrediction:
    sample_id: str
    text_kind: str
    text: str
    policy_scores: Dict[str, float]
    multi_label: List[int]
    binary_pred: int
    label_names: List[str]
    threshold: float
    label_vector_true: Optional[List[int]] = None
    gold_category: Optional[str] = None
    skipped: bool = False

    @property
    def max_score(self) -> float:
        return max(self.policy_scores.values()) if self.policy_scores else 0.0

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "sample_id": self.sample_id,
            "text_kind": self.text_kind,
            "text": self.text,
            # Raw P(Yes) per policy - the re-thresholdable payload.
            "policy_scores": {k: float(v) for k, v in self.policy_scores.items()},
            "policy_flags": {
                k: int(float(v) >= self.threshold)
                for k, v in self.policy_scores.items()
            },
            "max_score": float(self.max_score),
            "threshold": float(self.threshold),
            "multi_label": {c: int(v) for c, v in zip(CATEGORIES, self.multi_label)},
            "predicted_categories": list(self.label_names),
            "binary_pred": int(self.binary_pred),
        }
        if self.skipped:
            # Empty input (e.g. a missing raw_prompt): scored as safe
            # without a forward pass, flagged so it can be excluded later.
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
    """Zero-shot ShieldGemma recognizer with the GuardChat output schema."""

    def __init__(
        self,
        model: ShieldGemmaModel,
        mode: str = "guardchat",
        threshold: float = 0.5,
        role_prefix: bool = True,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.model = model
        self.mode = mode
        self.threshold = float(threshold)
        self.role_prefix = bool(role_prefix)
        self.policies = policies_for_mode(mode)
        self.policy_map = policy_map_for_mode(mode)

    @classmethod
    def from_pretrained(
        cls,
        weights: str = DEFAULT_LOCAL_DIR,
        mode: str = "guardchat",
        device: Optional[str] = None,
        dtype: str = "auto",
        threshold: float = 0.5,
        token: Optional[str] = None,
        batch_size: int = 4,
        max_length: int = 4096,
        role_prefix: bool = True,
        auto_download: bool = True,
    ) -> "RecognitionPipeline":
        cfg = ShieldGemmaConfig(
            model_path=weights,
            dtype=dtype,
            device=device,
            token=resolve_hf_token(token),
            batch_size=batch_size,
            max_length=max_length,
            auto_download=auto_download,
        )
        model = ShieldGemmaModel(cfg)
        return cls(model=model, mode=mode, threshold=threshold, role_prefix=role_prefix)

    @property
    def unreachable_categories(self) -> List[str]:
        """GuardChat categories the active policy set can never fire."""
        return unreachable_categories(self.mode)

    # -------------------------- Inference ------------------------- #

    def predict_text(self, text: str) -> Dict[str, float]:
        """Score one string against every policy of the active mode."""
        keys = list(self.policies.keys())
        prompts = [build_scoring_prompt(text, self.policies[k]) for k in keys]
        scores = self.model.score_prompts(prompts)
        return dict(zip(keys, scores))

    def _predict_one(self, sample: GuardChatSample, kind: str) -> RecognitionPrediction:
        k = normalise_text_kind(kind)
        text = text_for_kind(sample, k, role_prefix=self.role_prefix)

        if not text.strip():
            scores: Dict[str, float] = {key: 0.0 for key in self.policies}
            multi = [0] * NUM_CATEGORIES
            skipped = True
        else:
            scores = self.predict_text(text)
            multi = scores_to_guardchat_vector(
                scores, self.policy_map, threshold=self.threshold
            )
            skipped = False

        return RecognitionPrediction(
            sample_id=sample.sample_id,
            text_kind=k,
            text=text,
            policy_scores=scores,
            multi_label=multi,
            binary_pred=int(any(v == 1 for v in multi)),
            label_names=[c for c, v in zip(CATEGORIES, multi) if v == 1],
            threshold=self.threshold,
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
    ) -> List[RecognitionPrediction]:
        """Score every sample under one input representation.

        ``on_prediction`` is invoked after each sample - the CLI uses it
        to append to a resumable checkpoint file.
        """
        iterator: Any = samples
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(samples, desc=f"ShieldGemma[{normalise_text_kind(kind)}]")
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
    "TEXT_KINDS",
    "RecognitionPipeline",
    "RecognitionPrediction",
    "DEFAULT_MODEL_NAME",
]
