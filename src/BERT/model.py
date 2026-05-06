"""BERT multi-label classifier for GuardChat Task 1.

Wraps HuggingFace's :class:`AutoModelForSequenceClassification` with
``problem_type="multi_label_classification"``, which switches the head
to BCE-with-logits over six independent sigmoid outputs - matching the
schema produced by SafeGuider's :class:`MultiLabelClassifier` and the
PyTorch BiLSTM head.

The reference Colab notebook (vendors/BERT/mental_bert.py) used a
single-label cross-entropy head over four sentiment classes; here we
keep the encoder backbone (default ``bert-base-uncased``, matching the
paper's BERT citation) and swap the head for multi-label classification
over the six GuardChat NSFW categories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.utils import NUM_CATEGORIES

# Quiet the "Some weights of BertForSequenceClassification ... newly
# initialized" warning that fires every time we attach a fresh head.
try:
    from transformers import logging as _hf_logging
    _hf_logging.set_verbosity_error()
except Exception:  # pragma: no cover - transformers is a hard dep at runtime
    pass


DEFAULT_MODEL_NAME = "bert-base-uncased"


@dataclass
class BERTConfig:
    """Architectural settings for :class:`BERTClassifier`."""

    model_name: str = DEFAULT_MODEL_NAME
    num_classes: int = NUM_CATEGORIES
    max_length: int = 256


class BERTClassifier(nn.Module):
    """Thin wrapper exposing the same ``forward(...) -> (logits, probs)``
    contract as the SafeGuider / BiLSTM heads, so all three plug into
    the shared :class:`RecognitionPipeline` without branching.
    """

    def __init__(
        self,
        config: BERTConfig = BERTConfig(),
        model: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.config = config
        if model is None:
            from transformers import AutoModelForSequenceClassification
            model = AutoModelForSequenceClassification.from_pretrained(
                config.model_name,
                num_labels=config.num_classes,
                problem_type="multi_label_classification",
                ignore_mismatched_sizes=True,
            )
        self.model = model

    # ----------------------- Tokenizer accessor ---------------------- #

    @staticmethod
    def make_tokenizer(model_name: str = DEFAULT_MODEL_NAME):
        """Load the matching tokenizer. Kept as a static helper so the
        pipeline can build a tokenizer without first instantiating the
        whole model on GPU.
        """
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(
            model_name, clean_up_tokenization_spaces=True,
        )

    # --------------------------- Forward ----------------------------- #

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits, probs)``. If ``labels`` is given, the
        underlying HF model also computes BCE-with-logits internally;
        we pull the loss out via :meth:`loss_from_outputs`.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        logits = outputs.logits
        probs = torch.sigmoid(logits)
        return logits, probs

    def forward_with_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Convenience for training: returns the HF model output object
        directly so callers can read ``.loss`` and ``.logits``.
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(probs, multi_label_pred, binary_pred)``."""
        was_training = self.training
        self.eval()
        try:
            _, probs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
            multi = (probs >= threshold).long()
            binary = (multi.sum(dim=-1) > 0).long()
        finally:
            if was_training:
                self.train()
        return probs, multi, binary

    # --------------------- Save / load ------------------------------- #

    def save_pretrained(self, save_dir: str, safe_serialization: bool = False) -> None:
        # safetensors needs contiguous tensors; some HF backbones produce
        # non-contiguous weights after training. Default to the legacy
        # ``pytorch_model.bin`` writer to avoid that footgun.
        self.model.save_pretrained(save_dir, safe_serialization=safe_serialization)

    @classmethod
    def from_pretrained(
        cls,
        save_dir: str,
        num_classes: int = NUM_CATEGORIES,
        max_length: int = 256,
    ) -> "BERTClassifier":
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(
            save_dir,
            num_labels=num_classes,
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True,
        )
        cfg = BERTConfig(
            model_name=save_dir,
            num_classes=num_classes,
            max_length=max_length,
        )
        return cls(config=cfg, model=model)
