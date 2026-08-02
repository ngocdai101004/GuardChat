"""SBERT text encoder (``all-mpnet-base-v2``) for Task-2 semantic similarity.

Why this module exists
----------------------
Task 2 originally scored semantic preservation with CLIP's text tower.
A reviewer objected, correctly: CLIP's text encoder is trained to align
text with *images*, not to place two texts near each other, and its
77-token window truncates most GuardChat inputs. Neither property is what
"did the rewrite keep the meaning?" needs. ``all-mpnet-base-v2`` is
trained on 1B+ text pairs with a contrastive objective over text alone,
which is exactly the text-to-text setting, and reads 384 tokens instead
of 77.

Implemented against plain ``transformers`` rather than the
``sentence-transformers`` package: the model card publishes both paths
and states they are equivalent, and the manual one keeps the dependency
list unchanged. The equivalence rests on three things, all of which this
module reads out of the checkpoint instead of hard-coding, so a silently
different config cannot pass unnoticed:

    1. ``sentence_bert_config.json`` -> ``max_seq_length`` (384)
    2. ``1_Pooling/config.json``     -> mean pooling over the attention mask
    3. ``modules.json``              -> a final L2 ``Normalize`` layer

With unit-norm embeddings, cosine similarity is a dot product.

Truncation is reported, never hidden. GuardChat conversations average
~440 words - comfortably past 384 tokens - so a mean similarity computed
over whole dialogues is partly a comparison of prefixes. That is the same
failure mode the reviewer flagged in CLIP, just further out, and
:mod:`src.SBERT.similarity` answers it by also scoring turn by turn.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOCAL_DIR = os.path.join(_HERE, "weights", "all-mpnet-base-v2")

# Fallback only - the real value is read from sentence_bert_config.json.
# mpnet's positional table allows 512; sentence-transformers ships this
# checkpoint configured at 384, and published scores assume that.
FALLBACK_MAX_SEQ_LENGTH = 384

DEFAULT_BATCH_SIZE = 64

# The Hub repo carries several redundant serialisations of the same
# weights. safetensors plus the tokenizer and the sentence-transformers
# config files are all we load; the rest is ~1.5 GB of nothing.
IGNORE_PATTERNS = [
    "*.bin",            # pytorch_model.bin
    "*.h5", "*.msgpack", "tf_model.h5", "flax_model.msgpack",
    "onnx/*", "openvino/*", "*.onnx",
    "rust_model.ot",
]


def read_max_seq_length(model_path: str) -> Optional[int]:
    """``max_seq_length`` from ``sentence_bert_config.json``, if present.

    Returns ``None`` for a plain transformers checkpoint (no such file),
    which is not an error - it just means the caller has to decide.
    """
    cfg = os.path.join(model_path, "sentence_bert_config.json")
    if not os.path.isfile(cfg):
        return None
    try:
        with open(cfg, "r", encoding="utf-8") as f:
            value = json.load(f).get("max_seq_length")
    except (json.JSONDecodeError, OSError):
        return None
    return int(value) if value else None


def assert_mean_pooling(model_path: str) -> None:
    """Fail loudly if the checkpoint does not use mean pooling.

    This module hard-codes mean pooling. Pointing it at a CLS-pooling
    sentence-transformers model would silently produce embeddings that
    disagree with everyone else's numbers for the same model id, so the
    assumption is checked rather than assumed.
    """
    cfg = os.path.join(model_path, "1_Pooling", "config.json")
    if not os.path.isfile(cfg):
        return  # plain transformers checkpoint - nothing to contradict
    try:
        with open(cfg, "r", encoding="utf-8") as f:
            pooling = json.load(f)
    except (json.JSONDecodeError, OSError):
        return
    if not pooling.get("pooling_mode_mean_tokens", False):
        modes = [k for k, v in pooling.items()
                 if k.startswith("pooling_mode_") and v]
        raise ValueError(
            f"{model_path} is configured for {modes or ['<none>']}, but "
            f"src.SBERT.model implements mean pooling. Use the "
            f"sentence-transformers package for this checkpoint."
        )


def mean_pool(last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
    """Average token vectors over real tokens only.

    Padding must not dilute the mean, hence the mask - this is the step
    people most often get wrong when reimplementing SBERT by hand.
    """
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class SBERTEncoder:
    """Encodes text to unit-norm 768-d sentence embeddings.

    Kept deliberately thin: load, encode, count tokens. Everything that
    knows about GuardChat records lives in :mod:`src.SBERT.similarity`.
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[torch.device] = None,
        max_seq_length: Optional[int] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        assert_mean_pooling(model_path)

        self.model_path = model_path
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.batch_size = int(batch_size)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval().to(self.device)

        if max_seq_length is None:
            max_seq_length = read_max_seq_length(model_path) or FALLBACK_MAX_SEQ_LENGTH
        self.max_seq_length = int(max_seq_length)

    # ------------------------------ Encode --------------------------- #

    @torch.no_grad()
    def encode(
        self,
        texts: Sequence[str],
        batch_size: Optional[int] = None,
        normalize: bool = True,
    ) -> Tensor:
        """Embed ``texts`` -> ``(N, 768)`` on the encoder's device.

        Empty input returns an empty tensor rather than raising: a run
        where every row failed should still produce a summary saying so.
        """
        if not texts:
            return torch.empty(0, self.model.config.hidden_size, device=self.device)

        bs = int(batch_size or self.batch_size)
        chunks: List[Tensor] = []
        for start in range(0, len(texts), bs):
            batch = [str(t) for t in texts[start:start + bs]]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            ).to(self.device)
            out = self.model(**enc)
            emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
            if normalize:
                emb = F.normalize(emb, p=2, dim=1)
            chunks.append(emb)
        return torch.cat(chunks, dim=0)

    # --------------------------- Diagnostics ------------------------- #

    def token_counts(self, texts: Sequence[str]) -> List[int]:
        """Untruncated token length of each text, special tokens included.

        Compared against :attr:`max_seq_length` this is what tells us
        whether a similarity score saw the whole input or a prefix.
        """
        if not texts:
            return []
        enc = self.tokenizer(
            [str(t) for t in texts],
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )
        return [len(ids) for ids in enc["input_ids"]]

    @staticmethod
    def cosine_similarity(a: Tensor, b: Tensor) -> Tensor:
        """Row-wise cosine similarity between two ``(B, D)`` batches."""
        return F.cosine_similarity(a, b, dim=-1)


def load_encoder(
    model_path: str = DEFAULT_LOCAL_DIR,
    repo_id: str = DEFAULT_MODEL_NAME,
    device: Optional[torch.device] = None,
    max_seq_length: Optional[int] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    auto_download: bool = True,
    token: Optional[str] = None,
) -> SBERTEncoder:
    """Resolve the snapshot (downloading on first use) and build the encoder."""
    from src.utils import resolve_weights_path

    path = resolve_weights_path(
        model_path,
        repo_id=repo_id,
        token=token,
        auto_download=auto_download,
        ignore_patterns=IGNORE_PATTERNS,
        log_prefix="sbert",
    )
    return SBERTEncoder(
        path, device=device, max_seq_length=max_seq_length, batch_size=batch_size,
    )


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_LOCAL_DIR",
    "DEFAULT_MODEL_NAME",
    "FALLBACK_MAX_SEQ_LENGTH",
    "IGNORE_PATTERNS",
    "SBERTEncoder",
    "assert_mean_pooling",
    "load_encoder",
    "mean_pool",
    "read_max_seq_length",
]
