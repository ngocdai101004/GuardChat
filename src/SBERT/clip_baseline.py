"""The superseded CLIP similarity metric, kept runnable for comparison.

CLIP cosine similarity was Task 2's semantic-preservation metric in the
first revision, and review response W3 asked for a text-specific encoder
instead. Replacing a metric is a claim - "the old one was measuring the
wrong thing" - and a claim is worth more with both numbers on the table
than with one. So this adapter makes the vendored SafeGuider
:class:`CLIPEncoder` present the same surface as
:class:`src.SBERT.model.SBERTEncoder`, and
:mod:`src.SBERT.eval_similarity` can run either one over *identical*
records through *identical* code. Any divergence is then the encoder and
nothing else.

It lives inside the SBERT package rather than beside it for exactly that
reason: the two metrics are only meaningful next to each other.

What differs from SBERT, and why the numbers cannot be read the same way:

    * 77-token window against 384. On GuardChat this is the dominant
      effect, not a detail - conversations run to ~630 tokens.
    * The sentence vector is the hidden state at the EOS position, not a
      mean over tokens. That is how CLIP was trained and how SafeGuider
      uses it internally.
    * Embeddings are not L2-normalised here; the cosine does the
      normalising.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from torch import Tensor


DEFAULT_CLIP_MODEL = "openai/clip-vit-large-patch14"
DEFAULT_BATCH_SIZE = 64


class CLIPSimilarityEncoder:
    """Adapts the vendored ``CLIPEncoder`` to the SBERTEncoder interface.

    Only the four members :mod:`src.SBERT.similarity` actually calls are
    exposed: :meth:`encode`, :meth:`token_counts`,
    :meth:`cosine_similarity` and :attr:`max_seq_length`.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CLIP_MODEL,
        device: Optional[torch.device] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        verbose: bool = False,
    ) -> None:
        # Imported lazily: this pulls vendors/SafeGuider onto sys.path,
        # which a pure-SBERT run has no reason to pay for.
        from src.SafeGuider import CLIPEncoder

        self.encoder = CLIPEncoder(
            model_name=model_name, device=device, verbose=verbose,
        )
        self.batch_size = int(batch_size)
        self.model_path = model_name

    @property
    def max_seq_length(self) -> int:
        """77 for CLIP ViT-L/14 - the whole point of the comparison."""
        return int(self.encoder.max_length)

    @torch.no_grad()
    def encode(
        self,
        texts: Sequence[str],
        batch_size: Optional[int] = None,
    ) -> Tensor:
        """EOS-position embeddings, ``(N, 768)``, chunked."""
        if not texts:
            return torch.empty(0, self.encoder.hidden_size,
                               device=self.encoder.device)
        bs = int(batch_size or self.batch_size)
        chunks: List[Tensor] = [
            self.encoder.eos_embedding([str(t) for t in texts[i:i + bs]])
            for i in range(0, len(texts), bs)
        ]
        return torch.cat(chunks, dim=0)

    def token_counts(self, texts: Sequence[str]) -> List[int]:
        """Untruncated token length per text.

        The vendored encoder only counts one string at a time; the loop
        is here rather than there so the vendored file stays a faithful
        copy of upstream.
        """
        return [self.encoder.token_count(str(t)) for t in texts]

    @staticmethod
    def cosine_similarity(a: Tensor, b: Tensor) -> Tensor:
        import torch.nn.functional as F
        return F.cosine_similarity(a, b, dim=-1)


__all__ = ["DEFAULT_BATCH_SIZE", "DEFAULT_CLIP_MODEL", "CLIPSimilarityEncoder"]
