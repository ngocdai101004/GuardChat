"""Task 2 - NSFW concept removal via prompt rewriting.

The vendored ``SafeGuiderInputGuard`` already does the work: it runs the
original SafeGuider binary classifier over CLIP-EOS embeddings and, when
a prompt is unsafe, beam-searches for a token-deletion rewrite that
crosses the safety threshold while preserving cosine similarity to the
original embedding.

This module is a thin GuardChat-aware wrapper that:
    * accepts :class:`GuardChatSample` (or plain prompt strings),
    * always rewrites the enhanced toxic prompt regardless of the
      classifier's binary verdict (the test set is by construction
      adversarial, so we treat every entry as needing sanitisation),
    * emits records in the standardised evaluation schema used by both
      the CLIP-similarity metric in this repo and any downstream
      Safe Generation Rate evaluation against external T2I systems.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch

from . import (
    CLIPEncoder,
    DEFAULT_BEAM_WIDTH,
    DEFAULT_MAX_DEPTH,
    DEFAULT_SAFETY_THRESHOLD,
    DEFAULT_SIMILARITY_FLOOR,
    SafetyAwareBeamSearch,
    ThreeLayerClassifier,
)
from .data import GuardChatSample


@dataclass
class RewriteResult:
    """One rewrite output, ready for serialisation."""

    sample_id: str
    original_prompt: str
    rewritten_prompt: str
    was_modified: bool
    removed_tokens: List[str]
    original_safety: float
    modified_safety: float
    safeguider_similarity: float
    elapsed_sec: float
    label_names: List[str]
    source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "original_prompt": self.original_prompt,
            "rewritten_prompt": self.rewritten_prompt,
            "was_modified": self.was_modified,
            "removed_tokens": self.removed_tokens,
            "original_safety": self.original_safety,
            "modified_safety": self.modified_safety,
            "safeguider_similarity": self.safeguider_similarity,
            "elapsed_sec": self.elapsed_sec,
            "label_names": self.label_names,
            "source": self.source,
        }


class RewritePipeline:
    """SafeGuider beam-search rewriter wired to the GuardChat schema."""

    def __init__(
        self,
        weights: str,
        encoder_model: str = "openai/clip-vit-large-patch14",
        device: Optional[str] = None,
        beam_width: int = DEFAULT_BEAM_WIDTH,
        max_depth: int = DEFAULT_MAX_DEPTH,
        safety_threshold: float = DEFAULT_SAFETY_THRESHOLD,
        similarity_floor: float = DEFAULT_SIMILARITY_FLOOR,
        verbose: bool = False,
    ) -> None:
        if not os.path.isfile(weights):
            raise FileNotFoundError(
                f"SafeGuider binary classifier weights not found: {weights!r}. "
                f"Place SD1.4_safeguider.pt (or equivalent) at this path."
            )
        self.encoder = CLIPEncoder(model_name=encoder_model, device=device, verbose=verbose)
        self.classifier = ThreeLayerClassifier(dim=self.encoder.hidden_size).to(self.encoder.device)
        state = torch.load(weights, map_location=self.encoder.device, weights_only=False)
        self.classifier.load_state_dict(state)
        self.classifier.eval()

        self.beam_searcher = SafetyAwareBeamSearch(
            encoder=self.encoder,
            classifier=self.classifier,
            beam_width=beam_width,
            max_depth=max_depth,
            safety_threshold=safety_threshold,
            similarity_floor=similarity_floor,
            verbose=verbose,
        )

    @torch.no_grad()
    def rewrite_prompt(self, prompt: str, sample_id: str = "0",
                       label_names: Optional[List[str]] = None,
                       source: Optional[str] = None) -> RewriteResult:
        t0 = time.time()
        r = self.beam_searcher.rewrite(prompt)
        return RewriteResult(
            sample_id=sample_id,
            original_prompt=prompt,
            rewritten_prompt=r.modified_prompt,
            was_modified=bool(r.was_modified),
            removed_tokens=list(r.removed_tokens),
            original_safety=float(r.original_safety),
            modified_safety=float(r.modified_safety),
            safeguider_similarity=float(r.similarity),
            elapsed_sec=round(time.time() - t0, 4),
            label_names=list(label_names or []),
            source=source,
        )

    @torch.no_grad()
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
