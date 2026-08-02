"""Safety-aware beam search rewriter — extracted from upstream SafeGuider.

Source: ``stable-diffusion-1.4/scripts/safeguider_gene.py`` in
https://github.com/pgqihere/safeguider (SafeGuider, ACM CCS 2025). In
the upstream repo the rewriter is not a module: it is ~200 lines inlined
into the middle of a Stable-Diffusion sampling loop, entangled with LDM
model loading, DDIM sampling and PNG writing. This file is that logic
lifted out verbatim in behaviour, with the diffusion model replaced by a
standalone CLIP text encoder (``encoder.py``) so the rewriter can run
without Stable Diffusion checked out.

The algorithm, unchanged
------------------------
When the recognizer judges a prompt UNSAFE, search for a revision — by
DELETING WORDS, never replacing or inserting — such that:

    1. safety_score (P[class=safe]) >= safety_threshold  (0.80)
    2. cos(new EOS embedding, original) >= similarity_floor  (0.1)
    3. (tie-break) as few words removed as possible

    1) Rank each word by contribution to unsafe: delete it alone and
       measure how much the safety score rises.
    2) Beam search to max_depth, deleting one more word per step.
    3) Early stop once a candidate clears both thresholds.
    4) Fallback: no qualifying candidate -> take the highest-safety one
       among those still above similarity_floor.

Deliberate deviations from upstream
-----------------------------------
Two, both listed here so the reimplementation can be audited:

*Batched scoring.* Upstream encodes one candidate string per forward
pass. A 99-word GuardChat prompt (the test-split mean) expands to
thousands of candidates, so one-at-a-time encoding puts a 1,000-sample
run into the multi-hour range. Candidates within a beam step are
independent, so they are encoded in batches of ``batch_size`` and
memoised per prompt. The tokenizer pads to a fixed 77 either way, so a
batched forward pass is numerically the same as the loop it replaces —
this changes the cost, not the result.

*Termination.* Upstream lets ``candidates`` become the empty list when a
depth expands nothing (every word already deleted) and then spins
through the remaining depths doing no work. Here that breaks the loop.
Same output, fewer wasted iterations.

Everything else — the 0.80 / 0.1 thresholds, beam width 6, depth
min(25, len(words) - 1), the sort keys, the tie-breaks, the fallback
rule, the "class 1 = safe" convention — matches ``safeguider_gene.py``
line for line.

Encoder window
--------------
CLIP truncates at 77 tokens. Anything past that is invisible to the
recognizer, so deleting a word there cannot change the safety score and
the search will never touch it. That is a property of SafeGuider, not of
this port, but it bites hard on GuardChat's long prompts — so
:class:`BeamSearchResult` reports ``num_tokens`` and ``truncated`` and
the caller can quantify the reach.

This module is standalone: no Stable Diffusion / LDM import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from encoder import CLIPEncoder


# Defaults transcribed from `safeguider_gene.py` (beam_width = 6,
# max_depth = min(25, ...), `current_safety >= 0.80 and similarity >= 0.1`).
DEFAULT_BEAM_WIDTH: int = 6
DEFAULT_MAX_DEPTH: int = 25
DEFAULT_SAFETY_THRESHOLD: float = 0.80
DEFAULT_SIMILARITY_FLOOR: float = 0.1

# Not upstream: how many candidate strings share one encoder forward
# pass. Pure throughput knob — see "Deliberate deviations" above.
DEFAULT_BATCH_SIZE: int = 64


@dataclass
class BeamSearchResult:
    original_prompt: str
    modified_prompt: str
    was_modified: bool
    original_safety: float
    modified_safety: float
    similarity: float
    removed_tokens: List[str] = field(default_factory=list)

    # How the search ended:
    #   qualified   a candidate cleared both thresholds
    #   fallback    none did; took the safest above similarity_floor
    #   unchanged   nothing above similarity_floor - kept the original
    #   single_word too short to delete from
    outcome: str = "unchanged"
    depth_reached: int = 0

    # Encoder reach on the ORIGINAL prompt. `truncated` means the tail of
    # the prompt sat outside CLIP's window and no deletion there could
    # have moved the score.
    num_tokens: int = 0
    truncated: bool = False

    # Cost accounting: distinct strings actually pushed through CLIP.
    num_encoded: int = 0

    # Trace for debug/logging; not required for normal usage.
    log: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_prompt": self.original_prompt,
            "modified_prompt": self.modified_prompt,
            "was_modified": self.was_modified,
            "original_safety": self.original_safety,
            "modified_safety": self.modified_safety,
            "similarity": self.similarity,
            "removed_tokens": self.removed_tokens,
            "outcome": self.outcome,
            "depth_reached": self.depth_reached,
            "num_tokens": self.num_tokens,
            "truncated": self.truncated,
            "num_encoded": self.num_encoded,
        }


class SafetyAwareBeamSearch:
    """Rewriter using beam search at the word (whitespace split) level."""

    def __init__(
        self,
        encoder: CLIPEncoder,
        classifier: torch.nn.Module,
        beam_width: int = DEFAULT_BEAM_WIDTH,
        max_depth: int = DEFAULT_MAX_DEPTH,
        safety_threshold: float = DEFAULT_SAFETY_THRESHOLD,
        similarity_floor: float = DEFAULT_SIMILARITY_FLOOR,
        batch_size: int = DEFAULT_BATCH_SIZE,
        verbose: bool = False,
    ) -> None:
        self.encoder = encoder
        self.classifier = classifier.eval()
        self.device = self.encoder.device

        self.beam_width = int(beam_width)
        self.max_depth = int(max_depth)
        self.safety_threshold = float(safety_threshold)
        self.similarity_floor = float(similarity_floor)
        self.batch_size = max(1, int(batch_size))
        self.verbose = bool(verbose)

    # ============================ Public API ============================ #

    @torch.no_grad()
    def is_safe(self, prompt: str) -> Tuple[bool, float]:
        """The recognizer gate, as upstream applies it before rewriting.

        ``safeguider_gene.py`` computes ``argmax(probs)`` over the EOS
        embedding and skips the rewrite entirely when the answer is
        class 1. Returns ``(is_safe, P[safe])``.
        """
        eos = self.encoder.eos_embedding([prompt])
        _, probs = self.classifier(eos)
        if probs.dim() == 3:
            probs = probs.squeeze(1)
        return bool(int(probs[0].argmax().item()) == 1), float(probs[0, 1].item())

    @torch.no_grad()
    def rewrite(self, prompt: str) -> BeamSearchResult:
        """Rewrite an UNSAFE prompt, or return it unchanged if no solution exists."""
        words: List[str] = prompt.split()
        log: List[str] = []

        # One memo table per rewrite: candidate string -> (safety, similarity).
        # Beams overlap heavily (the same word set is reachable by many
        # deletion orders), so this saves more forward passes than the
        # batching does. Dropped on return, which bounds the memory.
        cache: Dict[str, Tuple[float, float]] = {}

        num_tokens = self._token_count(prompt)
        truncated = bool(num_tokens > int(getattr(self.encoder, "max_length", 77)))

        orig_eos = self.encoder.eos_embedding([prompt])
        orig_safety = float(self._safety_scores(orig_eos)[0].item())
        cache[prompt] = (orig_safety, 1.0)
        log.append(f"original safety: {orig_safety:.4f}"
                   + (f" | TRUNCATED at {self.encoder.max_length} tokens "
                      f"(prompt is {num_tokens})" if truncated else ""))

        def _unchanged(outcome: str, depth: int = 0) -> BeamSearchResult:
            return BeamSearchResult(
                original_prompt=prompt, modified_prompt=prompt, was_modified=False,
                original_safety=orig_safety, modified_safety=orig_safety,
                similarity=1.0, removed_tokens=[], outcome=outcome,
                depth_reached=depth, num_tokens=num_tokens, truncated=truncated,
                num_encoded=len(cache), log=log,
            )

        if len(words) <= 1:
            log.append("prompt has <=1 word; beam search is not possible.")
            return _unchanged("single_word")

        # 1) Impact ranking: delete each word on its own, measure the
        #    safety gain. One batched sweep instead of len(words) passes.
        singles: List[str] = []
        single_idx: List[int] = []
        for idx in range(len(words)):
            modified = " ".join(words[:idx] + words[idx + 1:])
            if not modified.strip():
                continue
            singles.append(modified)
            single_idx.append(idx)

        scored = self._evaluate(singles, orig_eos, cache)
        token_impacts: List[Tuple[int, float]] = [
            (idx, safety - orig_safety)
            for idx, (safety, _sim) in zip(single_idx, scored)
        ]
        # Stable sort, so ties keep prompt order exactly as upstream does.
        token_impacts.sort(key=lambda x: x[1], reverse=True)
        log.append(f"impact ranking (top 5): {token_impacts[:5]}")

        # 2) Beam search.
        max_depth = min(self.max_depth, len(words) - 1)
        # Each candidate: (removed_indices, improvement, similarity)
        candidates: List[Tuple[List[int], float, float]] = [([], 0.0, 1.0)]

        best_modified_prompt: Optional[str] = None
        best_safety_improvement: float = 0.0
        best_similarity: float = 0.0
        best_tokens_removed: List[str] = []
        # Every candidate ever scored, kept for the fallback rule.
        all_seen: List[Tuple[List[int], float, float, float, str]] = []
        depth = 0

        for depth in range(1, max_depth + 1):
            # Enumerate this depth's expansions, then score them together.
            expansions: List[Tuple[List[int], str]] = []
            for removed_indices, _imp, _sim in candidates:
                removed_set = set(removed_indices)
                for idx, _impact in token_impacts:
                    if idx in removed_set:
                        continue
                    new_indices = removed_indices + [idx]
                    keep = removed_set | {idx}
                    new_words = [w for i, w in enumerate(words) if i not in keep]
                    if not new_words:
                        continue
                    expansions.append((new_indices, " ".join(new_words)))

            if not expansions:
                log.append(f"depth {depth}: nothing left to expand; stopping.")
                break

            stats = self._evaluate([text for _idx, text in expansions], orig_eos, cache)

            step: List[Tuple[List[int], float, float, float]] = []
            qualified: List[Tuple[List[int], float, float]] = []
            for (new_indices, text), (cur_safety, sim) in zip(expansions, stats):
                improvement = cur_safety - orig_safety
                step.append((new_indices, improvement, sim, cur_safety))
                all_seen.append((new_indices, improvement, sim, cur_safety, text))

                if cur_safety >= self.safety_threshold and sim >= self.similarity_floor:
                    qualified.append((new_indices, improvement, sim))
                    # Maximise improvement, then minimise removals.
                    if (best_modified_prompt is None
                            or improvement > best_safety_improvement
                            or (improvement == best_safety_improvement
                                and len(new_indices) < len(best_tokens_removed))):
                        best_modified_prompt = text
                        best_safety_improvement = improvement
                        best_similarity = sim
                        best_tokens_removed = [words[i] for i in new_indices]

            if qualified:
                candidates = sorted(
                    qualified, key=lambda x: (x[1], -len(x[0])),
                )[-self.beam_width:]
                tag = "qualified"
            else:
                # No qualifier: keep the highest raw safety, fewer removals first.
                topk = sorted(step, key=lambda x: (x[3], -len(x[0])))[-self.beam_width:]
                candidates = [(ind, imp, sim) for ind, imp, sim, _s in topk]
                tag = "fallback"

            log.append(f"depth {depth}: {len(qualified)} qualified, "
                       f"{len(step)} expanded, picked={tag}")

            # Early stopping.
            if (best_modified_prompt is not None
                    and (best_safety_improvement + orig_safety) >= self.safety_threshold):
                log.append("early stop: found satisfactory solution.")
                break

        # 3) Final selection.
        if best_modified_prompt is not None:
            mod_safety = orig_safety + best_safety_improvement
            log.append(f"chose qualified: removed={best_tokens_removed} "
                       f"safety={mod_safety:.4f}")
            return BeamSearchResult(
                original_prompt=prompt,
                modified_prompt=best_modified_prompt,
                was_modified=True,
                original_safety=orig_safety,
                modified_safety=mod_safety,
                similarity=best_similarity,
                removed_tokens=best_tokens_removed,
                outcome="qualified",
                depth_reached=depth,
                num_tokens=num_tokens,
                truncated=truncated,
                num_encoded=len(cache),
                log=log,
            )

        # Fallback: highest safety among candidates above similarity_floor.
        valid = [c for c in all_seen if c[2] >= self.similarity_floor]
        if valid:
            best_indices, _imp, best_sim, best_safety, best_alt = max(
                valid, key=lambda x: x[3],
            )
            log.append(f"chose fallback: removed={[words[i] for i in best_indices]} "
                       f"safety={best_safety:.4f}")
            return BeamSearchResult(
                original_prompt=prompt,
                modified_prompt=best_alt,
                was_modified=(best_alt != prompt),
                original_safety=orig_safety,
                modified_safety=best_safety,
                similarity=best_sim,
                removed_tokens=[words[i] for i in best_indices],
                outcome="fallback",
                depth_reached=depth,
                num_tokens=num_tokens,
                truncated=truncated,
                num_encoded=len(cache),
                log=log,
            )

        log.append("no candidate satisfied similarity_floor - keeping the original.")
        return _unchanged("unchanged", depth)

    # ============================ Internals ============================== #

    @torch.no_grad()
    def _safety_scores(self, eos_emb: torch.Tensor) -> torch.Tensor:
        """P[class=safe] per row, given EOS embeddings (B, D)."""
        # Upstream unsqueezes to (B, 1, D) before the classifier (see
        # `compute_safety_score` in safeguider_gene.py) and squeezes the
        # result back. (B, D) gives the same numbers - every layer is a
        # Linear over the last dim - and avoids the reshape dance.
        _, probs = self.classifier(eos_emb)
        if probs.dim() == 3:
            probs = probs.squeeze(1)
        return probs[:, 1]

    @torch.no_grad()
    def _evaluate(
        self,
        prompts: Sequence[str],
        orig_eos: torch.Tensor,
        cache: Dict[str, Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """(safety, similarity-to-original) for each prompt, batched + memoised."""
        # dict.fromkeys de-duplicates while keeping order, so a repeated
        # candidate costs one forward pass rather than none-per-copy.
        todo = [p for p in dict.fromkeys(prompts) if p not in cache]
        for start in range(0, len(todo), self.batch_size):
            chunk = todo[start:start + self.batch_size]
            eos = self.encoder.eos_embedding(chunk)                  # (B, D)
            safeties = self._safety_scores(eos)                      # (B,)
            # orig_eos is (1, D) and broadcasts against (B, D).
            sims = self.encoder.cosine_similarity(orig_eos, eos)     # (B,)
            for text, s, c in zip(chunk, safeties.tolist(), sims.tolist()):
                cache[text] = (float(s), float(c))
        return [cache[p] for p in prompts]

    def _token_count(self, prompt: str) -> int:
        """Untruncated token count, for the ``truncated`` diagnostic."""
        counter = getattr(self.encoder, "token_count", None)
        if counter is None:
            return 0
        try:
            return int(counter(prompt))
        except Exception:  # noqa: BLE001 - diagnostic only, never fatal
            return 0
