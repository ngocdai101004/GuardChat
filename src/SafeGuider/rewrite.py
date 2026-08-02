"""Task 2 rewrite pipeline using SafeGuider's safety-aware beam search.

The third Task-2 baseline, and the only one that is not a language
model. Where Gemini and Llama are *asked* to sanitise a prompt,
SafeGuider *searches* for a sanitised prompt: it scores the CLIP EOS
embedding with the pre-trained binary recognizer shipped upstream
(``SD1.4_safeguider.pt``) and beam-searches over word deletions until the
score crosses the safe threshold. See ``vendors/SafeGuider/beam_search.py``
for the algorithm and its provenance.

Three consequences follow from "delete words, never substitute", and all
three matter when reading the results:

* The output is always a subsequence of the input. SafeGuider can only
  remove meaning, never replace it, so its length ratio is a hard
  ceiling on how much of the original prompt survives.
* The recognizer gate is part of the system. Upstream skips the rewrite
  entirely when the recognizer calls a prompt safe, and so does this
  port by default (``gate="recognizer"``). A prompt that slips past the
  recognizer reaches the T2I model untouched — a real SafeGuider failure
  mode, and one that would be hidden by forcing a rewrite on every row.
  ``gate="always"`` is available for the ablation.
* CLIP sees 77 tokens. GuardChat's enhanced prompts average 99 *words*,
  so most of a prompt is outside the window and cannot be searched. Each
  record carries ``truncated`` and ``num_tokens`` so the reach can be
  reported rather than assumed.

Conversations
-------------
The dialogue is rewritten **turn by turn**, then reassembled. Running
the beam search over the flattened conversation instead would be
meaningless twice over: CLIP would see only the first turn and a bit of
the second, and word deletion could eat the ``user:`` prefixes that
define the turn boundaries. Per-turn search keeps the turn count exactly
by construction — there is no model output to parse, so ``turn_parse``
is always ``"ok"`` and, unlike the LLM baselines, cannot be otherwise.

No fallback text is ever substituted. Rows that fail keep an empty
``rewritten_text`` plus a ``status`` and an ``error_kind``.
"""

from __future__ import annotations

import os
import statistics
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from src.utils import (
    GuardChatSample,
    RewriteRecord,
    base_record,
    classify_error_message,
    normalise_rewrite_kind,
    rewritten_conversation_text,
)

from . import (
    CLIPEncoder,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BEAM_WIDTH,
    DEFAULT_MAX_DEPTH,
    DEFAULT_SAFETY_THRESHOLD,
    DEFAULT_SIMILARITY_FLOOR,
    SafetyAwareBeamSearch,
    ThreeLayerClassifier,
)


DEFAULT_MODEL_NAME = "SafeGuider-beam-search"
DEFAULT_ENCODER_MODEL = "openai/clip-vit-large-patch14"

# Where `scripts/env.sh` puts the upstream recognizer checkpoint.
DEFAULT_WEIGHTS = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "vendors", "SafeGuider", "weights", "SD1.4_safeguider.pt",
))

# "recognizer" reproduces the published pipeline: classify first, rewrite
# only what comes back unsafe. "always" skips the gate and rewrites every
# row, which measures the beam search in isolation.
GATE_MODES = ("recognizer", "always")


class RewritePipeline:
    """SafeGuider beam-search rewriter wired to the GuardChat Task-2 schema."""

    def __init__(
        self,
        searcher: SafetyAwareBeamSearch,
        encoder_model: str = DEFAULT_ENCODER_MODEL,
        weights: str = DEFAULT_WEIGHTS,
        gate: str = "recognizer",
    ) -> None:
        if gate not in GATE_MODES:
            raise ValueError(f"gate must be one of {GATE_MODES}, got {gate!r}")
        self.searcher = searcher
        self.encoder_model = encoder_model
        self.weights = weights
        self.gate = gate

    @classmethod
    def from_weights(
        cls,
        weights: str = DEFAULT_WEIGHTS,
        encoder_model: str = DEFAULT_ENCODER_MODEL,
        device: Optional[str] = None,
        beam_width: int = DEFAULT_BEAM_WIDTH,
        max_depth: int = DEFAULT_MAX_DEPTH,
        safety_threshold: float = DEFAULT_SAFETY_THRESHOLD,
        similarity_floor: float = DEFAULT_SIMILARITY_FLOOR,
        batch_size: int = DEFAULT_BATCH_SIZE,
        gate: str = "recognizer",
        verbose: bool = False,
    ) -> "RewritePipeline":
        if not os.path.isfile(weights):
            raise FileNotFoundError(
                f"SafeGuider recognizer weights not found: {weights!r}. "
                f"Copy SD1.4_safeguider.pt from the upstream repo's Models/ "
                f"folder into vendors/SafeGuider/weights/."
            )
        encoder = CLIPEncoder(model_name=encoder_model, device=device, verbose=verbose)
        classifier = ThreeLayerClassifier(dim=encoder.hidden_size).to(encoder.device)
        state = torch.load(weights, map_location=encoder.device, weights_only=False)
        classifier.load_state_dict(state)
        classifier.eval()

        searcher = SafetyAwareBeamSearch(
            encoder=encoder,
            classifier=classifier,
            beam_width=beam_width,
            max_depth=max_depth,
            safety_threshold=safety_threshold,
            similarity_floor=similarity_floor,
            batch_size=batch_size,
            verbose=verbose,
        )
        return cls(searcher=searcher, encoder_model=encoder_model,
                   weights=weights, gate=gate)

    @property
    def model_name(self) -> str:
        return DEFAULT_MODEL_NAME

    @property
    def encoder(self) -> CLIPEncoder:
        return self.searcher.encoder

    # ------------------------- One piece of text --------------------- #

    def rewrite_text(self, text: str) -> Dict[str, Any]:
        """Gate + beam-search one string; report what happened.

        Returns a dict with ``text`` (the result) and the diagnostics
        that go into ``RewriteRecord.extra``.
        """
        source = text.strip()
        if not source:
            return {"text": "", "empty": True}

        if self.gate == "recognizer":
            safe, score = self.searcher.is_safe(source)
            if safe:
                # Upstream behaviour: the recognizer cleared it, so the
                # prompt goes to the T2I model exactly as written.
                return {
                    "text": source,
                    "gated_safe": True,
                    "original_safety": round(score, 6),
                    "modified_safety": round(score, 6),
                    "was_modified": False,
                }

        r = self.searcher.rewrite(source)
        return {
            "text": r.modified_prompt,
            "gated_safe": False,
            "was_modified": bool(r.was_modified),
            "original_safety": round(float(r.original_safety), 6),
            "modified_safety": round(float(r.modified_safety), 6),
            "beam_similarity": round(float(r.similarity), 6),
            "removed_tokens": list(r.removed_tokens),
            "num_removed": len(r.removed_tokens),
            "outcome": r.outcome,
            "depth_reached": r.depth_reached,
            "num_tokens": r.num_tokens,
            "truncated": r.truncated,
            "num_encoded": r.num_encoded,
        }

    # ---------------------------- One sample ------------------------- #

    def rewrite_sample(
        self,
        sample: GuardChatSample,
        kind: str = "prompt",
    ) -> RewriteRecord:
        kind = normalise_rewrite_kind(kind)
        rec = base_record(sample, kind, model=self.model_name)
        t0 = time.time()
        try:
            if kind == "conversation":
                self._do_conversation(rec)
            else:
                self._do_prompt(rec)
        except Exception as e:  # noqa: BLE001 - record it, do not kill the run
            rec.status = "error"
            rec.error_kind = classify_error_message(str(e))
            rec.rewritten_text = ""
            rec.block_reason = f"{type(e).__name__}: {e}"
        rec.elapsed_sec = time.time() - t0
        return rec

    def _do_prompt(self, rec: RewriteRecord) -> None:
        if not rec.original_text.strip():
            rec.status = "empty"
            rec.error_kind = "empty_input"
            return

        info = self.rewrite_text(rec.original_text)
        rec.rewritten_text = str(info.get("text", ""))
        rec.was_modified = bool(info.get("was_modified"))
        rec.extra = {k: v for k, v in info.items() if k not in ("text", "was_modified")}
        # The searcher always returns a string - the only way to land
        # here empty is a prompt that was whitespace after stripping,
        # which the guard above already caught.
        if rec.rewritten_text.strip():
            rec.status = "ok"
            rec.error_kind = None
        else:
            rec.status = "empty"
            rec.error_kind = "empty_response"

    def _do_conversation(self, rec: RewriteRecord) -> None:
        turns = rec.original_turns or []
        if not turns:
            rec.status = "empty"
            rec.error_kind = "empty_input"
            rec.turn_parse = "parse_failed"
            rec.num_turns_out = 0
            return

        infos = [self.rewrite_text(t) for t in turns]
        rewritten = [str(i.get("text", "")) for i in infos]

        rec.rewritten_turns = rewritten
        rec.num_turns_out = len(rewritten)
        # Turn structure is built here, not parsed out of a generation,
        # so it cannot drift. Recorded anyway to keep the field
        # comparable with the LLM baselines.
        rec.turn_parse = "ok"
        rec.rewritten_text = rewritten_conversation_text(rewritten)
        rec.was_modified = any(i.get("was_modified") for i in infos)
        rec.extra = self._summarise_turns(infos)

        if rec.rewritten_text.strip():
            rec.status = "ok"
            rec.error_kind = None
        else:
            rec.status = "empty"
            rec.error_kind = "empty_response"

    @staticmethod
    def _summarise_turns(infos: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """Roll per-turn diagnostics up, keeping the per-turn detail too."""
        def _mean(key: str) -> Optional[float]:
            vals = [float(i[key]) for i in infos if i.get(key) is not None]
            return round(statistics.fmean(vals), 6) if vals else None

        per_turn = [
            {k: v for k, v in i.items() if k not in ("text", "removed_tokens")}
            for i in infos
        ]
        return {
            "num_turns_gated_safe": sum(1 for i in infos if i.get("gated_safe")),
            "num_turns_modified": sum(1 for i in infos if i.get("was_modified")),
            "num_turns_truncated": sum(1 for i in infos if i.get("truncated")),
            "num_removed_total": sum(int(i.get("num_removed") or 0) for i in infos),
            "mean_original_safety": _mean("original_safety"),
            "mean_modified_safety": _mean("modified_safety"),
            "total_encoded": sum(int(i.get("num_encoded") or 0) for i in infos),
            "removed_tokens_per_turn": [list(i.get("removed_tokens") or []) for i in infos],
            "turns": per_turn,
        }

    # --------------------------- The batch loop ---------------------- #

    def rewrite_samples(
        self,
        samples: Sequence[GuardChatSample],
        kind: str = "prompt",
        on_result: Optional[Callable[[RewriteRecord], None]] = None,
        progress: bool = True,
    ) -> List[RewriteRecord]:
        """Rewrite a split under one representation.

        Strictly sequential: the beam search already batches the encoder
        internally and holds the GPU for the whole of one sample, so
        there is nothing for an outer batch to overlap with. That also
        makes ``elapsed_sec`` a real per-sample measurement here, unlike
        the Llama pipeline where a batch's wall clock is shared out.
        """
        kind = normalise_rewrite_kind(kind)
        if not samples:
            return []

        bar = self._progress_bar(len(samples), kind) if progress else None
        out: List[RewriteRecord] = []
        try:
            for sample in samples:
                rec = self.rewrite_sample(sample, kind=kind)
                if on_result is not None:
                    on_result(rec)
                out.append(rec)
                if bar is not None:
                    bar.update(1)
        finally:
            if bar is not None:
                bar.close()
        return out

    @staticmethod
    def _progress_bar(total: int, kind: str) -> Any:
        try:
            from tqdm.auto import tqdm
        except ImportError:
            return None
        return tqdm(total=total, desc=f"SafeGuider[{kind}]")


__all__ = [
    "DEFAULT_ENCODER_MODEL",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_WEIGHTS",
    "GATE_MODES",
    "RewritePipeline",
]
