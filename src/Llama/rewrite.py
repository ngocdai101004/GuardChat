"""Task 2 rewrite pipeline using local Llama-3.1-8B-Instruct.

The open-source counterpart to :mod:`src.Gemini.rewrite`. Both rewriters
share the system prompts and the ``[Tn]`` turn contract
(:mod:`src.utils.rewrite_prompt`) and the record schema
(:class:`src.utils.RewriteRecord`), so Table 2 compares the models
rather than two different task framings.

Rewrites both GuardChat input representations:

``prompt``
    The enhanced adversarial prompt - one string in, one string out.
``conversation``
    The 6-9 turn dialogue, sent as a single ``[T1] ... [Tn]`` block and
    parsed back into turns so we can verify the model did not merge,
    drop or invent any.

Retries
-------
Llama-3.1-8B is a general instruct model, not a safety model: it refuses
some GuardChat inputs outright, and at 8B it sometimes breaks the turn
contract. Those rows are retried up to ``retries`` times, but the retry
**samples** instead of decoding greedily - a greedy re-run of the same
prompt is bit-for-bit identical, so retrying it would burn GPU time to
reproduce the same failure. The first attempt stays greedy, so a run in
which nothing fails is exactly reproducible.

There is no provider to block a request here, so ``blocked`` never
appears; the local analogue is ``refusal``.

No fallback text is ever substituted. Failed rows keep an empty
``rewritten_text`` plus a ``status`` and an ``error_kind``.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.utils import (
    GuardChatSample,
    RewriteRecord,
    base_record,
    build_rewrite_messages,
    classify_error_message,
    cleanup_rewrite_response,
    looks_like_refusal,
    normalise_rewrite_kind,
    parse_turns,
    resolve_hf_token,
    rewrite_system_prompt_for,
    rewritten_conversation_text,
)

from .model import (
    DEFAULT_LOCAL_DIR,
    DEFAULT_MODEL_NAME,
    GenerationConfig,
    LlamaConfig,
    LlamaModel,
)


# Statuses worth another attempt. A blank or unparseable answer is often
# a decoding accident an 8B model recovers from once the sampler is
# perturbed; a refusal likewise.
_RETRYABLE = frozenset({"refusal", "empty", "parse_failed"})


class RewritePipeline:
    """Local Llama-3.1-8B-Instruct rewriter wired to the GuardChat schema."""

    def __init__(
        self,
        model: LlamaModel,
        system_prompts: Optional[Dict[str, str]] = None,
        retries: int = 3,
    ) -> None:
        self.model = model
        self.system_prompts = dict(system_prompts or {})
        self.retries = max(1, int(retries))

    @classmethod
    def from_pretrained(
        cls,
        weights: str = DEFAULT_LOCAL_DIR,
        dtype: str = "auto",
        device: Optional[str] = None,
        token: Optional[str] = None,
        batch_size: int = 8,
        max_new_tokens: Optional[int] = None,
        retries: int = 3,
        auto_download: bool = True,
        system_prompts: Optional[Dict[str, str]] = None,
    ) -> "RewritePipeline":
        model = LlamaModel(LlamaConfig(
            model_path=weights,
            dtype=dtype,
            device=device,
            token=resolve_hf_token(token),
            batch_size=batch_size,
            generation=GenerationConfig(max_new_tokens=max_new_tokens),
            auto_download=auto_download,
        ))
        return cls(model=model, system_prompts=system_prompts, retries=retries)

    @property
    def model_name(self) -> str:
        return DEFAULT_MODEL_NAME

    def system_prompt(self, kind: str) -> str:
        return self.system_prompts.get(kind) or rewrite_system_prompt_for(kind)

    # ------------------------ Per-sample plumbing -------------------- #

    def _prepare(
        self,
        sample: GuardChatSample,
        kind: str,
    ) -> Tuple[RewriteRecord, Optional[List[Dict[str, str]]], int]:
        """Build the record, the chat messages, and the source length.

        ``messages`` is ``None`` when the row has nothing to rewrite; the
        record is already finalised in that case.
        """
        rec = base_record(sample, kind, model=self.model_name)

        if kind == "conversation":
            payload: Any = rec.original_turns
            has_content = bool(rec.original_turns)
        else:
            payload = rec.original_text
            has_content = bool(rec.original_text.strip())
        if not has_content:
            rec.status = "empty"
            rec.error_kind = "empty_input"
            return rec, None, 0

        messages = build_rewrite_messages(
            payload, kind=kind, system_prompt=self.system_prompt(kind),
        )
        return rec, messages, self.model.count_tokens(rec.original_text)

    def _finalise(self, rec: RewriteRecord, raw_text: str) -> None:
        """Turn one raw generation into a finished record."""
        rec.raw_response = raw_text

        if not raw_text.strip():
            rec.status = "empty"
            rec.error_kind = "empty_response"
            return
        if looks_like_refusal(raw_text):
            rec.status = "refusal"
            rec.error_kind = "model_refusal"
            return

        cleaned = cleanup_rewrite_response(raw_text, kind=rec.text_kind)
        if rec.text_kind == "conversation":
            self._finish_conversation(rec, cleaned)
        else:
            self._finish_prompt(rec, cleaned)

    @staticmethod
    def _finish_prompt(rec: RewriteRecord, cleaned: str) -> None:
        rec.rewritten_text = cleaned
        if cleaned.strip():
            rec.status = "ok"
            rec.error_kind = None
        else:
            rec.status = "empty"
            rec.error_kind = "empty_response"
        rec.was_modified = (
            cleaned.strip().lower() != rec.original_text.strip().lower()
        )

    @staticmethod
    def _finish_conversation(rec: RewriteRecord, cleaned: str) -> None:
        turns, parse_status = parse_turns(cleaned, expected=rec.num_turns_in)
        rec.turn_parse = parse_status
        rec.rewritten_turns = turns
        rec.num_turns_out = len(turns)

        if not turns:
            rec.status = "parse_failed"
            rec.error_kind = "parse_failed"
            rec.rewritten_text = ""
            return

        rec.rewritten_text = rewritten_conversation_text(turns)
        if rec.rewritten_text.strip():
            rec.status = "ok"
            rec.error_kind = None
        else:
            rec.status = "empty"
            rec.error_kind = "empty_response"
        rec.was_modified = (
            rec.rewritten_text.strip().lower() != rec.original_text.strip().lower()
        )

    # --------------------------- Inference --------------------------- #

    def rewrite_sample(
        self,
        sample: GuardChatSample,
        kind: str = "prompt",
    ) -> RewriteRecord:
        """Rewrite a single sample. Convenience wrapper over the batch path."""
        return self.rewrite_samples([sample], kind=kind, progress=False)[0]

    def rewrite_samples(
        self,
        samples: Sequence[GuardChatSample],
        kind: str = "prompt",
        on_result: Optional[Callable[[RewriteRecord], None]] = None,
        progress: bool = True,
    ) -> List[RewriteRecord]:
        """Rewrite a batch of samples under one representation.

        Work proceeds one GPU batch at a time. Within a batch, the first
        pass decodes greedily; rows that come back unusable are re-run
        with sampling until they succeed or ``retries`` is exhausted.
        Retrying inside the batch - rather than sweeping the whole split
        again at the end - keeps ``on_result`` in dataset order and lets
        the checkpoint advance batch by batch.
        """
        kind = normalise_rewrite_kind(kind)
        if not samples:
            return []

        bar = self._progress_bar(len(samples), kind) if progress else None
        out: List[RewriteRecord] = []
        size = max(1, int(self.model.config.batch_size))

        try:
            for start in range(0, len(samples), size):
                records = self._rewrite_chunk(samples[start:start + size], kind)
                for rec in records:
                    if on_result is not None:
                        on_result(rec)
                    out.append(rec)
                if bar is not None:
                    bar.update(len(records))
        finally:
            if bar is not None:
                bar.close()
        return out

    def _rewrite_chunk(
        self,
        chunk: Sequence[GuardChatSample],
        kind: str,
    ) -> List[RewriteRecord]:
        t0 = time.time()
        prepared = [self._prepare(s, kind) for s in chunk]
        records = [p[0] for p in prepared]

        # Rows with no input never reach the GPU.
        pending = [i for i, (_r, msgs, _n) in enumerate(prepared) if msgs is not None]

        for attempt in range(self.retries):
            if not pending:
                break
            batch = [prepared[i][1] for i in pending]
            budget = self.model.budget_for([prepared[i][2] for i in pending], kind)
            try:
                replies = self.model.generate_batch(
                    batch, max_new_tokens=budget, sample=(attempt > 0),
                )
            except Exception as e:  # noqa: BLE001 - record, do not kill the run
                # OOM is the realistic case here and it takes the whole
                # batch with it. Mark every row still pending so the
                # cause survives into the results file.
                self._fail_batch(records, pending, attempt, e)
                pending = []
                break

            still: List[int] = []
            for i, raw in zip(pending, replies):
                rec = records[i]
                rec.attempts = attempt + 1
                self._finalise(rec, raw)
                if rec.status in _RETRYABLE and attempt + 1 < self.retries:
                    still.append(i)
            pending = still

        # One wall-clock measurement per batch, shared out across its
        # rows - per-sample timings are meaningless under batching.
        elapsed = (time.time() - t0) / max(1, len(records))
        for rec in records:
            rec.elapsed_sec = elapsed
        return records

    @staticmethod
    def _fail_batch(
        records: List[RewriteRecord],
        pending: Sequence[int],
        attempt: int,
        exc: Exception,
    ) -> None:
        kind = classify_error_message(str(exc))
        for i in pending:
            rec = records[i]
            rec.attempts = attempt + 1
            rec.status = "error"
            rec.error_kind = kind
            rec.block_reason = f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _progress_bar(total: int, kind: str) -> Any:
        try:
            from tqdm.auto import tqdm
        except ImportError:
            return None
        return tqdm(total=total, desc=f"Llama[{kind}]")


__all__ = ["RewritePipeline"]
