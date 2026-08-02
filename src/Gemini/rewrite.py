"""Task 2 rewrite pipeline using the Gemini 2.5 Flash API.

Rewrites both GuardChat input representations:

``prompt``
    The enhanced adversarial prompt - one string in, one string out.
``conversation``
    The 6-9 turn dialogue, sent as a single ``[T1] ... [Tn]`` block and
    parsed back into turns so we can verify the model did not merge,
    drop or invent any.

The task framing, system prompts and parsers are shared with the local
Llama-3.1-8B rewriter (:mod:`src.utils.rewrite_prompt`) and the record
schema is shared with every Task-2 baseline
(:class:`src.utils.RewriteRecord`), so the two runs are directly
comparable and one aggregator can compose Table 2.

API only - there is no checkpoint on disk. Calls are fanned out across a
small thread pool because 1,000 samples x 2 representations is a long
wait at one request at a time; the shared evaluation loop checkpoints
each result as it lands, so ordering across workers does not matter.

No fallback text is ever substituted. When Gemini blocks, refuses or
errors, the record keeps an empty ``rewritten_text``, a ``status``, and
an ``error_kind`` saying why - separating "this sample is unrewritable"
(``provider_block``, ``model_refusal``) from "the run hit a limit"
(``quota``, ``auth``, ``network``), because only the first belongs in
the paper. Filling those rows with "a serene landscape" would hand the
rewriter a free Safe Generation Rate point for a sample it failed to
handle.

Failures - including safety blocks - are retried up to
``config.retries`` times before being recorded.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.utils import (
    GuardChatSample,
    RewriteRecord,
    base_record,
    build_rewrite_user_message,
    classify_error_message,
    cleanup_rewrite_response,
    looks_like_refusal,
    normalise_rewrite_kind,
    parse_turns,
    rewrite_system_prompt_for,
    rewritten_conversation_text,
)

from .client import (
    DEFAULT_MODEL_NAME,
    GeminiClient,
    GeminiClientConfig,
    GenerationConfig,
)


# Rewriting a 6-9 turn dialogue means re-emitting all of it, so the
# conversation branch needs a much larger budget than the single-prompt
# branch. The longest test conversation is ~5.3k characters. Output
# tokens are billed on what is actually produced, so a generous ceiling
# costs nothing on short samples and prevents truncation on long ones.
DEFAULT_MAX_OUTPUT_TOKENS: Dict[str, int] = {
    "prompt": 2048,
    "conversation": 4096,
}


class RewritePipeline:
    """Gemini 2.5 Flash rewriter wired to the GuardChat schema."""

    def __init__(
        self,
        client: GeminiClient,
        system_prompts: Optional[Dict[str, str]] = None,
        workers: int = 4,
    ) -> None:
        self.client = client
        self.system_prompts = dict(system_prompts or {})
        self.workers = max(1, int(workers))

    @classmethod
    def from_api_key(
        cls,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        max_output_tokens: Optional[int] = None,
        temperature: float = 0.0,
        thinking_budget: Optional[int] = 0,
        relax_safety: bool = True,
        retries: int = 3,
        retry_blocked: bool = True,
        backoff_seconds: float = 2.0,
        request_timeout: Optional[float] = 120.0,
        workers: int = 4,
        system_prompts: Optional[Dict[str, str]] = None,
    ) -> "RewritePipeline":
        """Build a pipeline from explicit settings.

        ``api_key`` defaults to the ``GEMINI_API_KEY`` /
        ``GOOGLE_API_KEY`` environment variables.

        ``max_output_tokens=None`` selects the per-kind default from
        :data:`DEFAULT_MAX_OUTPUT_TOKENS` at call time; pass a number to
        pin both representations to the same ceiling.
        """
        cfg = GeminiClientConfig(
            model_name=model_name,
            api_key=api_key,
            relax_safety=relax_safety,
            retries=retries,
            retry_blocked=retry_blocked,
            backoff_seconds=backoff_seconds,
            request_timeout=request_timeout,
            generation=GenerationConfig(
                max_output_tokens=max_output_tokens or DEFAULT_MAX_OUTPUT_TOKENS["prompt"],
                temperature=temperature,
                thinking_budget=thinking_budget,
            ),
        )
        client = GeminiClient(cfg)
        pipe = cls(client=client, system_prompts=system_prompts, workers=workers)
        pipe._pinned_max_tokens = max_output_tokens
        return pipe

    # Set by ``from_api_key``; None means "use the per-kind default".
    _pinned_max_tokens: Optional[int] = None

    @property
    def model_name(self) -> str:
        return self.client.config.model_name

    def system_prompt(self, kind: str) -> str:
        return self.system_prompts.get(kind) or rewrite_system_prompt_for(kind)

    def _apply_token_budget(self, kind: str) -> None:
        """Point the client's generation config at this kind's ceiling."""
        self.client.config.generation.max_output_tokens = (
            self._pinned_max_tokens or DEFAULT_MAX_OUTPUT_TOKENS[kind]
        )

    # --------------------------- Inference -------------------------- #

    def rewrite_sample(
        self,
        sample: GuardChatSample,
        kind: str = "prompt",
    ) -> RewriteRecord:
        """Rewrite one sample under one representation."""
        kind = normalise_rewrite_kind(kind)
        rec = base_record(sample, kind, model=self.model_name)

        payload = rec.original_turns if kind == "conversation" else rec.original_text
        if not (rec.original_turns if kind == "conversation" else rec.original_text.strip()):
            rec.status = "empty"
            rec.error_kind = "empty_input"
            rec.raw_response = ""
            return rec

        user_msg = build_rewrite_user_message(payload, kind=kind)

        t0 = time.time()
        try:
            resp = self.client.generate(user_msg, self.system_prompt(kind))
            raw_text = resp.text or ""
            rec.attempts = int(resp.attempts)
            rec.blocked = bool(resp.blocked)
            rec.block_reason = resp.block_reason
            rec.finish_reason = resp.finish_reason
            truncated = bool(resp.truncated)
        except Exception as e:  # noqa: BLE001 - record it, do not kill the run
            # The exception text carries the HTTP status and the
            # provider's status name, which is what distinguishes "this
            # sample is unrewritable" from "the account ran out of
            # quota". Keep both the classification and the raw message.
            rec.elapsed_sec = time.time() - t0
            rec.attempts = self.client.config.retries
            rec.status = "error"
            rec.error_kind = classify_error_message(str(e))
            rec.blocked = True
            rec.block_reason = f"{type(e).__name__}: {e}"
            return rec
        rec.elapsed_sec = time.time() - t0
        rec.raw_response = raw_text

        if rec.blocked:
            # Survived every retry - this is a property of the sample,
            # not of the run, and belongs in the results.
            rec.status = "blocked"
            rec.error_kind = "provider_block"
            return rec
        if not raw_text.strip():
            # Empty with no block reason is almost always the thinking
            # budget eating the whole response; see GenerationConfig.
            rec.status = "empty"
            rec.error_kind = "empty_response"
            return rec
        if looks_like_refusal(raw_text):
            rec.status = "refusal"
            rec.error_kind = "model_refusal"
            return rec
        if truncated:
            # Keep the partial text for inspection but do not pass it on
            # as a finished rewrite.
            rec.status = "parse_failed"
            rec.error_kind = "truncated"
            rec.block_reason = rec.block_reason or "MAX_TOKENS"
            return rec

        cleaned = cleanup_rewrite_response(raw_text, kind=kind)
        if kind == "conversation":
            self._finish_conversation(rec, cleaned)
        else:
            self._finish_prompt(rec, cleaned)
        return rec

    @staticmethod
    def _finish_prompt(rec: RewriteRecord, cleaned: str) -> None:
        rec.rewritten_text = cleaned
        if cleaned.strip():
            rec.status = "ok"
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
            return

        rec.rewritten_text = rewritten_conversation_text(turns)
        if rec.rewritten_text.strip():
            rec.status = "ok"
        else:
            rec.status = "empty"
            rec.error_kind = "empty_response"
        rec.was_modified = (
            rec.rewritten_text.strip().lower() != rec.original_text.strip().lower()
        )

    def rewrite_samples(
        self,
        samples: Sequence[GuardChatSample],
        kind: str = "prompt",
        on_result: Optional[Callable[[RewriteRecord], None]] = None,
        progress: bool = True,
    ) -> List[RewriteRecord]:
        """Rewrite a batch, fanning requests out across the thread pool.

        ``on_result`` is invoked once per finished sample (the shared
        evaluation loop uses it to checkpoint) and must be thread-safe.
        Results come back in completion order; the caller restores
        dataset order from the checkpoint.
        """
        kind = normalise_rewrite_kind(kind)
        self._apply_token_budget(kind)
        if not samples:
            return []

        bar = self._progress_bar(len(samples), kind) if progress else None

        def _one(sample: GuardChatSample) -> RewriteRecord:
            rec = self.rewrite_sample(sample, kind=kind)
            if on_result is not None:
                on_result(rec)
            if bar is not None:
                bar.update(1)
            return rec

        try:
            if self.workers == 1:
                return [_one(s) for s in samples]
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                return list(pool.map(_one, samples))
        finally:
            if bar is not None:
                bar.close()

    @staticmethod
    def _progress_bar(total: int, kind: str) -> Any:
        try:
            from tqdm.auto import tqdm
        except ImportError:
            return None
        return tqdm(total=total, desc=f"Gemini[{kind}]")


__all__ = [
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "RewritePipeline",
]
