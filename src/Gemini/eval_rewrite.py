"""CLI: run Gemini 2.5 Flash Task-2 rewriting on GuardChat (zero-shot).

    # GEMINI_API_KEY is read from the repo-root .env if not exported
    python -m src.Gemini.eval_rewrite \
        --test build_dataset/dataset/final_df_test.json \
        --text-kind all \
        --output-dir experiment_results/task2/gemini

``--text-kind all`` writes one JSON file per input representation:

    gemini_task2_prompt.json        enhanced prompt      -> P_safe
    gemini_task2_conversation.json  multi-turn dialogue  -> P_safe

Output schema is the shared Task-2 schema
(:class:`src.utils.RewriteRecord`), identical to what the local Llama
rewriter and SafeGuider produce, so a single aggregator can compose
Table 2 across every row.

This step only produces P_safe. The two paper metrics are computed
downstream from the ``rewritten_text`` field:

    * Safe Generation Rate - feed P_safe to FLUX.1 / Gemini 2.5 Flash
      Image / DALL-E 3 and score the images with the safety gate. Rows
      whose ``status`` is not ``ok`` never reach a T2I model and count as
      failures.
    * Semantic similarity - SBERT cosine between ``original_text`` and
      ``rewritten_text``.

Long runs are checkpointed to ``<output>.partial.jsonl`` after every
sample; re-running with ``--resume`` skips whatever is already there,
which matters because these are paid API calls.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

# Repo-root sys.path bootstrap.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.utils import (  # noqa: E402
    REWRITE_KINDS,
    load_guardchat,
    normalise_rewrite_kind,
    print_summary,
    rewrite_kind,
    save_rewrite_kind,
)
from src.Gemini.client import DEFAULT_MODEL_NAME  # noqa: E402
from src.Gemini.rewrite import DEFAULT_MAX_OUTPUT_TOKENS, RewritePipeline  # noqa: E402


SLUG = "gemini"
DEFAULT_OUTPUT_DIR = os.path.join(_REPO_ROOT, "experiment_results", "task2", SLUG)

# Failures that say something about the account or the network rather
# than about the sample. A run carrying any of these is incomplete and
# should be resumed, not reported.
INFRASTRUCTURE_ERROR_KINDS = frozenset({
    "quota", "auth", "model_not_found", "timeout", "network", "server_error",
})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run Gemini 2.5 Flash Task-2 rewriting on GuardChat."
    )
    p.add_argument("--test", type=str,
                   default=os.path.join(_REPO_ROOT, "build_dataset", "dataset",
                                        "final_df_test.json"),
                   help="Local JSON/JSONL path or a HuggingFace repo id. "
                        "Default: build_dataset/dataset/final_df_test.json.")
    p.add_argument("--split", type=str, default="test",
                   help="HF split when --test is a repo id. Default: test.")
    p.add_argument("--text-kind", type=str, default="all",
                   choices=list(REWRITE_KINDS) + ["single", "all"],
                   help="Input representation to rewrite. 'all' runs "
                        "prompt and conversation in one go (two output "
                        "files).")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME,
                   help=f"Gemini model id. Default: {DEFAULT_MODEL_NAME}.")
    p.add_argument("--api-key", type=str, default=None,
                   help="Gemini API key. Default reads GEMINI_API_KEY / "
                        "GOOGLE_API_KEY from the environment or the "
                        "repo-root .env file.")
    p.add_argument("--max-output-tokens", type=int, default=None,
                   help="Pin both representations to this ceiling. Default "
                        f"is per-kind: {DEFAULT_MAX_OUTPUT_TOKENS}. A "
                        "conversation rewrite has to re-emit every turn, so "
                        "it needs far more room than a single prompt.")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Greedy by default - a sanitised prompt should be "
                        "reproducible, not creative.")
    p.add_argument("--thinking-budget", type=int, default=0,
                   help="Gemini 2.5 reasoning-token budget. 0 (default) "
                        "disables it: thinking tokens come out of "
                        "--max-output-tokens and can starve the answer. "
                        "Pass -1 to leave the model default alone.")
    p.add_argument("--workers", type=int, default=4,
                   help="Concurrent API requests. Lower this if the key is "
                        "rate-limited (free tier is a few RPM). Default: 4.")
    p.add_argument("--retries", type=int, default=3,
                   help="Total attempts per sample (not extra attempts). "
                        "Covers transient errors (429 / 5xx / timeouts), "
                        "empty responses and - unless --no-retry-blocked - "
                        "safety blocks. Default: 3.")
    p.add_argument("--no-retry-blocked", action="store_true",
                   help="Do not re-send a request the safety filter killed. "
                        "By default a block is retried like any other "
                        "failure; a sample still blocked after every "
                        "attempt is recorded with error_kind "
                        "'provider_block' and an empty rewrite.")
    p.add_argument("--backoff-seconds", type=float, default=2.0,
                   help="Base delay for exponential backoff. Default: 2.0.")
    p.add_argument("--request-timeout", type=float, default=120.0,
                   help="Per-request HTTP timeout in seconds. Default: 120.")
    p.add_argument("--no-relax-safety", action="store_true",
                   help="Keep Gemini's default safety thresholds. By "
                        "default they are relaxed to BLOCK_NONE so the "
                        "model is allowed to read GuardChat's adversarial "
                        "inputs; without this it refuses many of them and "
                        "the rewriter cannot be measured.")
    p.add_argument("--no-preflight", action="store_true",
                   help="Skip the one-call check that the model id works. "
                        "The check costs a few tokens and turns a retired "
                        "model id into an immediate error instead of 1,000 "
                        "identical failures.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap the number of samples (smoke tests).")
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                   help=f"Where the per-kind JSON files land. "
                        f"Default: {DEFAULT_OUTPUT_DIR}.")
    p.add_argument("--resume", action="store_true",
                   help="Reuse any '<output>.partial.jsonl' checkpoint "
                        "instead of re-spending API calls.")
    p.add_argument("--keep-checkpoint", action="store_true",
                   help="Keep the .partial.jsonl file after a successful run.")
    p.add_argument("--system-prompt-file", type=str, default=None,
                   help="Override the single-prompt system prompt with a "
                        "text file.")
    p.add_argument("--system-prompt-conversation-file", type=str, default=None,
                   help="Override the conversation system prompt with a "
                        "text file. Must still ask for [Tn] markers, or the "
                        "turn parser will report parse_failed.")
    return p


def _read_optional(path: Optional[str] = None) -> str:
    if not path:
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main() -> int:
    args = build_parser().parse_args()

    print(f"Loading test split from {args.test} (split={args.split})")
    samples = load_guardchat(args.test, split=args.split)
    if args.limit:
        samples = samples[: int(args.limit)]
    print(f"  -> {len(samples)} samples")

    kinds: List[str]
    if args.text_kind == "all":
        kinds = list(REWRITE_KINDS)
    else:
        kinds = [normalise_rewrite_kind(args.text_kind)]

    system_prompts: Dict[str, str] = {}
    override = _read_optional(args.system_prompt_file)
    if override:
        system_prompts["prompt"] = override
    override = _read_optional(args.system_prompt_conversation_file)
    if override:
        system_prompts["conversation"] = override

    pipe = RewritePipeline.from_api_key(
        api_key=args.api_key,
        model_name=args.model,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        # -1 is the CLI's way of saying "leave the model default alone".
        thinking_budget=None if args.thinking_budget < 0 else args.thinking_budget,
        relax_safety=not args.no_relax_safety,
        retries=args.retries,
        retry_blocked=not args.no_retry_blocked,
        backoff_seconds=args.backoff_seconds,
        request_timeout=args.request_timeout,
        workers=args.workers,
        system_prompts=system_prompts or None,
    )
    print(f"Rewriter: {pipe.model_name} "
          f"(temperature={args.temperature}, workers={args.workers}, "
          f"safety={'default' if args.no_relax_safety else 'BLOCK_NONE'})")
    if system_prompts:
        print(f"  system prompt overridden for: {sorted(system_prompts)}")

    if not args.no_preflight:
        try:
            pipe.client.preflight()
        except RuntimeError as e:
            print(f"\nERROR: {e}", file=sys.stderr)
            return 2

    meta: Dict[str, object] = {
        "task": "task2_rewrite",
        "model": pipe.model_name,
        "test": args.test,
        "num_samples": len(samples),
        "temperature": args.temperature,
        "thinking_budget": args.thinking_budget,
        "max_output_tokens": args.max_output_tokens or DEFAULT_MAX_OUTPUT_TOKENS,
        "relax_safety": not args.no_relax_safety,
        "retries": args.retries,
        "retry_blocked": not args.no_retry_blocked,
        "workers": args.workers,
        "system_prompt_overridden": sorted(system_prompts) or None,
    }

    written: List[str] = []
    for kind in kinds:
        print(f"\n=== {kind} ===")
        res = rewrite_kind(
            lambda pending, k, cb: pipe.rewrite_samples(pending, kind=k,
                                                        on_result=cb),
            samples, kind, args.output_dir, SLUG, resume=args.resume,
        )
        out_path = save_rewrite_kind(res, kind, meta,
                                     keep_checkpoint=args.keep_checkpoint)
        written.append(out_path)
        print(f"Saved -> {out_path}")

        summary = res["summary"]
        print_summary(summary)

        # A rewriter that returns nothing usable cannot be measured; a
        # rewriter that returns a tenth of the text is on its way to a
        # perfect Safe Generation Rate and a worthless similarity score.
        # Both are worth seeing before paying to generate images.
        failed = summary.get("num_samples", 0) - summary.get("num_usable", 0)
        if failed:
            errs: Dict[str, int] = summary.get("error_kind_counts") or {}
            print(f"  WARNING: {failed}/{summary['num_samples']} samples "
                  f"produced no usable rewrite ({errs or summary['status_counts']}). "
                  f"These rows have an empty 'rewritten_text' and count as "
                  f"failures in SGR.")
            # Distinguish a rewriter that could not handle the sample
            # from a run that hit an account limit. The first is a
            # result; the second means re-run with --resume once the
            # limit clears, or the numbers understate the rewriter.
            infra = {k: v for k, v in errs.items()
                     if k in INFRASTRUCTURE_ERROR_KINDS}
            if infra:
                print(f"  ACTION NEEDED: {sum(infra.values())} of those are "
                      f"infrastructure failures, not model behaviour "
                      f"({infra}). Fix the cause and re-run with --resume - "
                      f"finished samples are not re-billed.")
        ratio = summary.get("mean_length_ratio")
        if ratio is not None and ratio < 0.5:
            print(f"  WARNING: rewrites average {ratio:.0%} of the original "
                  f"length - the model may be deleting rather than "
                  f"sanitising. Check semantic similarity before reporting.")
        if kind == "conversation":
            match = summary.get("turn_count_match_rate")
            if match is not None and match < 1.0:
                print(f"  NOTE: turn count preserved for {match:.1%} of "
                      f"conversations ({summary.get('turn_parse_counts')}). "
                      f"Mismatched rows are still stored; see 'turn_parse'.")

    print("\nDone. Files written:")
    for pth in written:
        print(f"  {pth}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
