"""CLI: run local Llama-3.1-8B-Instruct Task-2 rewriting on GuardChat.

    python -m src.Llama.eval_rewrite \
        --test build_dataset/dataset/final_df_test.json \
        --text-kind all \
        --output-dir experiment_results/task2/llama

``--text-kind all`` writes one JSON file per input representation:

    llama_task2_prompt.json         enhanced prompt      -> P_safe
    llama_task2_conversation.json   multi-turn dialogue  -> P_safe

Output schema is the shared Task-2 schema
(:class:`src.utils.RewriteRecord`), identical to what the Gemini
rewriter produces, so a single aggregator can compose Table 2 across
every row.

``meta-llama/Llama-3.1-8B-Instruct`` is a **gated** repo: accept the
licence on the model page, then put ``HF_TOKEN=hf_...`` in the repo-root
``.env`` (or export it). Weights land in
``src/Llama/weights/Llama-3.1-8B-Instruct`` (~16 GB) on first use.

This step only produces P_safe. Safe Generation Rate and SBERT semantic
similarity are computed downstream from the ``rewritten_text`` field.

Long runs are checkpointed to ``<output>.partial.jsonl`` after every
batch; re-running with ``--resume`` skips whatever is already there.
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
    DTYPE_CHOICES,
    REWRITE_KINDS,
    load_guardchat,
    normalise_rewrite_kind,
    print_summary,
    rewrite_kind,
    save_rewrite_kind,
)
from src.Llama.model import (  # noqa: E402
    DEFAULT_LOCAL_DIR,
    DEFAULT_MODEL_NAME,
    MAX_NEW_TOKENS_CAP,
)
from src.Llama.rewrite import RewritePipeline  # noqa: E402


SLUG = "llama"
DEFAULT_OUTPUT_DIR = os.path.join(_REPO_ROOT, "experiment_results", "task2", SLUG)

# Failures that say something about the machine rather than about the
# sample. A run carrying any of these is incomplete and should be
# resumed, not reported.
INFRASTRUCTURE_ERROR_KINDS = frozenset({
    "oom", "auth", "model_not_found", "network", "timeout", "unknown",
})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run Llama-3.1-8B-Instruct Task-2 rewriting on GuardChat."
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
                        "files) with the model loaded once.")
    p.add_argument("--weights", type=str, default=DEFAULT_LOCAL_DIR,
                   help="Local snapshot dir (downloaded on first use) or an "
                        f"HF id. Default: src/Llama/weights/"
                        f"Llama-3.1-8B-Instruct, auto-populated from "
                        f"{DEFAULT_MODEL_NAME}.")
    p.add_argument("--no-auto-download", action="store_true",
                   help="Fail instead of fetching the snapshot when the "
                        "local weights folder is empty.")
    p.add_argument("--dtype", type=str, default="auto", choices=list(DTYPE_CHOICES),
                   help="Weight dtype. 'auto' = float32 on CPU, bfloat16 "
                        "elsewhere. int8/nf4 need bitsandbytes (CUDA only).")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cuda", "mps", "cpu"],
                   help="Compute device. 'auto' prefers cuda > mps > cpu.")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Sequences generated together. The main lever on "
                        "throughput and on memory. Lower it before lowering "
                        "precision; 'conversation' needs far more KV cache "
                        "than 'prompt'. Default: 8.")
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="Pin the generation window. Default is adaptive - "
                        "derived per batch from the longest source text, "
                        f"capped at {MAX_NEW_TOKENS_CAP}. A rewrite is about "
                        "as long as its input, so a fixed worst-case window "
                        "would waste most of every short batch.")
    p.add_argument("--retries", type=int, default=3,
                   help="Total attempts per sample (not extra attempts). "
                        "Attempts after the first switch from greedy to "
                        "sampled decoding - re-running a greedy generation "
                        "is bit-for-bit identical, so it could only "
                        "reproduce the same failure. Only refusals, empty "
                        "answers and unparseable turn blocks are retried. "
                        "Default: 3.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap the number of samples (smoke tests).")
    p.add_argument("--token", type=str, default=None,
                   help="HuggingFace token override. Default reads HF_TOKEN "
                        "from the environment or the repo-root .env file. "
                        "Required: this repo is gated.")
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                   help=f"Where the per-kind JSON files land. "
                        f"Default: {DEFAULT_OUTPUT_DIR}.")
    p.add_argument("--resume", action="store_true",
                   help="Reuse any '<output>.partial.jsonl' checkpoint "
                        "instead of re-generating from scratch.")
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

    pipe = RewritePipeline.from_pretrained(
        weights=args.weights,
        dtype=args.dtype,
        device=args.device,
        token=args.token,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        retries=args.retries,
        auto_download=not args.no_auto_download,
        system_prompts=system_prompts or None,
    )
    print(f"Loaded {args.weights} on {pipe.model.device} "
          f"({pipe.model.dtype_name})")
    print(f"Batch size {args.batch_size}, retries {args.retries} "
          f"(attempt 1 greedy, later attempts sampled)")
    if system_prompts:
        print(f"  system prompt overridden for: {sorted(system_prompts)}")

    meta: Dict[str, object] = {
        "task": "task2_rewrite",
        "model": DEFAULT_MODEL_NAME,
        "weights": args.weights,
        "dtype": pipe.model.dtype_name,
        "device": str(pipe.model.device),
        "test": args.test,
        "num_samples": len(samples),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens or MAX_NEW_TOKENS_CAP,
        "retries": args.retries,
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

        failed = summary.get("num_samples", 0) - summary.get("num_usable", 0)
        if failed:
            errs: Dict[str, int] = summary.get("error_kind_counts") or {}
            print(f"  WARNING: {failed}/{summary['num_samples']} samples "
                  f"produced no usable rewrite ({errs or summary['status_counts']}). "
                  f"These rows have an empty 'rewritten_text' and count as "
                  f"failures in SGR.")
            infra = {k: v for k, v in errs.items()
                     if k in INFRASTRUCTURE_ERROR_KINDS}
            if infra:
                print(f"  ACTION NEEDED: {sum(infra.values())} of those are "
                      f"machine failures, not model behaviour ({infra}). "
                      f"Lower --batch-size for 'oom', fix the cause "
                      f"otherwise, then re-run with --resume.")
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
