"""CLI: run SafeGuider Task-2 rewriting (safety-aware beam search) on GuardChat.

    python -m src.SafeGuider.eval_rewrite \
        --test build_dataset/dataset/final_df_test.json \
        --text-kind all \
        --output-dir experiment_results/task2/safeguider

``--text-kind all`` writes one JSON file per input representation:

    safeguider_task2_prompt.json        enhanced prompt      -> P_safe
    safeguider_task2_conversation.json  multi-turn dialogue  -> P_safe

Output schema is the shared Task-2 schema
(:class:`src.utils.RewriteRecord`), identical to what the Gemini and
Llama rewriters produce, so one aggregator composes Table 2 across every
row. SafeGuider's own diagnostics — recognizer scores, deleted words,
CLIP truncation — ride along in the per-record ``extra`` object.

Needs the upstream recognizer checkpoint at
``vendors/SafeGuider/weights/SD1.4_safeguider.pt`` (from the SafeGuider
release) and the CLIP text encoder, which is fetched into
``vendors/SafeGuider/weights/clip-vit-large-patch14/`` on first use.

This step produces P_safe only. Safe Generation Rate and SBERT semantic
similarity are computed downstream from the ``rewritten_text`` field.

Long runs are checkpointed to ``<output>.partial.jsonl`` after every
sample; re-running with ``--resume`` skips whatever is already there.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Tuple

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
from src.SafeGuider import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_BEAM_WIDTH,
    DEFAULT_PATIENCE,
    DEFAULT_MAX_DEPTH,
    DEFAULT_SAFETY_THRESHOLD,
    DEFAULT_SIMILARITY_FLOOR,
)
from src.SafeGuider.rewrite import (  # noqa: E402
    DEFAULT_ENCODER_MODEL,
    DEFAULT_MODEL_NAME,
    DEFAULT_WEIGHTS,
    GATE_MODES,
    RewritePipeline,
)


SLUG = "safeguider"
DEFAULT_OUTPUT_DIR = os.path.join(_REPO_ROOT, "experiment_results", "task2", SLUG)

# Failures that say something about the machine rather than about the
# sample. A run carrying any of these is incomplete and should be
# resumed, not reported. There is no provider here, so the API-side
# kinds (quota, server_error, ...) cannot occur.
INFRASTRUCTURE_ERROR_KINDS = frozenset({"oom", "network", "timeout", "unknown"})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run SafeGuider beam-search Task-2 rewriting on GuardChat."
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
                        "files) with the encoder loaded once.")
    p.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS,
                   help="Upstream binary recognizer checkpoint. "
                        "Default: vendors/SafeGuider/weights/"
                        "SD1.4_safeguider.pt.")
    p.add_argument("--encoder-model", type=str, default=DEFAULT_ENCODER_MODEL,
                   help="Text encoder the recognizer was trained on. Must "
                        "match the checkpoint: clip-vit-large-patch14 for "
                        "SD1.4 (768), OpenCLIP ViT-H/14 for SD2.1 (1024), "
                        "T5-XXL for Flux (4096).")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cuda", "cpu"],
                   help="Compute device. 'auto' picks cuda when available. "
                        "Accepted (and equal to 'auto') so a shared "
                        "DEVICE=auto in the environment does not have to be "
                        "special-cased per baseline.")

    p.add_argument("--gate", type=str, default="recognizer", choices=list(GATE_MODES),
                   help="'recognizer' (default) reproduces the published "
                        "pipeline: classify first, rewrite only what comes "
                        "back unsafe, so prompts the recognizer misses pass "
                        "through untouched. 'always' skips the gate and "
                        "beam-searches every row, measuring the rewriter "
                        "alone.")

    # Beam-search hyper-parameters (upstream defaults).
    p.add_argument("--beam-width", type=int, default=DEFAULT_BEAM_WIDTH,
                   help=f"Candidates kept per depth. Default: {DEFAULT_BEAM_WIDTH}.")
    p.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH,
                   help="Most words the search may delete. Capped at "
                        f"len(words) - 1. Default: {DEFAULT_MAX_DEPTH}.")
    p.add_argument("--safety-threshold", type=float, default=DEFAULT_SAFETY_THRESHOLD,
                   help="P[safe] a candidate must reach to qualify. "
                        f"Default: {DEFAULT_SAFETY_THRESHOLD}.")
    p.add_argument("--similarity-floor", type=float, default=DEFAULT_SIMILARITY_FLOOR,
                   help="Minimum CLIP-EOS cosine to the original. "
                        f"Default: {DEFAULT_SIMILARITY_FLOOR}.")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                   help="Candidate strings per encoder forward pass. Pure "
                        "throughput knob - results do not depend on it. "
                        f"Lower it on a small GPU. Default: {DEFAULT_BATCH_SIZE}.")

    p.add_argument("--patience", type=int, default=DEFAULT_PATIENCE,
                   help="Stop a search after this many consecutive depths "
                        "fail to raise the best safety score. 0 (default) "
                        "is upstream behaviour: always run to --max-depth. "
                        "NOT an upstream parameter - it changes results and "
                        "must be reported. Prefer it to a low --max-depth: "
                        "prompts that do succeed often need depth 14-24, so "
                        "capping depth discards successes to save time on "
                        "hopeless cases, while patience abandons only the "
                        "plateaus. Per-sample effect is in extra.halt_reason.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap the number of samples (smoke tests).")
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                   help=f"Where the per-kind JSON files land. "
                        f"Default: {DEFAULT_OUTPUT_DIR}.")
    p.add_argument("--resume", action="store_true",
                   help="Reuse any '<output>.partial.jsonl' checkpoint "
                        "instead of re-searching from scratch.")
    p.add_argument("--keep-checkpoint", action="store_true",
                   help="Keep the .partial.jsonl file after a successful run.")
    p.add_argument("--verbose", action="store_true",
                   help="Log encoder token counts and per-depth beam state.")
    return p


def _report_failures(summary: Dict[str, Any]) -> None:
    """Rows that produced no usable P_safe, split by whose fault it is."""
    failed = summary.get("num_samples", 0) - summary.get("num_usable", 0)
    if not failed:
        return

    errs: Dict[str, int] = summary.get("error_kind_counts") or {}
    print(f"  WARNING: {failed}/{summary['num_samples']} samples produced "
          f"no usable rewrite ({errs or summary['status_counts']}). These "
          f"rows have an empty 'rewritten_text' and count as failures "
          f"in SGR.")

    infra = {k: v for k, v in errs.items() if k in INFRASTRUCTURE_ERROR_KINDS}
    if infra:
        print(f"  ACTION NEEDED: {sum(infra.values())} of those are machine "
              f"failures, not method behaviour ({infra}). Lower "
              f"--batch-size for 'oom', fix the cause otherwise, then "
              f"re-run with --resume.")


def _diagnostic_units(
    kind: str,
    extras: List[Dict[str, Any]],
) -> Tuple[str, int, int, int]:
    """``(unit, total, gated, truncated)`` over the rows that actually ran.

    A conversation's diagnostics are per turn, a prompt's are per row, so
    the unit differs - but numerator and denominator must always come
    from the same population.
    """
    if kind == "conversation":
        return (
            "turns",
            sum(len(e.get("turns") or []) for e in extras),
            sum(int(e.get("num_turns_gated_safe") or 0) for e in extras),
            sum(int(e.get("num_turns_truncated") or 0) for e in extras),
        )
    return (
        "prompts",
        len(extras),
        sum(1 for e in extras if e.get("gated_safe")),
        sum(1 for e in extras if e.get("truncated")),
    )


def _outcome_counts(
    kind: str,
    extras: List[Dict[str, Any]],
    key: str = "outcome",
) -> Dict[str, int]:
    """Tally one per-search field, flattened to turn level for conversations."""
    counts: Dict[str, int] = {}
    for e in extras:
        rows = e.get("turns") if kind == "conversation" else [e]
        for row in rows or []:
            label = str(row.get(key)
                        or ("gated_safe" if row.get("gated_safe") else "n/a"))
            counts[label] = counts.get(label, 0) + 1
    return counts


def _report(kind: str, summary: Dict[str, Any], records: List[Dict[str, Any]]) -> None:
    """Print the shared summary plus the SafeGuider-specific caveats."""
    print_summary(summary)
    _report_failures(summary)

    # Rows that failed carry no diagnostics at all. Counting them in the
    # denominators below - while they can never contribute to a numerator
    # - would quietly deflate every rate printed here.
    extras = [e for e in (r.get("extra") for r in records) if e]
    if not extras:
        return

    unit, total, gated, truncated = _diagnostic_units(kind, extras)
    if total:
        # The headline caveat: these rows reached the T2I model exactly as
        # the attacker wrote them, so they are recognizer misses, not
        # rewrites, and they will show up as SGR failures.
        print(f"  gate: {gated}/{total} {unit} ({gated / total:.1%}) were judged "
              f"SAFE by the recognizer and passed through unmodified.")
        print(f"  encoder window: {truncated}/{total} {unit} "
              f"({truncated / total:.1%}) exceed CLIP's 77 tokens, so their "
              f"tail was never searched.")

    outcomes = _outcome_counts(kind, extras)
    if outcomes:
        print(f"  search outcome: "
              f"{', '.join(f'{k}={v}' for k, v in sorted(outcomes.items()))}")

    halts = _outcome_counts(kind, extras, key="halt_reason")
    if halts:
        # Counts, not settings - see halt_reason in BeamSearchResult.
        print(f"  halted because: "
              f"{', '.join(f'{k} x{v}' for k, v in sorted(halts.items()))}")


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

    pipe = RewritePipeline.from_weights(
        weights=args.weights,
        encoder_model=args.encoder_model,
        # CLIPEncoder auto-detects on None.
        device=None if args.device == "auto" else args.device,
        beam_width=args.beam_width,
        max_depth=args.max_depth,
        safety_threshold=args.safety_threshold,
        similarity_floor=args.similarity_floor,
        batch_size=args.batch_size,
        patience=args.patience,
        gate=args.gate,
        verbose=args.verbose,
    )
    print(f"Loaded {args.weights}")
    print(f"  encoder {args.encoder_model} on {pipe.encoder.device} "
          f"(dim {pipe.encoder.hidden_size}, window "
          f"{pipe.encoder.max_length} tokens)")
    print(f"  beam width {args.beam_width}, max depth {args.max_depth}, "
          f"safety >= {args.safety_threshold}, similarity >= "
          f"{args.similarity_floor}, gate={args.gate}")
    if args.patience:
        print(f"  patience {args.patience}: searches that stall for that "
              f"many depths give up early. This is NOT upstream behaviour "
              f"- report it with the results.")

    meta: Dict[str, object] = {
        "task": "task2_rewrite",
        "model": DEFAULT_MODEL_NAME,
        "weights": args.weights,
        "encoder_model": args.encoder_model,
        "device": str(pipe.encoder.device),
        "test": args.test,
        "num_samples": len(samples),
        "gate": args.gate,
        "beam_width": args.beam_width,
        "max_depth": args.max_depth,
        "safety_threshold": args.safety_threshold,
        "similarity_floor": args.similarity_floor,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "conversation_strategy": "per_turn",
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
        _report(kind, res["summary"], res["rewrites"])

    print("\nDone. Files written:")
    for pth in written:
        print(f"  {pth}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
