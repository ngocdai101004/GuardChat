"""CLI: SBERT semantic similarity over finished Task-2 result files.

Offline metric pass. It reads ``<slug>_task2_<kind>.json`` files that a
rewriter already produced and scores how much meaning survived the
rewrite, writing one sidecar per input plus a combined table::

    python -m src.SBERT.eval_similarity \\
        --results experiment_results/task2/llama \\
                  experiment_results/task2/gemini \\
        --output-dir experiment_results/task2/similarity

``--results`` takes files or folders. A folder is scanned
non-recursively for ``*_task2_prompt.json`` and
``*_task2_conversation.json``, so debug subfolders are not swept up with
the real runs.

Outputs::

    <output-dir>/<slug>_task2_<kind>_sbert.json   per-sample scores + summary
    <output-dir>/sbert_similarity_summary.json    one row per input file

This replaces CLIP as the reported semantic-similarity metric; see
:func:`src.utils.metrics.clip_cosine_similarity` for why. Nothing is
written back into the source result files - they stay exactly as the
rewriter left them, so the metric can be redefined and re-run without
re-spending an API budget.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.utils import resolve_device, resolve_hf_token  # noqa: E402
from src.SBERT.model import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_LOCAL_DIR,
    DEFAULT_MODEL_NAME,
    load_encoder,
)
from src.SBERT.similarity import (  # noqa: E402
    load_result_file,
    output_path,
    score_records,
    summarise_scores,
)

DEFAULT_OUTPUT_DIR = os.path.join(
    _REPO_ROOT, "experiment_results", "task2", "similarity"
)
SUMMARY_FILENAME = "sbert_similarity_summary.json"

# Only the two representations Task 2 rewrites. Matching on the exact
# suffix keeps stray files (checkpoints, sidecars from an earlier pass)
# out of a folder scan.
RESULT_GLOBS = ("*_task2_prompt.json", "*_task2_conversation.json")


def collect_inputs(paths: Sequence[str]) -> List[str]:
    """Expand files and folders into a de-duplicated list of result files."""
    found: List[str] = []
    for path in paths:
        if os.path.isdir(path):
            for pattern in RESULT_GLOBS:
                found.extend(sorted(glob.glob(os.path.join(path, pattern))))
        elif os.path.isfile(path):
            found.append(path)
        else:
            raise FileNotFoundError(f"--results entry not found: {path}")

    unique: List[str] = []
    seen = set()
    for path in found:
        real = os.path.realpath(path)
        if real not in seen:
            seen.add(real)
            unique.append(path)
    return unique


def baseline_slug(path: str) -> str:
    """``.../llama_task2_prompt.json`` -> ``llama``."""
    stem = os.path.basename(path)
    return stem.split("_task2_", 1)[0] if "_task2_" in stem else stem


def score_one(
    encoder,
    path: str,
    output_dir: str,
    per_turn: bool,
    batch_size: Optional[int],
    encoder_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Score one result file and write its sidecar. Returns the table row."""
    kind, records, meta = load_result_file(path)
    slug = baseline_slug(path)
    print(f"\n[{slug} / {kind}] {len(records)} record(s) from {path}")

    started = time.time()
    scored = score_records(
        encoder, records, per_turn=per_turn, batch_size=batch_size,
    )
    elapsed = time.time() - started

    summary = summarise_scores(scored, encoder_meta=encoder_meta)
    summary["elapsed_sec"] = round(elapsed, 2)

    out_path = output_path(output_dir, path)
    payload = {
        "summary": summary,
        "scores": scored,
        "meta": {
            "metric": "sbert_cosine_similarity",
            "baseline": slug,
            "text_kind": kind,
            "source_file": os.path.relpath(path, _REPO_ROOT),
            "source_model": meta.get("model"),
            **encoder_meta,
        },
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print_summary(summary)
    print(f"  -> {out_path}")

    return {
        "baseline": slug,
        "text_kind": kind,
        "source_model": meta.get("model"),
        "source_file": os.path.relpath(path, _REPO_ROOT),
        **{k: summary.get(k) for k in (
            "num_total", "num_scored", "num_unscorable",
            "mean_similarity", "median_similarity", "std_similarity",
            "mean_similarity_penalised", "num_truncated", "fraction_truncated",
        )},
        "mean_similarity_per_turn": (summary.get("per_turn") or {}).get(
            "mean_similarity"),
    }


def print_summary(summary: Dict[str, Any]) -> None:
    for key, val in summary.items():
        if isinstance(val, float):
            print(f"  {key:>28}: {val:.4f}")
        elif isinstance(val, dict):
            pretty = ", ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in sorted(val.items())
            )
            print(f"  {key:>28}: {pretty}")
        else:
            print(f"  {key:>28}: {val}")


def print_table(rows: Sequence[Dict[str, Any]]) -> None:
    """The cross-baseline comparison, in the shape the paper table needs."""
    if not rows:
        return
    header = "%-14s %-13s %7s %7s %9s %11s %9s" % (
        "baseline", "kind", "n", "scored", "mean", "penalised", "per-turn")
    print("\n" + "=" * len(header))
    print("  SBERT semantic similarity  (all-mpnet-base-v2)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        def fmt(key: str) -> str:
            v = r.get(key)
            return f"{v:.4f}" if isinstance(v, float) else "-"
        print("%-14s %-13s %7s %7s %9s %11s %9s" % (
            r["baseline"], r["text_kind"],
            r.get("num_total", "-"), r.get("num_scored", "-"),
            fmt("mean_similarity"), fmt("mean_similarity_penalised"),
            fmt("mean_similarity_per_turn"),
        ))
    print("-" * len(header))
    print("mean      : scored rows only (rewriter failures excluded)")
    print("penalised : failures counted as 0.0 - use this to compare rewriters")
    print("per-turn  : conversations only; turn i vs turn i, averaged")


def main() -> int:
    p = argparse.ArgumentParser(
        description="SBERT semantic similarity for Task-2 rewrites."
    )
    p.add_argument("--results", type=str, nargs="+", required=True,
                   help="Task-2 result JSON files, or folders holding them.")
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                   help=f"Where sidecars land. Default: {DEFAULT_OUTPUT_DIR}.")
    p.add_argument("--weights", type=str, default=DEFAULT_LOCAL_DIR,
                   help="Local snapshot folder, or a HuggingFace id. "
                        f"Default: {DEFAULT_LOCAL_DIR}.")
    p.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME,
                   help=f"Repo id to fetch when --weights is empty. "
                        f"Default: {DEFAULT_MODEL_NAME}.")
    p.add_argument("--device", type=str, default="auto",
                   choices=("auto", "cuda", "cpu", "mps"),
                   help="Torch device. Default: auto.")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                   help=f"Texts per encoder pass. Throughput only; scores do "
                        f"not depend on it. Default: {DEFAULT_BATCH_SIZE}.")
    p.add_argument("--max-seq-length", type=int, default=None,
                   help="Override the checkpoint's token window (384). "
                        "Raising it past 512 is not possible - mpnet's "
                        "positional table stops there.")
    p.add_argument("--no-per-turn", action="store_true",
                   help="Skip the per-turn conversation score. Faster, but "
                        "drops the one number that is immune to the 384-token "
                        "window.")
    p.add_argument("--token", type=str, default=None,
                   help="HuggingFace token for the (open) model download.")
    args = p.parse_args()

    inputs = collect_inputs(args.results)
    if not inputs:
        print("No Task-2 result files matched --results.", file=sys.stderr)
        return 1

    device = resolve_device(args.device)
    print(f"[sbert] device: {device}")
    print(f"[sbert] scoring {len(inputs)} file(s)")

    encoder = load_encoder(
        model_path=args.weights,
        repo_id=args.model_name,
        device=device,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        token=resolve_hf_token(args.token),
    )
    encoder_meta = {
        "encoder": args.model_name,
        "encoder_path": encoder.model_path,
        "max_seq_length": encoder.max_seq_length,
        "pooling": "mean",
        "normalised": True,
    }
    print(f"[sbert] max_seq_length: {encoder.max_seq_length}")

    rows = [
        score_one(
            encoder, path, args.output_dir,
            per_turn=not args.no_per_turn,
            batch_size=args.batch_size,
            encoder_meta=encoder_meta,
        )
        for path in inputs
    ]

    print_table(rows)

    summary_path = os.path.join(args.output_dir, SUMMARY_FILENAME)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"metric": "sbert_cosine_similarity",
                   **encoder_meta, "rows": rows}, f, indent=2, ensure_ascii=False)
    print(f"\nCombined summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
