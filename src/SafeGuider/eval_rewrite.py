"""CLI: run SafeGuider Task 2 (NSFW concept removal via prompt rewriting).

Output schema is designed to feed two downstream evaluations:
    1. CLIP cosine similarity between original and rewritten prompts
       (computed here using the same CLIP text encoder used by SafeGuider).
    2. Safe Generation Rate (SGR) against FLUX.1, Gemini, and DALL-E 3 -
       run separately by feeding ``rewritten_prompt`` to each T2I system
       and judging the resulting images. We keep this script T2I-free so
       it has no proprietary-API dependencies.

Usage:
    python -m src.SafeGuider.eval_rewrite \
        --test data/guardchat/test.jsonl \
        --weights vendors/SafeGuider/weights/SD1.4_safeguider.pt \
        --output results/safeguider_task2.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

# Repo-root sys.path bootstrap.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402

from src.SafeGuider import (  # noqa: E402
    DEFAULT_BEAM_WIDTH,
    DEFAULT_MAX_DEPTH,
    DEFAULT_SAFETY_THRESHOLD,
    DEFAULT_SIMILARITY_FLOOR,
)
from src.SafeGuider.data import load_guardchat  # noqa: E402
from src.SafeGuider.metrics import clip_cosine_similarity  # noqa: E402
from src.SafeGuider.rewrite import RewritePipeline  # noqa: E402


def _summarise(records: List[Dict[str, Any]]) -> Dict[str, float]:
    if not records:
        return {}
    sims = [r["clip_similarity"] for r in records if r.get("clip_similarity") is not None]
    modified = sum(1 for r in records if r["was_modified"])
    return {
        "num_samples": len(records),
        "fraction_modified": modified / len(records),
        "mean_clip_similarity": float(np.mean(sims)) if sims else 0.0,
        "median_clip_similarity": float(np.median(sims)) if sims else 0.0,
        "mean_safeguider_similarity": float(
            np.mean([r["safeguider_similarity"] for r in records])
        ),
        "mean_modified_safety": float(
            np.mean([r["modified_safety"] for r in records])
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Run SafeGuider Task 2 (rewrite) on GuardChat.")
    p.add_argument("--test", required=True, type=str,
                   help="GuardChat test split (JSON/JSONL).")
    p.add_argument("--weights", required=True, type=str,
                   help="SafeGuider binary classifier .pt path "
                        "(e.g. vendors/SafeGuider/weights/SD1.4_safeguider.pt).")
    p.add_argument("--encoder-model", type=str, default="openai/clip-vit-large-patch14")
    p.add_argument("--device", type=str, default=None, choices=[None, "cuda", "cpu"])

    # Beam-search hyper-params (mirror vendored defaults).
    p.add_argument("--beam-width", type=int, default=DEFAULT_BEAM_WIDTH)
    p.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    p.add_argument("--safety-threshold", type=float, default=DEFAULT_SAFETY_THRESHOLD)
    p.add_argument("--similarity-floor", type=float, default=DEFAULT_SIMILARITY_FLOOR)

    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on number of samples (for smoke tests).")
    p.add_argument("--output", type=str, default=None,
                   help="Where to dump the JSON of rewrite results.")
    args = p.parse_args()

    print(f"Loading test split from {args.test}")
    samples = load_guardchat(args.test)
    if args.limit:
        samples = samples[: int(args.limit)]
    print(f"  -> rewriting {len(samples)} samples")

    pipe = RewritePipeline(
        weights=args.weights,
        encoder_model=args.encoder_model,
        device=args.device,
        beam_width=args.beam_width,
        max_depth=args.max_depth,
        safety_threshold=args.safety_threshold,
        similarity_floor=args.similarity_floor,
    )

    results = pipe.rewrite_samples(samples)

    # CLIP cosine similarity on (original, rewritten) pairs - shared encoder.
    originals = [r.original_prompt for r in results]
    rewrites = [r.rewritten_prompt for r in results]
    clip_sims = clip_cosine_similarity(pipe.encoder, originals, rewrites)

    records: List[Dict[str, Any]] = []
    for r, sim in zip(results, clip_sims):
        rec = r.to_dict()
        rec["clip_similarity"] = float(sim)
        records.append(rec)

    summary = _summarise(records)
    print("\n[Task 2 summary]")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:>26}: {v:.4f}")
        else:
            print(f"  {k:>26}: {v}")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        payload = {"summary": summary, "rewrites": records}
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nSaved rewrite results -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
