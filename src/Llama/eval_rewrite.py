"""CLI: run Llama-3.1-8B-Instruct Task 2 rewriting on GuardChat.

    python -m src.Llama.eval_rewrite \
        --test data/guardchat/test.jsonl \
        --weights src/Llama/weights/Llama-3.1-8B-Instruct \
        --dtype bfloat16 \
        --output results/llama_task2.json

Output schema is aligned with ``safeguider_task2.json`` (the
SafeGuider-specific beam-search fields - ``removed_tokens``,
``original_safety``, etc. - are simply absent for LLM-based rewriters)
so a shared aggregator can compose Table 2 across all baselines.

Safe Generation Rate (SGR) is **not** computed here - feed the
``rewritten_prompt`` field to FLUX.1 / Gemini / DALL-E 3 in a separate
pipeline, judge the resulting images, and compute SGR externally. This
script keeps the rewriting step free of T2I dependencies.

CLIP cosine similarity between original and rewritten prompts is
computed via the shared CLIP encoder vendored under
``vendors/SafeGuider/`` - same encoder used by SafeGuider's Task 2
evaluation, so similarity numbers are directly comparable across
baselines.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np  # noqa: E402

# Repo-root sys.path bootstrap.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.utils import clip_cosine_similarity, load_guardchat  # noqa: E402
from src.Llama.model import DEFAULT_LOCAL_DIR  # noqa: E402
from src.Llama.rewrite import RewritePipeline  # noqa: E402


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
        "mean_elapsed_sec": float(np.mean([r["elapsed_sec"] for r in records])),
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run Llama-3.1-8B-Instruct Task 2 rewriting on GuardChat."
    )
    p.add_argument("--test", type=str, default="multimedia-synergy-lab/GuardChat",
                   help="HuggingFace repo id (default 'multimedia-synergy-lab/"
                        "GuardChat') or local JSON/JSONL path.")
    p.add_argument("--split", type=str, default="test",
                   help="HF split when --test is a repo id. Default: test.")
    p.add_argument("--weights", type=str, default=DEFAULT_LOCAL_DIR,
                   help="Local snapshot dir (or HF id). "
                        "Default: src/Llama/weights/Llama-3.1-8B-Instruct.")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32", "int8", "nf4"])
    p.add_argument("--device", type=str, default=None, choices=[None, "cuda", "cpu"])
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on number of samples (smoke tests).")
    p.add_argument("--output", type=str, default=None,
                   help="JSON output path. Optional.")
    p.add_argument("--clip-encoder", type=str, default="openai/clip-vit-large-patch14",
                   help="CLIP encoder used for cosine similarity. "
                        "Same as SafeGuider's Task 2 encoder for cross-baseline "
                        "comparability.")
    p.add_argument("--system-prompt-file", type=str, default=None,
                   help="Override the default rewrite system prompt with a "
                        "text file. Useful for prompt-engineering ablations.")
    args = p.parse_args()

    print(f"Loading test split from {args.test} (split={args.split})")
    samples = load_guardchat(args.test, split=args.split)
    if args.limit:
        samples = samples[: int(args.limit)]
    print(f"  -> rewriting {len(samples)} samples")

    custom_prompt = None
    if args.system_prompt_file:
        with open(args.system_prompt_file, "r", encoding="utf-8") as f:
            custom_prompt = f.read()

    pipe = RewritePipeline.from_pretrained(
        weights=args.weights,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        system_prompt=custom_prompt,
    )

    results = pipe.rewrite_samples(samples)

    # Compute CLIP similarity using the SafeGuider-vendored encoder so
    # numbers are directly comparable to the SafeGuider Task 2 row.
    print(f"Computing CLIP similarity with {args.clip_encoder!r}")
    from src.SafeGuider import CLIPEncoder
    encoder = CLIPEncoder(model_name=args.clip_encoder, device=args.device)
    originals = [r.original_prompt for r in results]
    rewrites = [r.rewritten_prompt for r in results]
    sims = clip_cosine_similarity(encoder, originals, rewrites)

    records: List[Dict[str, Any]] = []
    for r, sim in zip(results, sims):
        rec = r.to_dict()
        rec["clip_similarity"] = float(sim)
        records.append(rec)

    summary = _summarise(records)
    summary["model"] = "Llama-3.1-8B-Instruct"
    summary["dtype"] = args.dtype
    summary["clip_encoder"] = args.clip_encoder

    print("\n[Task 2 summary - Llama-3.1-8B-Instruct]")
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
