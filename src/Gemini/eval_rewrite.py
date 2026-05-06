"""CLI: run Gemini 2.5 Flash Task 2 rewriting on GuardChat.

    export GEMINI_API_KEY=...

    python -m src.Gemini.eval_rewrite \
        --test data/guardchat/test.jsonl \
        --model gemini-2.5-flash \
        --output results/gemini_task2.json

Output schema is aligned with ``llama_task2.json`` (plus two Gemini-
specific diagnostics, ``blocked`` and ``block_reason``) so a shared
aggregator can compose Table 2 across baselines.

CLIP cosine similarity uses the SafeGuider-vendored encoder
(``openai/clip-vit-large-patch14``) so similarity numbers are directly
comparable across baselines.
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
from src.Gemini.client import DEFAULT_MODEL_NAME  # noqa: E402
from src.Gemini.rewrite import RewritePipeline  # noqa: E402


def _summarise(records: List[Dict[str, Any]]) -> Dict[str, float]:
    if not records:
        return {}
    sims = [r["clip_similarity"] for r in records if r.get("clip_similarity") is not None]
    modified = sum(1 for r in records if r["was_modified"])
    blocked = sum(1 for r in records if r.get("blocked"))
    return {
        "num_samples": len(records),
        "fraction_modified": modified / len(records),
        "fraction_blocked": blocked / len(records),
        "mean_clip_similarity": float(np.mean(sims)) if sims else 0.0,
        "median_clip_similarity": float(np.median(sims)) if sims else 0.0,
        "mean_elapsed_sec": float(np.mean([r["elapsed_sec"] for r in records])),
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run Gemini 2.5 Flash Task 2 rewriting on GuardChat."
    )
    p.add_argument("--test", required=True, type=str,
                   help="GuardChat test split (JSON/JSONL).")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME,
                   help=f"Gemini model id. Default: {DEFAULT_MODEL_NAME}.")
    p.add_argument("--api-key", type=str, default=None,
                   help="Gemini API key. If unset, reads GEMINI_API_KEY / "
                        "GOOGLE_API_KEY env vars.")
    p.add_argument("--max-output-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--backoff-seconds", type=float, default=2.0)
    p.add_argument("--request-timeout", type=float, default=60.0)
    p.add_argument("--no-relax-safety", action="store_true",
                   help="Keep Gemini's default safety thresholds. Default "
                        "behaviour relaxes them to BLOCK_NONE so the model "
                        "is allowed to read GuardChat's adversarial inputs.")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on number of samples (smoke tests).")
    p.add_argument("--output", type=str, default=None,
                   help="JSON output path. Optional.")
    p.add_argument("--clip-encoder", type=str, default="openai/clip-vit-large-patch14",
                   help="CLIP encoder used for cosine similarity. "
                        "Same as SafeGuider's Task 2 encoder.")
    p.add_argument("--clip-device", type=str, default=None,
                   choices=[None, "cuda", "cpu"],
                   help="Device for the CLIP encoder. Default: auto.")
    p.add_argument("--system-prompt-file", type=str, default=None,
                   help="Override the default rewrite system prompt with a "
                        "text file.")
    args = p.parse_args()

    print(f"Loading test split from {args.test}")
    samples = load_guardchat(args.test)
    if args.limit:
        samples = samples[: int(args.limit)]
    print(f"  -> rewriting {len(samples)} samples via Gemini API")

    custom_prompt = None
    if args.system_prompt_file:
        with open(args.system_prompt_file, "r", encoding="utf-8") as f:
            custom_prompt = f.read()

    pipe = RewritePipeline.from_api_key(
        api_key=args.api_key,
        model_name=args.model,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        relax_safety=not args.no_relax_safety,
        retries=args.retries,
        backoff_seconds=args.backoff_seconds,
        request_timeout=args.request_timeout,
        system_prompt=custom_prompt,
    )

    results = pipe.rewrite_samples(samples)

    print(f"Computing CLIP similarity with {args.clip_encoder!r}")
    from src.SafeGuider import CLIPEncoder
    encoder = CLIPEncoder(model_name=args.clip_encoder, device=args.clip_device)
    originals = [r.original_prompt for r in results]
    rewrites = [r.rewritten_prompt for r in results]
    sims = clip_cosine_similarity(encoder, originals, rewrites)

    records: List[Dict[str, Any]] = []
    for r, sim in zip(results, sims):
        rec = r.to_dict()
        rec["clip_similarity"] = float(sim)
        records.append(rec)

    summary = _summarise(records)
    summary["model"] = args.model
    summary["clip_encoder"] = args.clip_encoder

    print(f"\n[Task 2 summary - {args.model}]")
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
