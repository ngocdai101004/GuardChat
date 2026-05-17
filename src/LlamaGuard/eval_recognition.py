"""CLI: evaluate Llama-Guard-3-8B on GuardChat Task 1 (zero-shot).

    python -m src.LlamaGuard.eval_recognition \
        --test data/guardchat/test.jsonl \
        --weights src/LlamaGuard/weights/Llama-Guard-3-8B \
        --mode native \
        --dtype bfloat16 \
        --text-kind both \
        --output results/llamaguard_task1.json

Output schema is identical to the supervised baselines
(``safeguider_task1.json``, ``bilstm_task1.json``, ``bert_task1.json``)
so a single aggregator can compose Table 1 across all five baselines.

There is no ``train_recognition.py`` for this baseline - Llama-Guard is
used strictly zero-shot.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

# Repo-root sys.path bootstrap.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.utils import load_guardchat, summarise_recognition  # noqa: E402
from src.LlamaGuard.model import DEFAULT_LOCAL_DIR  # noqa: E402
from src.LlamaGuard.recognition import RecognitionPipeline  # noqa: E402


def _evaluate_one_kind(pipe: RecognitionPipeline, samples, kind: str):
    preds = pipe.predict_samples(samples, kind=kind)
    y_true = [list(s.label_vector) for s in samples]
    y_pred = [list(p.multi_label) for p in preds]
    metrics = summarise_recognition(y_true, y_pred)
    return preds, metrics


def main() -> int:
    p = argparse.ArgumentParser(
        description="Evaluate Llama-Guard-3-8B on GuardChat Task 1."
    )
    p.add_argument("--test", type=str, default="multimedia-synergy-lab/GuardChat",
                   help="HuggingFace repo id (default 'multimedia-synergy-lab/"
                        "GuardChat') or local JSON/JSONL path.")
    p.add_argument("--split", type=str, default="test",
                   help="HF split when --test is a repo id. Default: test.")
    p.add_argument("--weights", type=str, default=DEFAULT_LOCAL_DIR,
                   help="Local snapshot dir (or HF id). "
                        "Default: src/LlamaGuard/weights/Llama-Guard-3-8B.")
    p.add_argument("--mode", type=str, default="native",
                   choices=["native", "custom"],
                   help="`native`: use S1-S14 + post-hoc mapping (no shocking). "
                        "`custom`: pass GuardChat 6-category taxonomy directly.")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32", "int8", "nf4"],
                   help="Weight dtype. int8/nf4 require bitsandbytes >= 0.43.")
    p.add_argument("--device", type=str, default=None, choices=[None, "cuda", "cpu"],
                   help="Force a single device. Default uses device_map='auto'.")
    p.add_argument("--text-kind", type=str, default="both",
                   choices=["single", "conversation", "both"])
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on number of samples (for smoke tests).")
    p.add_argument("--output", type=str, default=None,
                   help="JSON output path. Optional.")
    args = p.parse_args()

    print(f"Loading test split from {args.test} (split={args.split})")
    samples = load_guardchat(args.test, split=args.split)
    if args.limit:
        samples = samples[: int(args.limit)]
    print(f"  -> {len(samples)} samples")

    pipe = RecognitionPipeline.from_pretrained(
        weights=args.weights,
        mode=args.mode,
        device=args.device,
        dtype=args.dtype,
    )

    kinds: List[str]
    if args.text_kind == "both":
        kinds = ["single", "conversation"]
    else:
        kinds = [args.text_kind]

    out: Dict[str, Dict] = {}
    for kind in kinds:
        preds, metrics = _evaluate_one_kind(pipe, samples, kind)
        out[kind] = {
            "metrics": metrics,
            "predictions": [p.to_dict() for p in preds],
        }
        print(f"\n[{kind}-turn]")
        for key in ["macro_f1", "recall_binary", "asr"]:
            print(f"  {key:>14}: {metrics[key]:.4f}")
        per_class = {k: v for k, v in metrics.items() if k.startswith("f1_")}
        for k, v in per_class.items():
            print(f"  {k:>14}: {v:.4f}")

    out["meta"] = {
        "mode": args.mode,
        "dtype": args.dtype,
        "weights": args.weights,
        "num_samples": len(samples),
    }

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nSaved results -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
