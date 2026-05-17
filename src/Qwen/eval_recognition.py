"""CLI: evaluate Qwen2.5-7B-Instruct on GuardChat Task 1 (zero-shot).

    python -m src.Qwen.eval_recognition \
        --test data/guardchat/test.jsonl \
        --weights src/Qwen/weights/Qwen2.5-7B-Instruct \
        --dtype bfloat16 \
        --text-kind both \
        --output results/qwen_task1.json

Output schema is identical to the supervised baselines and to
LlamaGuard's, so a single aggregator can compose Table 1 across all
five baselines.

There is no ``train_recognition.py`` for this baseline - Qwen is used
strictly zero-shot. The classifier is configured entirely by the
system prompt in :mod:`src.Qwen.prompts`.
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
from src.Qwen.model import DEFAULT_LOCAL_DIR  # noqa: E402
from src.Qwen.recognition import RecognitionPipeline  # noqa: E402


def _evaluate_one_kind(pipe: RecognitionPipeline, samples, kind: str):
    preds = pipe.predict_samples(samples, kind=kind)
    y_true = [list(s.label_vector) for s in samples]
    y_pred = [list(p.multi_label) for p in preds]
    metrics = summarise_recognition(y_true, y_pred)
    return preds, metrics


def main() -> int:
    p = argparse.ArgumentParser(
        description="Evaluate Qwen2.5-7B-Instruct on GuardChat Task 1."
    )
    p.add_argument("--test", type=str, default="multimedia-synergy-lab/GuardChat",
                   help="HuggingFace repo id (default 'multimedia-synergy-lab/"
                        "GuardChat') or local JSON/JSONL path.")
    p.add_argument("--split", type=str, default="test",
                   help="HF split when --test is a repo id. Default: test.")
    p.add_argument("--weights", type=str, default=DEFAULT_LOCAL_DIR,
                   help="Local snapshot dir (or HF id). "
                        "Default: src/Qwen/weights/Qwen2.5-7B-Instruct.")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32", "int8", "nf4"],
                   help="Weight dtype. int8/nf4 require bitsandbytes >= 0.43.")
    p.add_argument("--device", type=str, default=None, choices=[None, "cuda", "cpu"],
                   help="Force a single device. Default uses device_map='auto'.")
    p.add_argument("--text-kind", type=str, default="both",
                   choices=["single", "conversation", "both"])
    p.add_argument("--max-new-tokens", type=int, default=64,
                   help="Token budget for the JSON verdict. 64 fits 6 keys.")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on number of samples (for smoke tests).")
    p.add_argument("--output", type=str, default=None,
                   help="JSON output path. Optional.")
    p.add_argument("--system-prompt-file", type=str, default=None,
                   help="Override the default system prompt with a text file. "
                        "Useful for prompt-engineering ablations.")
    args = p.parse_args()

    print(f"Loading test split from {args.test} (split={args.split})")
    samples = load_guardchat(args.test, split=args.split)
    if args.limit:
        samples = samples[: int(args.limit)]
    print(f"  -> {len(samples)} samples")

    custom_prompt = None
    if args.system_prompt_file:
        with open(args.system_prompt_file, "r", encoding="utf-8") as f:
            custom_prompt = f.read()

    pipe = RecognitionPipeline.from_pretrained(
        weights=args.weights,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        system_prompt=custom_prompt,
    )

    kinds: List[str]
    if args.text_kind == "both":
        kinds = ["single", "conversation"]
    else:
        kinds = [args.text_kind]

    out: Dict[str, Dict] = {}
    for kind in kinds:
        preds, metrics = _evaluate_one_kind(pipe, samples, kind)
        n_parse_fail = sum(1 for p in preds if not p.parse_ok)
        out[kind] = {
            "metrics": metrics,
            "parse_failures": n_parse_fail,
            "predictions": [p.to_dict() for p in preds],
        }
        print(f"\n[{kind}-turn]")
        for key in ["macro_f1", "recall_binary", "asr"]:
            print(f"  {key:>14}: {metrics[key]:.4f}")
        per_class = {k: v for k, v in metrics.items() if k.startswith("f1_")}
        for k, v in per_class.items():
            print(f"  {k:>14}: {v:.4f}")
        if n_parse_fail:
            print(f"  parse_failures: {n_parse_fail} / {len(preds)}")

    out["meta"] = {
        "dtype": args.dtype,
        "weights": args.weights,
        "max_new_tokens": args.max_new_tokens,
        "num_samples": len(samples),
        "system_prompt_overridden": custom_prompt is not None,
    }

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nSaved results -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
