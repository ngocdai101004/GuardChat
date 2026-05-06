"""CLI: evaluate the BERT Task 1 baseline on GuardChat.

    python -m src.BERT.eval_recognition \
        --test data/guardchat/test.jsonl \
        --weights src/BERT/weights/bert_multilabel \
        --text-kind both \
        --output results/bert_task1.json

Output JSON schema is identical to ``safeguider_task1.json`` and
``bilstm_task1.json`` so a single aggregator can produce Table 1.
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
from src.BERT.recognition import RecognitionPipeline  # noqa: E402


DEFAULT_WEIGHTS = os.path.join(_HERE, "weights", "bert_multilabel")


def _evaluate_one_kind(pipe: RecognitionPipeline, samples, kind: str, batch_size: int):
    preds = pipe.predict_samples(samples, kind=kind, batch_size=batch_size)
    y_true = [list(s.label_vector) for s in samples]
    y_pred = [list(p.multi_label) for p in preds]
    metrics = summarise_recognition(y_true, y_pred)
    return preds, metrics


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate BERT Task 1 on GuardChat.")
    p.add_argument("--test", required=True, type=str)
    p.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS,
                   help="HuggingFace save_pretrained directory.")
    p.add_argument("--device", type=str, default=None, choices=[None, "cuda", "cpu"])
    p.add_argument("--text-kind", type=str, default="both",
                   choices=["single", "conversation", "both"])
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max-length", type=int, default=None,
                   help="Override max_length (else uses recognition_meta.json).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    print(f"Loading test split from {args.test}")
    samples = load_guardchat(args.test)
    print(f"  -> {len(samples)} samples")

    pipe = RecognitionPipeline.from_pretrained(
        checkpoint=args.weights,
        device=args.device,
        threshold=args.threshold,
        max_length=args.max_length,
    )

    kinds: List[str]
    if args.text_kind == "both":
        kinds = ["single", "conversation"]
    else:
        kinds = [args.text_kind]

    out: Dict[str, Dict] = {}
    for kind in kinds:
        preds, metrics = _evaluate_one_kind(pipe, samples, kind, args.batch_size)
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

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nSaved results -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
