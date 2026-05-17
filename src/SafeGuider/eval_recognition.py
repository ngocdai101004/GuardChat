"""CLI: evaluate SafeGuider Task 1 (multi-label unsafe text recognition).

Mirrors Table 1 of the paper: report Macro-F1 over six categories plus
binary recall / Attack Success Rate (ASR), separately for the
single-turn and multi-turn input representations.

Usage:
    python -m src.SafeGuider.eval_recognition \
        --test data/guardchat/test.jsonl \
        --weights src/SafeGuider/weights/recognition_multilabel.pt \
        --text-kind both \
        --output results/safeguider_task1.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

# Repo-root sys.path bootstrap (see train_recognition.py for rationale).
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.utils import load_guardchat, summarise_recognition  # noqa: E402
from src.SafeGuider.recognition import RecognitionPipeline  # noqa: E402


DEFAULT_WEIGHTS = os.path.join(_HERE, "weights", "recognition_multilabel.pt")


def _evaluate_one_kind(
    pipe: RecognitionPipeline,
    samples,
    kind: str,
    batch_size: int,
):
    preds = pipe.predict_samples(samples, kind=kind, batch_size=batch_size)
    y_true = [list(s.label_vector) for s in samples]
    y_pred = [list(p.multi_label) for p in preds]
    metrics = summarise_recognition(y_true, y_pred)
    return preds, metrics


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate SafeGuider Task 1 on GuardChat.")
    p.add_argument("--test", type=str, default="multimedia-synergy-lab/GuardChat",
                   help="Either a HuggingFace dataset repo id (default: "
                        "'multimedia-synergy-lab/GuardChat') or a local "
                        "JSON/JSONL path with verified test samples.")
    p.add_argument("--split", type=str, default="test",
                   help="HF split when --test is a repo id (train/test/full). "
                        "Default: test.")
    p.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS,
                   help="Multi-label classifier .pt path.")
    p.add_argument("--encoder-model", type=str, default="openai/clip-vit-large-patch14")
    p.add_argument("--device", type=str, default=None, choices=[None, "cuda", "cpu"])
    p.add_argument("--text-kind", type=str, default="both",
                   choices=["single", "conversation", "both"])
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Sigmoid decision threshold (default 0.5).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--output", type=str, default=None,
                   help="Optional JSON file to dump predictions + metrics.")
    args = p.parse_args()

    print(f"Loading test split from {args.test} (split={args.split})")
    samples = load_guardchat(args.test, split=args.split)
    print(f"  -> {len(samples)} samples")

    pipe = RecognitionPipeline.from_pretrained(
        weights=args.weights,
        encoder_model=args.encoder_model,
        device=args.device,
        threshold=args.threshold,
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
