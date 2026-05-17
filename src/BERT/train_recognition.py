"""CLI: fine-tune BERT for GuardChat Task 1 (multi-label recognition).

Usage matches SafeGuider's and BiLSTM's training CLIs:

    python -m src.BERT.train_recognition \
        --train data/guardchat/train.jsonl \
        --safe  data/diffusiondb_safe.json \
        --output src/BERT/weights/bert_multilabel \
        --text-kind conversation \
        --epochs 10 --batch-size 32 --lr 2e-5 --weight-decay 1e-2

Notes:
    * ``--output`` is a *directory* (HuggingFace ``save_pretrained`` layout),
      not a single ``.pt`` file. The matching tokenizer and a small
      ``recognition_meta.json`` are dropped alongside the model.
    * Recipe follows Section 6.1: AdamW @ lr=2e-5, weight decay 0.01,
      batch 32, 10 epochs, BCE-with-logits (multi-label) over six
      categories.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import List, Tuple

import numpy as np
import torch

# Repo-root sys.path bootstrap.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.utils import (  # noqa: E402
    GuardChatSample,
    load_guardchat,
    load_safe_prompts,
)
from src.BERT.model import DEFAULT_MODEL_NAME  # noqa: E402
from src.BERT.recognition import RecognitionTrainer, TrainConfig  # noqa: E402


DEFAULT_OUTPUT = os.path.join(_HERE, "weights", "bert_multilabel")


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _split_train_val(
    samples: List[GuardChatSample],
    val_fraction: float,
    seed: int,
) -> Tuple[List[GuardChatSample], List[GuardChatSample]]:
    if val_fraction <= 0:
        return list(samples), []
    rng = random.Random(seed)
    idxs = list(range(len(samples)))
    rng.shuffle(idxs)
    n_val = int(round(len(samples) * val_fraction))
    val_idx = set(idxs[:n_val])
    train, val = [], []
    for i, s in enumerate(samples):
        (val if i in val_idx else train).append(s)
    return train, val


def main() -> int:
    p = argparse.ArgumentParser(description="Fine-tune BERT Task 1 (recognition).")
    p.add_argument("--train", type=str, default="multimedia-synergy-lab/GuardChat",
                   help="HuggingFace repo id (default 'multimedia-synergy-lab/"
                        "GuardChat') or local JSON/JSONL path with train samples.")
    p.add_argument("--train-split", type=str, default="train",
                   help="HF split when --train is a repo id. Default: train.")
    p.add_argument("--safe", type=str, default=None,
                   help="Local JSON of benign prompts loaded as label-0 samples.")
    p.add_argument("--val", type=str, default=None,
                   help="Optional GuardChat val source (HF repo id or local).")
    p.add_argument("--val-split", type=str, default="test",
                   help="HF split when --val is a repo id. Default: test.")
    p.add_argument("--text-kind", type=str, default="conversation",
                   choices=["single", "conversation"])
    p.add_argument("--device", type=str, default=None, choices=[None, "cuda", "cpu"])
    p.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME,
                   help=f"HF model id or local dir. Default: {DEFAULT_MODEL_NAME}.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=111)
    p.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                   help="Output DIRECTORY (HF save_pretrained layout).")
    p.add_argument("--history-out", type=str, default=None)
    args = p.parse_args()

    _set_seed(args.seed)

    print(f"Loading train samples from {args.train} (split={args.train_split})")
    train_samples = list(load_guardchat(args.train, split=args.train_split))
    if args.safe:
        print(f"Loading safe prompts from {args.safe}")
        safe = load_safe_prompts(args.safe)
        print(f"  -> {len(safe)} safe prompts")
        train_samples.extend(safe)
    print(f"Total train samples: {len(train_samples)}")

    if args.val:
        print(f"Loading val samples from {args.val} (split={args.val_split})")
        val_samples = load_guardchat(args.val, split=args.val_split)
    else:
        train_samples, val_samples = _split_train_val(
            train_samples, args.val_fraction, args.seed,
        )
    print(f"Train/Val sizes: {len(train_samples)} / {len(val_samples)}")

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        threshold=args.threshold,
        seed=args.seed,
        max_length=args.max_length,
        model_name=args.model_name,
        val_fraction=args.val_fraction,
        grad_clip=args.grad_clip,
        num_workers=args.num_workers,
    )
    device = torch.device(args.device) if args.device else None
    trainer = RecognitionTrainer(config=cfg, device=device)
    summary = trainer.fit(
        train_samples=train_samples,
        val_samples=val_samples,
        text_kind=args.text_kind,
        save_dir=args.output,
    )

    if args.history_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.history_out)) or ".", exist_ok=True)
        with open(args.history_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved training history -> {args.history_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
