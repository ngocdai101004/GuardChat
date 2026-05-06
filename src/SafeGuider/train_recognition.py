"""CLI: train the SafeGuider multi-label recognition head on GuardChat.

Reproduces the paper's Task 1 recipe (Section 6.1):
    9,000 harmful conversational samples from GuardChat (train split)
  + 5,000 safe prompts from DiffusionDB
  -> AdamW (lr=2e-5, weight_decay=1e-2), batch size 32, 10 epochs.

Usage:
    python -m src.SafeGuider.train_recognition \
        --train data/guardchat/train.jsonl \
        --safe  data/diffusiondb_safe.json \
        --output src/SafeGuider/weights/recognition_multilabel.pt

The script encodes the chosen text representation through the CLIP text
encoder once, caches the EOS embeddings, then trains the multi-label
MLP. Set ``--text-kind single`` to train on enhanced prompts only;
``conversation`` (default) trains on the flattened multi-turn dialogue.
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

# Allow `python src/SafeGuider/train_recognition.py` (no -m) by ensuring
# the repo root is on sys.path so that `import src.SafeGuider...` works.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.SafeGuider import CLIPEncoder  # noqa: E402
from src.utils import (  # noqa: E402
    GuardChatSample,
    load_guardchat,
    load_safe_prompts,
)
from src.SafeGuider.recognition import RecognitionTrainer, TrainConfig  # noqa: E402


DEFAULT_OUTPUT = os.path.join(_HERE, "weights", "recognition_multilabel.pt")


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
        return samples, []
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
    p = argparse.ArgumentParser(description="Train SafeGuider Task 1 (recognition).")
    p.add_argument("--train", required=True, type=str,
                   help="GuardChat train split (JSON/JSONL).")
    p.add_argument("--safe", type=str, default=None,
                   help="JSON of benign prompts (e.g. DiffusionDB) to add as label-0 samples.")
    p.add_argument("--val", type=str, default=None,
                   help="Optional GuardChat val split. If omitted, "
                        "--val-fraction of --train is held out.")
    p.add_argument("--text-kind", type=str, default="conversation",
                   choices=["single", "conversation"],
                   help="Which representation to feed the encoder.")
    p.add_argument("--encoder-model", type=str, default="openai/clip-vit-large-patch14")
    p.add_argument("--device", type=str, default=None, choices=[None, "cuda", "cpu"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Sigmoid threshold for multi-label and binary predictions.")
    p.add_argument("--val-fraction", type=float, default=0.1,
                   help="Fraction of train held out for validation if --val is not given.")
    p.add_argument("--seed", type=int, default=111)
    p.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    p.add_argument("--history-out", type=str, default=None,
                   help="Optional path to dump per-epoch metrics JSON.")
    args = p.parse_args()

    _set_seed(args.seed)

    print(f"Loading train samples from {args.train}")
    train_samples = load_guardchat(args.train)
    if args.safe:
        print(f"Loading safe prompts from {args.safe}")
        safe_samples = load_safe_prompts(args.safe)
        print(f"  -> {len(safe_samples)} safe prompts")
        train_samples = list(train_samples) + safe_samples
    print(f"Total train samples: {len(train_samples)}")

    if args.val:
        print(f"Loading val samples from {args.val}")
        val_samples = load_guardchat(args.val)
    else:
        train_samples, val_samples = _split_train_val(
            train_samples, args.val_fraction, args.seed,
        )
    print(f"Train/Val sizes: {len(train_samples)} / {len(val_samples)}")

    encoder = CLIPEncoder(model_name=args.encoder_model, device=args.device)
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        threshold=args.threshold,
        seed=args.seed,
        val_fraction=args.val_fraction,
    )
    trainer = RecognitionTrainer(encoder=encoder, config=cfg)
    summary = trainer.fit(
        train_samples=train_samples,
        val_samples=val_samples,
        text_kind=args.text_kind,
        save_path=args.output,
    )

    if args.history_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.history_out)) or ".", exist_ok=True)
        with open(args.history_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved training history -> {args.history_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
