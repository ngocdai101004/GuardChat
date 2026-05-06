"""Task 1 pipeline + trainer for the BERT baseline.

Mirrors the public surface of :mod:`src.SafeGuider.recognition` and
:mod:`src.BiLSTM.recognition` so a shared benchmark aggregator can swap
backbones without touching call sites:

    pipe = RecognitionPipeline.from_pretrained(checkpoint_dir)
    preds = pipe.predict_samples(samples, kind="conversation")

Training implements the paper's Task 1 recipe (Section 6.1):

    14,000 samples (9,000 GuardChat conversational + 5,000 DiffusionDB
    safe), AdamW (lr=2e-5, weight_decay=0.01), batch size 32, 10 epochs,
    BCE-with-logits over six categories (handled internally by the
    HF model when ``problem_type='multi_label_classification'``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from src.utils import (
    CATEGORIES,
    NUM_CATEGORIES,
    GuardChatSample,
    split_texts_and_labels,
    summarise_recognition,
)

from .model import BERTClassifier, BERTConfig, DEFAULT_MODEL_NAME


# ----------------------- Inference pipeline ------------------------ #

@dataclass
class RecognitionPrediction:
    sample_id: str
    text: str
    probs: List[float]
    multi_label: List[int]
    binary_pred: int
    label_names: List[str]
    label_vector_true: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "sample_id": self.sample_id,
            "text": self.text,
            "probs": {c: p for c, p in zip(CATEGORIES, self.probs)},
            "multi_label": {c: int(v) for c, v in zip(CATEGORIES, self.multi_label)},
            "predicted_categories": self.label_names,
            "binary_pred": int(self.binary_pred),
        }
        if self.label_vector_true is not None:
            out["label_vector_true"] = {
                c: int(v) for c, v in zip(CATEGORIES, self.label_vector_true)
            }
        return out


class _BERTTextDataset(Dataset):
    """Tokenises lazily so very long corpora don't materialise all input
    ids at once. Items are dicts compatible with the HF model.
    """

    def __init__(
        self,
        texts: Sequence[str],
        labels: Optional[Sequence[Sequence[int]]],
        tokenizer,
        max_length: int,
    ) -> None:
        self.texts = list(texts)
        self.labels = (
            torch.tensor(labels, dtype=torch.float32)
            if labels is not None else None
        )
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


class RecognitionPipeline:
    """Inference wrapper: tokenizer + checkpoint -> predictions."""

    def __init__(
        self,
        model: BERTClassifier,
        tokenizer,
        max_length: int,
        device: Optional[torch.device] = None,
        threshold: float = 0.5,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.threshold = float(threshold)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str,
        device: Optional[str] = None,
        threshold: float = 0.5,
        max_length: Optional[int] = None,
    ) -> "RecognitionPipeline":
        """Load a checkpoint directory written by :class:`RecognitionTrainer`.

        We honour an accompanying ``recognition_meta.json`` if present,
        which carries the ``max_length`` used at training time. Falls
        back to the explicit ``max_length`` argument or the BERT default
        (256) when the file is missing.
        """
        if not os.path.isdir(checkpoint):
            raise FileNotFoundError(
                f"BERT checkpoint dir not found: {checkpoint!r}. "
                f"Expected a HuggingFace `save_pretrained` folder."
            )

        meta_path = os.path.join(checkpoint, "recognition_meta.json")
        meta: Dict[str, Any] = {}
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        eff_max = (
            int(max_length) if max_length is not None
            else int(meta.get("max_length", 256))
        )

        tokenizer = BERTClassifier.make_tokenizer(checkpoint)
        model = BERTClassifier.from_pretrained(
            checkpoint,
            num_classes=int(meta.get("num_classes", NUM_CATEGORIES)),
            max_length=eff_max,
        )
        dev = torch.device(device) if device else None
        return cls(
            model=model,
            tokenizer=tokenizer,
            max_length=eff_max,
            device=dev,
            threshold=threshold,
        )

    def save(self, save_dir: str) -> None:
        """Persist HF model + tokenizer + a small JSON of training meta."""
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        meta = {
            "num_classes": self.model.config.num_classes,
            "max_length": self.max_length,
            "categories": list(CATEGORIES),
        }
        with open(os.path.join(save_dir, "recognition_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @torch.no_grad()
    def predict_batch(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
    ) -> List[Tuple[List[float], List[int], int]]:
        ds = _BERTTextDataset(texts, labels=None, tokenizer=self.tokenizer,
                              max_length=self.max_length)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        out: List[Tuple[List[float], List[int], int]] = []
        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attn = batch["attention_mask"].to(self.device)
            probs, multi, binary = self.model.predict(
                input_ids=input_ids,
                attention_mask=attn,
                threshold=self.threshold,
            )
            for i in range(probs.size(0)):
                out.append((
                    [float(probs[i, j].item()) for j in range(NUM_CATEGORIES)],
                    [int(multi[i, j].item()) for j in range(NUM_CATEGORIES)],
                    int(binary[i].item()),
                ))
        return out

    def predict_samples(
        self,
        samples: Sequence[GuardChatSample],
        kind: str = "single",
        batch_size: int = 32,
    ) -> List[RecognitionPrediction]:
        texts = [s.text_for(kind=kind) for s in samples]
        outs = self.predict_batch(texts, batch_size=batch_size)
        results: List[RecognitionPrediction] = []
        for s, (probs, multi, binary) in zip(samples, outs):
            results.append(RecognitionPrediction(
                sample_id=s.sample_id,
                text=s.text_for(kind=kind),
                probs=probs,
                multi_label=multi,
                binary_pred=binary,
                label_names=[c for c, v in zip(CATEGORIES, multi) if v == 1],
                label_vector_true=list(s.label_vector),
            ))
        return results


# ------------------------------ Trainer ----------------------------- #

@dataclass
class TrainConfig:
    """Hyperparameters - mirror SafeGuider/BiLSTM trainers (paper recipe)."""

    epochs: int = 10
    batch_size: int = 32
    lr: float = 2e-5
    weight_decay: float = 1e-2
    threshold: float = 0.5
    seed: int = 111
    max_length: int = 256
    model_name: str = DEFAULT_MODEL_NAME
    val_fraction: float = 0.1
    grad_clip: float = 1.0
    num_workers: int = 0


class RecognitionTrainer:
    """Fine-tune a HuggingFace BERT for multi-label GuardChat Task 1."""

    def __init__(
        self,
        config: TrainConfig = TrainConfig(),
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.tokenizer = BERTClassifier.make_tokenizer(config.model_name)
        self.model = BERTClassifier(
            BERTConfig(
                model_name=config.model_name,
                num_classes=NUM_CATEGORIES,
                max_length=config.max_length,
            )
        )

    def fit(
        self,
        train_samples: Sequence[GuardChatSample],
        val_samples: Optional[Sequence[GuardChatSample]] = None,
        text_kind: str = "conversation",
        save_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)

        train_texts, train_labels = split_texts_and_labels(train_samples, kind=text_kind)
        train_loader = DataLoader(
            _BERTTextDataset(train_texts, train_labels, self.tokenizer, self.config.max_length),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )

        val_loader = None
        if val_samples:
            val_texts, val_labels = split_texts_and_labels(val_samples, kind=text_kind)
            val_loader = DataLoader(
                _BERTTextDataset(val_texts, val_labels, self.tokenizer, self.config.max_length),
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
            )

        self.model.to(self.device)
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        history: List[Dict[str, float]] = []
        best_macro = -1.0
        best_dir: Optional[str] = save_dir
        for epoch in range(1, self.config.epochs + 1):
            tr_loss = self._train_one_epoch(train_loader, opt)
            entry: Dict[str, float] = {"epoch": epoch, "train_loss": tr_loss}
            if val_loader is not None:
                vmetrics = self._evaluate(val_loader)
                entry.update(vmetrics)
                if vmetrics["macro_f1"] > best_macro:
                    best_macro = vmetrics["macro_f1"]
                    if best_dir is not None:
                        self._save_checkpoint(best_dir)
                        entry["saved"] = 1.0
            print(f"[epoch {epoch:02d}] " + " ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in entry.items()
            ))
            history.append(entry)

        if save_dir and (val_loader is None or best_macro < 0):
            # Either no val set, or no improvement was ever recorded.
            self._save_checkpoint(save_dir)

        return {"history": history, "best_macro_f1": best_macro}

    # ----------------------- Internals ---------------------------- #

    def _save_checkpoint(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        # Save HF model + tokenizer + meta in one folder so the
        # inference pipeline can load via from_pretrained(save_dir).
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        meta = {
            "num_classes": NUM_CATEGORIES,
            "max_length": self.config.max_length,
            "categories": list(CATEGORIES),
            "model_name": self.config.model_name,
        }
        with open(os.path.join(save_dir, "recognition_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def _train_one_epoch(self, loader: DataLoader, opt) -> float:
        self.model.train()
        total = 0.0
        n = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attn = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            opt.zero_grad()
            outputs = self.model.forward_with_loss(
                input_ids=input_ids,
                attention_mask=attn,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            if self.config.grad_clip and self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip,
                )
            opt.step()

            total += float(loss.item()) * labels.size(0)
            n += int(labels.size(0))
        return total / max(n, 1)

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_true: List[List[int]] = []
        all_pred: List[List[int]] = []
        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attn = batch["attention_mask"].to(self.device)
            target = batch["labels"]
            _, probs = self.model(input_ids=input_ids, attention_mask=attn)
            pred = (probs.cpu() >= self.config.threshold).long().tolist()
            all_pred.extend(pred)
            all_true.extend(target.long().tolist())
        return summarise_recognition(all_true, all_pred)
