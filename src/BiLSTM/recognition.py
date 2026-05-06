"""Task 1 pipeline + trainer for the BiLSTM baseline.

API mirrors :mod:`src.SafeGuider.recognition` so that downstream
benchmark code can swap baselines transparently:

    pipe = RecognitionPipeline.from_pretrained(checkpoint)
    preds = pipe.predict_samples(samples, kind="conversation")

Training follows the paper's Task 1 recipe (Section 6.1):

    14,000 samples (9,000 GuardChat conversational + 5,000 DiffusionDB
    safe), AdamW (lr=2e-5, weight_decay=0.01), batch size 32, 10 epochs,
    BCEWithLogits over six categories, early-stop by macro-F1 on a held-
    out fraction of training (or an explicit val split).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.utils import (
    CATEGORIES,
    NUM_CATEGORIES,
    GuardChatSample,
    split_texts_and_labels,
    summarise_recognition,
)

from .model import BiLSTMClassifier, BiLSTMConfig
from .tokenizer import Vocab


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


class _IdsDataset(Dataset):
    def __init__(self, ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> None:
        self.ids = ids
        self.labels = labels

    def __len__(self) -> int:
        return int(self.ids.size(0))

    def __getitem__(self, idx: int):
        if self.labels is None:
            return self.ids[idx]
        return self.ids[idx], self.labels[idx]


def _encode_texts(texts: Sequence[str], vocab: Vocab, max_len: int) -> torch.Tensor:
    rows = [vocab.encode_text(t, max_len=max_len) for t in texts]
    return torch.tensor(rows, dtype=torch.long)


class RecognitionPipeline:
    """Inference wrapper: vocab + checkpoint -> predictions."""

    def __init__(
        self,
        model: BiLSTMClassifier,
        vocab: Vocab,
        max_len: int,
        device: Optional[torch.device] = None,
        threshold: float = 0.5,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device).eval()
        self.vocab = vocab
        self.max_len = int(max_len)
        self.threshold = float(threshold)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str,
        device: Optional[str] = None,
        threshold: float = 0.5,
    ) -> "RecognitionPipeline":
        """Load a ``.pt`` produced by :class:`RecognitionTrainer`.

        The checkpoint bundles model state dict, ``BiLSTMConfig`` kwargs,
        the serialised :class:`Vocab` (``itos`` list), and ``max_len``.
        """
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"BiLSTM checkpoint not found: {checkpoint!r}")
        bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = BiLSTMConfig(**bundle["config"])
        model = BiLSTMClassifier(cfg)
        model.load_state_dict(bundle["state_dict"])
        vocab = Vocab(bundle["vocab_itos"])
        dev = torch.device(device) if device else None
        return cls(
            model=model,
            vocab=vocab,
            max_len=int(bundle["max_len"]),
            device=dev,
            threshold=threshold,
        )

    def save(self, path: str) -> None:
        """Persist model state, config, vocab, and ``max_len`` together."""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        cfg = self.model.config
        bundle = {
            "state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "config": {
                "vocab_size": cfg.vocab_size,
                "num_classes": cfg.num_classes,
                "embed_dim": cfg.embed_dim,
                "hidden1": cfg.hidden1,
                "hidden2": cfg.hidden2,
                "dense_dim": cfg.dense_dim,
                "dropout": cfg.dropout,
                "pad_index": cfg.pad_index,
            },
            "vocab_itos": self.vocab.itos,
            "max_len": self.max_len,
        }
        torch.save(bundle, path)

    @torch.no_grad()
    def predict_batch(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
    ) -> List[Tuple[List[float], List[int], int]]:
        ids = _encode_texts(texts, self.vocab, self.max_len)
        out: List[Tuple[List[float], List[int], int]] = []
        for start in range(0, ids.size(0), batch_size):
            chunk = ids[start:start + batch_size].to(self.device)
            probs, multi, binary = self.model.predict(chunk, threshold=self.threshold)
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
    """Hyperparameters - mirror SafeGuider's recognition trainer.

    The notebook used Adam at default lr; we follow the paper's recipe
    instead (AdamW @ 2e-5, weight_decay=0.01) so all baselines share the
    same optimisation budget.
    """

    epochs: int = 10
    batch_size: int = 32
    lr: float = 2e-5
    weight_decay: float = 1e-2
    threshold: float = 0.5
    seed: int = 111
    max_len: int = 256
    vocab_size: int = 30_000
    val_fraction: float = 0.1
    embed_dim: int = 100
    hidden1: int = 128
    hidden2: int = 64
    dense_dim: int = 64
    dropout: float = 0.5


class RecognitionTrainer:
    """Build vocab + train BiLSTM on GuardChat / DiffusionDB samples."""

    def __init__(
        self,
        config: TrainConfig = TrainConfig(),
        device: Optional[torch.device] = None,
    ) -> None:
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.vocab: Optional[Vocab] = None
        self.model: Optional[BiLSTMClassifier] = None

    def fit(
        self,
        train_samples: Sequence[GuardChatSample],
        val_samples: Optional[Sequence[GuardChatSample]] = None,
        text_kind: str = "conversation",
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)

        train_texts, train_labels = split_texts_and_labels(train_samples, kind=text_kind)
        # Build vocab on the training corpus only to avoid test leakage.
        self.vocab = Vocab.build(
            train_texts,
            max_size=self.config.vocab_size,
        )
        print(f"[BiLSTM] vocab size = {len(self.vocab)}")

        train_ids = _encode_texts(train_texts, self.vocab, self.config.max_len)
        train_y = torch.tensor(train_labels, dtype=torch.float32)
        train_ds = _IdsDataset(train_ids, train_y)
        train_loader = DataLoader(
            train_ds, batch_size=self.config.batch_size, shuffle=True,
        )

        val_loader = None
        if val_samples:
            val_texts, val_labels = split_texts_and_labels(val_samples, kind=text_kind)
            val_ids = _encode_texts(val_texts, self.vocab, self.config.max_len)
            val_y = torch.tensor(val_labels, dtype=torch.float32)
            val_loader = DataLoader(
                _IdsDataset(val_ids, val_y),
                batch_size=self.config.batch_size, shuffle=False,
            )

        cfg = BiLSTMConfig(
            vocab_size=len(self.vocab),
            num_classes=NUM_CATEGORIES,
            embed_dim=self.config.embed_dim,
            hidden1=self.config.hidden1,
            hidden2=self.config.hidden2,
            dense_dim=self.config.dense_dim,
            dropout=self.config.dropout,
            pad_index=self.vocab.pad_index,
        )
        self.model = BiLSTMClassifier(cfg).to(self.device)

        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        loss_fn = nn.BCEWithLogitsLoss()

        history: List[Dict[str, float]] = []
        best_macro = -1.0
        best_state = None
        for epoch in range(1, self.config.epochs + 1):
            tr_loss = self._train_one_epoch(train_loader, opt, loss_fn)
            entry: Dict[str, float] = {"epoch": epoch, "train_loss": tr_loss}
            if val_loader is not None:
                vmetrics = self._evaluate(val_loader)
                entry.update(vmetrics)
                if vmetrics["macro_f1"] > best_macro:
                    best_macro = vmetrics["macro_f1"]
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.model.state_dict().items()
                    }
            print(f"[epoch {epoch:02d}] " + " ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in entry.items()
            ))
            history.append(entry)

        if best_state is not None:
            self.model.load_state_dict(best_state)

        if save_path:
            pipe = RecognitionPipeline(
                model=self.model,
                vocab=self.vocab,
                max_len=self.config.max_len,
                device=self.device,
                threshold=self.config.threshold,
            )
            pipe.save(save_path)
            print(f"Saved BiLSTM bundle -> {save_path}")

        return {"history": history, "best_macro_f1": best_macro}

    # ----------------------- Internals ---------------------------- #

    def _train_one_epoch(self, loader: DataLoader, opt, loss_fn) -> float:
        self.model.train()
        total = 0.0
        n = 0
        for ids, target in loader:
            ids = ids.to(self.device)
            target = target.to(self.device)
            opt.zero_grad()
            logits, _ = self.model(ids)
            loss = loss_fn(logits, target)
            loss.backward()
            opt.step()
            total += float(loss.item()) * target.size(0)
            n += int(target.size(0))
        return total / max(n, 1)

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_true: List[List[int]] = []
        all_pred: List[List[int]] = []
        for ids, target in loader:
            ids = ids.to(self.device)
            _, probs = self.model(ids)
            pred = (probs >= self.config.threshold).long().cpu().tolist()
            true = target.long().cpu().tolist()
            all_true.extend(true)
            all_pred.extend(pred)
        return summarise_recognition(all_true, all_pred)
