"""Task 1 - Multi-Label Unsafe Text Recognition pipeline.

Builds the SafeGuider recognition system for GuardChat. The training
recipe follows the paper (Section 6.1, "Fine-tuning Strategies and
Hyperparameters"):

    SafeGuider's ANN was trained on 14,000 samples (9,000 harmful
    conversational samples from GuardChat and 5,000 safe prompts from
    DiffusionDB) using AdamW (lr = 2 x 1e-5, batch size 32,
    weight decay 0.01) for 10 epochs.

This module exposes :class:`RecognitionPipeline` that wires together the
CLIP text encoder (vendored ``CLIPEncoder``) and the multi-label MLP
:class:`~src.SafeGuider.classifier.MultiLabelClassifier`, plus a
:class:`RecognitionTrainer` for the training loop.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

# Vendored re-export (sys.path setup happens in this package's __init__).
from . import CLIPEncoder
from .classifier import MultiLabelClassifier
from .data import (
    CATEGORIES,
    NUM_CATEGORIES,
    GuardChatSample,
    split_texts_and_labels,
)
from .metrics import summarise_recognition


# ------------------------ Inference pipeline ------------------------ #

@dataclass
class RecognitionPrediction:
    sample_id: str
    text: str
    probs: List[float]                 # length 6
    multi_label: List[int]             # length 6
    binary_pred: int                   # 0/1
    label_names: List[str]             # subset of CATEGORIES
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


class RecognitionPipeline:
    """SafeGuider Task 1 wrapper: CLIP-EOS encoder + multi-label MLP."""

    def __init__(
        self,
        encoder: CLIPEncoder,
        classifier: MultiLabelClassifier,
        threshold: float = 0.5,
    ) -> None:
        self.encoder = encoder
        self.classifier = classifier.to(self.encoder.device).eval()
        self.threshold = float(threshold)

    @classmethod
    def from_pretrained(
        cls,
        weights: str,
        encoder_model: str = "openai/clip-vit-large-patch14",
        device: Optional[str] = None,
        threshold: float = 0.5,
        verbose: bool = False,
    ) -> "RecognitionPipeline":
        if not os.path.isfile(weights):
            raise FileNotFoundError(f"Recognition weights not found: {weights!r}")
        encoder = CLIPEncoder(model_name=encoder_model, device=device, verbose=verbose)
        clf = MultiLabelClassifier(dim=encoder.hidden_size).to(encoder.device)
        state = torch.load(weights, map_location=encoder.device, weights_only=False)
        clf.load_state_dict(state)
        clf.eval()
        return cls(encoder=encoder, classifier=clf, threshold=threshold)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        torch.save(self.classifier.state_dict(), path)

    @torch.no_grad()
    def predict_batch(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
    ) -> List[Tuple[List[float], List[int], int]]:
        """Encode + classify a list of texts. Returns ``[(probs, multi, binary), ...]``."""
        out: List[Tuple[List[float], List[int], int]] = []
        for start in range(0, len(texts), batch_size):
            chunk = list(texts[start:start + batch_size])
            if not chunk:
                continue
            eos = self.encoder.eos_embedding(chunk)             # (B, D)
            probs, multi, binary = self.classifier.predict(eos, threshold=self.threshold)
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


# --------------------------- Training -------------------------------- #

class _EmbeddingTensorDataset(Dataset):
    """Wraps cached EOS embeddings + multi-hot labels."""

    def __init__(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        if embeddings.size(0) != labels.size(0):
            raise ValueError(
                f"embeddings and labels must align: got {embeddings.size(0)} "
                f"vs {labels.size(0)}."
            )
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self) -> int:
        return int(self.embeddings.size(0))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]


@torch.no_grad()
def encode_texts(
    encoder: CLIPEncoder,
    texts: Sequence[str],
    batch_size: int = 32,
    desc: str = "encoding",
) -> torch.Tensor:
    """Run the CLIP text encoder over ``texts`` and stack EOS embeddings."""
    pieces: List[torch.Tensor] = []
    try:
        from tqdm import tqdm
        rng = tqdm(range(0, len(texts), batch_size), desc=desc)
    except ImportError:
        rng = range(0, len(texts), batch_size)
    for start in rng:
        chunk = list(texts[start:start + batch_size])
        if not chunk:
            continue
        eos = encoder.eos_embedding(chunk).detach().cpu()
        pieces.append(eos)
    if not pieces:
        return torch.empty(0)
    return torch.cat(pieces, dim=0)


@dataclass
class TrainConfig:
    """Hyperparameters mirroring the paper's Task 1 setup."""

    epochs: int = 10
    batch_size: int = 32
    lr: float = 2e-5
    weight_decay: float = 1e-2
    threshold: float = 0.5
    seed: int = 111
    val_fraction: float = 0.1


class RecognitionTrainer:
    """Train :class:`MultiLabelClassifier` on cached EOS embeddings."""

    def __init__(
        self,
        encoder: CLIPEncoder,
        config: TrainConfig = TrainConfig(),
        device: Optional[torch.device] = None,
    ) -> None:
        self.encoder = encoder
        self.config = config
        self.device = device or encoder.device
        self.classifier = MultiLabelClassifier(dim=encoder.hidden_size).to(self.device)

    # -------------------- core training loop -------------------- #

    def fit(
        self,
        train_samples: Sequence[GuardChatSample],
        val_samples: Optional[Sequence[GuardChatSample]] = None,
        text_kind: str = "conversation",
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Train on ``train_samples``; optionally evaluate on ``val_samples``.

        ``text_kind`` chooses between flattened conversations and the
        single-turn enhanced prompt for the input representation. The
        paper trains on conversational samples plus benign single
        prompts; both feed naturally through the same encoder.
        """
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)

        train_texts, train_labels = split_texts_and_labels(train_samples, kind=text_kind)
        train_emb = encode_texts(self.encoder, train_texts,
                                 batch_size=self.config.batch_size,
                                 desc="encoding/train")
        train_emb = train_emb.to(self.device)
        train_y = torch.tensor(train_labels, dtype=torch.float32, device=self.device)
        train_ds = _EmbeddingTensorDataset(train_emb, train_y)
        train_loader = DataLoader(train_ds, batch_size=self.config.batch_size, shuffle=True)

        val_loader = None
        if val_samples:
            val_texts, val_labels = split_texts_and_labels(val_samples, kind=text_kind)
            val_emb = encode_texts(self.encoder, val_texts,
                                   batch_size=self.config.batch_size,
                                   desc="encoding/val").to(self.device)
            val_y = torch.tensor(val_labels, dtype=torch.float32, device=self.device)
            val_ds = _EmbeddingTensorDataset(val_emb, val_y)
            val_loader = DataLoader(val_ds, batch_size=self.config.batch_size, shuffle=False)

        opt = torch.optim.AdamW(
            self.classifier.parameters(),
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
                        for k, v in self.classifier.state_dict().items()
                    }
            print(f"[epoch {epoch:02d}] " + " ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in entry.items()
            ))
            history.append(entry)

        if best_state is not None:
            self.classifier.load_state_dict(best_state)

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
            torch.save(self.classifier.state_dict(), save_path)
            print(f"Saved classifier weights -> {save_path}")

        return {"history": history, "best_macro_f1": best_macro}

    # -------------------- internals -------------------- #

    def _train_one_epoch(self, loader: DataLoader, opt, loss_fn) -> float:
        self.classifier.train()
        total = 0.0
        n = 0
        for emb, target in loader:
            emb = emb.to(self.device)
            target = target.to(self.device)
            opt.zero_grad()
            logits, _ = self.classifier(emb)
            loss = loss_fn(logits, target)
            loss.backward()
            opt.step()
            total += float(loss.item()) * target.size(0)
            n += int(target.size(0))
        return total / max(n, 1)

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.classifier.eval()
        all_true: List[List[int]] = []
        all_pred: List[List[int]] = []
        for emb, target in loader:
            emb = emb.to(self.device)
            _, probs = self.classifier(emb)
            pred = (probs >= self.config.threshold).long().cpu().tolist()
            true = target.long().cpu().tolist()
            all_true.extend(true)
            all_pred.extend(pred)
        return summarise_recognition(all_true, all_pred)
