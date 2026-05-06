"""PyTorch BiLSTM classifier for GuardChat Task 1.

PyTorch port of a stacked Bidirectional-LSTM architecture
(originally a Keras-based sentiment classifier):

    Embedding(input_dim, 100)
    -> Bidirectional(LSTM(128, return_sequences=True)) + L2 reg
    -> BatchNorm + Dropout(0.5)
    -> Bidirectional(LSTM(64))
    -> BatchNorm + Dropout(0.5)
    -> Dense(64, ReLU) + Dropout(0.5)
    -> Dense(num_classes)

Two adaptations versus the reference design:

1. The original used a single-label softmax over four sentiment
   classes. GuardChat Task 1 is multi-label over six NSFW categories,
   so the final layer outputs 6 logits and the loss is BCEWithLogits.

2. PyTorch's :class:`nn.LSTM` does not expose Keras-style L2 kernel
   regularisation, so we approximate it via AdamW's ``weight_decay``
   parameter set in the trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import NUM_CATEGORIES


@dataclass
class BiLSTMConfig:
    """Architectural hyperparameters for :class:`BiLSTMClassifier`."""

    vocab_size: int
    num_classes: int = NUM_CATEGORIES
    embed_dim: int = 100
    hidden1: int = 128
    hidden2: int = 64
    dense_dim: int = 64
    dropout: float = 0.5
    pad_index: int = 0


class BiLSTMClassifier(nn.Module):
    """Stacked BiLSTM with batch-norm + dropout, 6-way sigmoid head."""

    def __init__(self, config: BiLSTMConfig) -> None:
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embed_dim,
            padding_idx=config.pad_index,
        )
        self.lstm1 = nn.LSTM(
            input_size=config.embed_dim,
            hidden_size=config.hidden1,
            batch_first=True,
            bidirectional=True,
        )
        # 2 * hidden1 because the LSTM is bidirectional.
        self.bn1 = nn.BatchNorm1d(2 * config.hidden1)
        self.dropout1 = nn.Dropout(config.dropout)

        self.lstm2 = nn.LSTM(
            input_size=2 * config.hidden1,
            hidden_size=config.hidden2,
            batch_first=True,
            bidirectional=True,
        )
        self.bn2 = nn.BatchNorm1d(2 * config.hidden2)
        self.dropout2 = nn.Dropout(config.dropout)

        self.fc1 = nn.Linear(2 * config.hidden2, config.dense_dim)
        self.dropout3 = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(config.dense_dim, config.num_classes)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits, probs)`` of shape ``(B, num_classes)``."""
        # input_ids: (B, T) -> embed: (B, T, embed_dim)
        x = self.embedding(input_ids)

        # First BiLSTM keeps the sequence; apply BatchNorm over channels.
        x, _ = self.lstm1(x)                       # (B, T, 2*hidden1)
        x = x.transpose(1, 2)                      # (B, 2*hidden1, T)
        x = self.bn1(x)
        x = x.transpose(1, 2)                      # (B, T, 2*hidden1)
        x = self.dropout1(x)

        # Second BiLSTM collapses the sequence; take the final state of
        # both directions and concatenate (matches Keras' default
        # ``return_sequences=False`` for a Bidirectional LSTM).
        _, (h2, _) = self.lstm2(x)                 # h2: (2, B, hidden2)
        # h2[0] = forward last hidden, h2[1] = backward last hidden.
        x = torch.cat([h2[0], h2[1]], dim=-1)      # (B, 2*hidden2)
        x = self.bn2(x)
        x = self.dropout2(x)

        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        logits = self.fc2(x)
        probs = torch.sigmoid(logits)
        return logits, probs

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(probs, multi_label_pred, binary_pred)``.

        ``binary_pred`` is 1 iff any class fires above ``threshold`` -
        same convention as the SafeGuider multi-label head.
        """
        was_training = self.training
        self.eval()
        try:
            _, probs = self.forward(input_ids)
            multi = (probs >= threshold).long()
            binary = (multi.sum(dim=-1) > 0).long()
        finally:
            if was_training:
                self.train()
        return probs, multi, binary
