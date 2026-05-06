r"""Multi-label safety classifier for SafeGuider on GuardChat.

The vendored ``ThreeLayerClassifier`` is a 2-way (safe/unsafe) softmax
head trained on CLIP EOS embeddings. Task 1 of the GuardChat benchmark
requires a multi-label prediction over six NSFW categories
(``CATEGORIES`` in :mod:`.data`), so we provide a sigmoid-headed variant
with the same hidden architecture.

Architecture matches the original 3-layer SafeGuider MLP exactly except
for the final layer:

    Linear(D, 1024) -> ReLU -> Dropout(0.5)
    Linear(1024, 512) -> ReLU -> Dropout(0.5)
    Linear(512, NUM_CATEGORIES)  # sigmoid applied separately

The binary verdict $\hat{Y}_{binary}$ is recovered as ``any(prob >= 0.5)``
following the convention in the paper: an input is unsafe if any of the
six category heads fires.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import NUM_CATEGORIES


class MultiLabelClassifier(nn.Module):
    """6-way multi-label MLP over CLIP EOS embeddings."""

    def __init__(
        self,
        dim: int,
        num_classes: int = NUM_CATEGORIES,
        hidden1: int = 1024,
        hidden2: int = 512,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_classes = int(num_classes)
        self.fc1 = nn.Linear(self.dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, self.num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits, probs)`` of shape ``(B, num_classes)``."""
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        logits = self.fc3(h)
        probs = torch.sigmoid(logits)
        return logits, probs

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(probs, multi_label_pred, binary_pred)``.

        ``multi_label_pred`` is the 0/1 multi-hot prediction.
        ``binary_pred`` is 1 iff any class fires above ``threshold``.
        """
        _, probs = self.forward(x)
        multi = (probs >= threshold).long()
        binary = (multi.sum(dim=-1) > 0).long()
        return probs, multi, binary
