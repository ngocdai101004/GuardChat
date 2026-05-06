"""Word-level tokenizer + vocab for the BiLSTM baseline.

The reference Colab notebook (vendors/BiLSTM/sentiment_analysis.py) builds
its vocabulary with torchtext's ``basic_english`` tokenizer and a fixed
vocab cap. torchtext is now in maintenance mode and ships only on a
narrow Python/PyTorch matrix, so we re-implement the same idea with the
standard library:

    text  -> ``preprocess_text``  (URL/HTML strip, punctuation/digit
                                   replacement, emoji strip, lowercase)
          -> ``basic_tokenize``    (lowercase whitespace split with
                                   apostrophe + punctuation splitting)
          -> ``Vocab.encode``      (fixed-size int sequence with PAD)

The vocab object pickles to a small dict, so we serialise it as JSON for
portability instead of pickle - compatible with any deployment without
Python version pinning.
"""

from __future__ import annotations

import json
import os
import re
import string
from collections import Counter
from typing import Iterable, List, Optional, Sequence


# Preserve the regex patterns from the reference notebook so the
# preprocessing pipeline matches what the original BiLSTM was trained on.
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<.*?>")
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"   # emoticons
    "\U0001F300-\U0001F5FF"   # symbols & pictographs
    "\U0001F680-\U0001F6FF"   # transport & map symbols
    "\U0001F1E0-\U0001F1FF"   # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F926-\U0001F937"
    "‍"
    "♀-♂"
    "]+",
    flags=re.UNICODE,
)
_PUNCT_DIGIT_TABLE = str.maketrans(
    {ch: " " for ch in (string.punctuation + string.digits)}
)


def preprocess_text(text: str) -> str:
    """Replicate the notebook's ``preprocess_text``.

    Strip URLs / HTML, replace punctuation + digits with spaces, drop
    emojis, normalise whitespace, lowercase.
    """
    if not text:
        return ""
    text = _URL_RE.sub("", text)
    text = _HTML_RE.sub("", text)
    text = text.translate(_PUNCT_DIGIT_TABLE)
    text = _EMOJI_RE.sub(" ", text)
    text = " ".join(text.split())
    return text.lower()


# basic_english-style tokenizer: lowercase + whitespace split. After the
# preprocessing above all punctuation is already stripped, so a plain
# whitespace split is equivalent to torchtext's ``basic_english``.
def basic_tokenize(text: str) -> List[str]:
    return preprocess_text(text).split()


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_INDEX = 0
UNK_INDEX = 1


class Vocab:
    """Minimal vocab object: token -> integer id, with PAD and UNK."""

    def __init__(self, itos: Sequence[str]) -> None:
        if itos[PAD_INDEX] != PAD_TOKEN or itos[UNK_INDEX] != UNK_TOKEN:
            raise ValueError(
                f"itos must start with [{PAD_TOKEN!r}, {UNK_TOKEN!r}]; "
                f"got {itos[:2]!r}"
            )
        self.itos: List[str] = list(itos)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad_index(self) -> int:
        return PAD_INDEX

    @property
    def unk_index(self) -> int:
        return UNK_INDEX

    def encode(self, tokens: Sequence[str]) -> List[int]:
        unk = UNK_INDEX
        return [self.stoi.get(t, unk) for t in tokens]

    def encode_text(self, text: str, max_len: int) -> List[int]:
        ids = self.encode(basic_tokenize(text))
        if len(ids) < max_len:
            ids = ids + [PAD_INDEX] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        return ids

    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        max_size: Optional[int] = 30_000,
        min_freq: int = 1,
    ) -> "Vocab":
        """Build a vocab from a corpus of strings."""
        counter: Counter = Counter()
        for t in texts:
            counter.update(basic_tokenize(t))
        # Sort by frequency (desc), then lexicographically for stability.
        items = [(tok, cnt) for tok, cnt in counter.items() if cnt >= min_freq]
        items.sort(key=lambda kv: (-kv[1], kv[0]))
        if max_size is not None:
            items = items[: max(0, max_size - 2)]   # reserve PAD + UNK slots
        itos: List[str] = [PAD_TOKEN, UNK_TOKEN] + [tok for tok, _ in items]
        return cls(itos)

    # ----------------------- Serialisation ------------------------ #

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"itos": self.itos}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "Vocab":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["itos"])
