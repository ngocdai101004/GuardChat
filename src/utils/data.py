"""GuardChat data loader.

GuardChat samples couple an enhanced toxic prompt with a 6--10 turn
adversarial conversation and a multi-label vector over six NSFW
categories. This module normalises a few common on-disk formats into a
single :class:`GuardChatSample` representation that the recognition and
rewriting pipelines can both consume.

Supported input formats
-----------------------

* A JSON file containing a list of samples.
* A JSONL file containing one sample per line.
* A JSON dict ``{"data": [...]}``.

Each sample is expected to expose at least one of:

* ``enhanced_prompt`` (str) - target single-turn prompt for Task 1/2.
* ``conversation`` (list of turn dicts with ``role`` + ``content``) - the
  multi-turn dialogue for Task 1.

Multi-label annotations may appear as either:

* ``labels``: list of category strings, e.g. ``["sexual", "violence"]``.
* ``label_vector``: list of 6 ints in canonical order.
* a dict mapping each category to 0/1.

Safe prompts (used as negative samples when training the recognition
head) are loaded by :func:`load_safe_prompts` from a JSON list of
strings or ``{"prompt": ...}`` dicts and assigned an all-zero label
vector.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union


# Canonical category ordering used everywhere downstream. The paper lists
# these six categories (Section 3, Appendix A); fixing the order makes
# label vectors comparable across train/eval/inference.
CATEGORIES: Tuple[str, ...] = (
    "sexual",
    "illegal",
    "shocking",
    "violence",
    "self-harm",
    "harassment",
)
NUM_CATEGORIES: int = len(CATEGORIES)

_CATEGORY_INDEX: Dict[str, int] = {c: i for i, c in enumerate(CATEGORIES)}

# Common spelling variants that may appear in source datasets — mapped
# back to the canonical names. See Appendix A (label normalisation).
_LABEL_ALIASES: Dict[str, str] = {
    "self_harm": "self-harm",
    "selfharm": "self-harm",
    "self-injury": "self-harm",
    "nudity": "sexual",
    "sexual_content": "sexual",
    "sex": "sexual",
    "hate": "harassment",
    "discrimination": "harassment",
    "abuse": "harassment",
    "violence_graphic": "violence",
    "physical_harm": "violence",
    "illegal_activity": "illegal",
    "illegal_activities": "illegal",
}


def _canonical_category(label: str) -> Optional[str]:
    """Map a raw label string to the canonical category name, or ``None``."""
    if not label:
        return None
    s = label.strip().lower().replace(" ", "_")
    if s in _CATEGORY_INDEX:
        return s
    s2 = s.replace("_", "-")
    if s2 in _CATEGORY_INDEX:
        return s2
    if s in _LABEL_ALIASES:
        return _LABEL_ALIASES[s]
    if s2 in _LABEL_ALIASES:
        return _LABEL_ALIASES[s2]
    return None


def label_vector_from_labels(
    labels: Union[Sequence[str], Dict[str, Any], Sequence[int], None],
) -> List[int]:
    """Convert a flexible label spec into the canonical 6-dim 0/1 vector."""
    vec = [0] * NUM_CATEGORIES
    if labels is None:
        return vec

    # Already a length-6 vector
    if isinstance(labels, (list, tuple)) and len(labels) == NUM_CATEGORIES \
            and all(isinstance(x, (int, float)) for x in labels):
        return [int(bool(x)) for x in labels]

    if isinstance(labels, dict):
        for k, v in labels.items():
            cat = _canonical_category(str(k))
            if cat is not None and float(v) > 0:
                vec[_CATEGORY_INDEX[cat]] = 1
        return vec

    if isinstance(labels, (list, tuple)):
        for k in labels:
            cat = _canonical_category(str(k))
            if cat is not None:
                vec[_CATEGORY_INDEX[cat]] = 1
        return vec

    raise ValueError(f"Unsupported labels payload: {labels!r}")


def flatten_conversation(
    conversation: Sequence[Dict[str, Any]],
    role_prefix: bool = True,
    sep: str = "\n",
) -> str:
    """Concatenate a list of turn dicts into a single string.

    ``conversation`` items must expose ``content`` (str). ``role`` is used
    when ``role_prefix=True`` so that downstream encoders see who said
    what (matches the paper's "concatenated dialogue history" baseline).
    """
    parts: List[str] = []
    for turn in conversation:
        content = str(turn.get("content", "")).strip()
        if not content:
            continue
        if role_prefix:
            role = str(turn.get("role", "user")).strip() or "user"
            parts.append(f"{role}: {content}")
        else:
            parts.append(content)
    return sep.join(parts)


@dataclass
class GuardChatSample:
    """One GuardChat sample, normalised across input formats."""

    sample_id: str
    enhanced_prompt: str
    conversation: List[Dict[str, Any]] = field(default_factory=list)
    label_vector: List[int] = field(default_factory=lambda: [0] * NUM_CATEGORIES)
    source: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def label_names(self) -> List[str]:
        return [c for c, v in zip(CATEGORIES, self.label_vector) if v == 1]

    @property
    def is_unsafe(self) -> bool:
        return any(v == 1 for v in self.label_vector)

    def conversation_text(self, role_prefix: bool = True, sep: str = "\n") -> str:
        return flatten_conversation(self.conversation, role_prefix=role_prefix, sep=sep)

    def text_for(self, kind: str = "single") -> str:
        """Return the text that should feed the recognition encoder.

        ``kind`` is one of ``"single"`` (the enhanced prompt only) or
        ``"conversation"`` (the flattened multi-turn history). The two
        modes correspond to the paper's $X_{single}$ and $X_{conv}$.
        """
        if kind == "single":
            return self.enhanced_prompt
        if kind == "conversation":
            text = self.conversation_text()
            return text if text else self.enhanced_prompt
        raise ValueError(f"kind must be 'single' or 'conversation', got {kind!r}")


# --------------------------- File reading ---------------------------- #

def _iter_json_records(path: str) -> Iterable[Dict[str, Any]]:
    """Yield raw record dicts from a .json or .jsonl file."""
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        if ext == ".jsonl":
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
            return
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected list, list-of-dict, or {{'data': [...]}}.")
    for it in data:
        if isinstance(it, str):
            # Bare prompt; wrap into a minimal dict.
            yield {"enhanced_prompt": it}
        elif isinstance(it, dict):
            yield it
        else:
            raise ValueError(f"{path}: unsupported record {it!r}")


def _record_to_sample(rec: Dict[str, Any], idx: int) -> GuardChatSample:
    sid = str(rec.get("id", rec.get("sample_id", idx)))
    prompt = (
        rec.get("enhanced_prompt")
        or rec.get("prompt")
        or rec.get("rewritten_prompt")
        or rec.get("text")
        or ""
    )
    conv = rec.get("conversation") or rec.get("turns") or []
    if isinstance(conv, dict):
        # Some serialisers write {"1": {role, content}, ...}
        conv = [conv[k] for k in sorted(conv.keys(), key=lambda x: int(x))]
    if not isinstance(conv, list):
        conv = []

    labels = (
        rec.get("label_vector")
        or rec.get("labels")
        or rec.get("categories")
        or rec.get("category")
    )
    if isinstance(labels, str):
        labels = [labels]
    label_vec = label_vector_from_labels(labels)

    return GuardChatSample(
        sample_id=sid,
        enhanced_prompt=str(prompt),
        conversation=list(conv),
        label_vector=label_vec,
        source=rec.get("source"),
        raw=rec,
    )


def load_guardchat(path: str) -> List[GuardChatSample]:
    """Load a GuardChat split from a JSON/JSONL file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"GuardChat file not found: {path!r}")
    samples: List[GuardChatSample] = []
    for i, rec in enumerate(_iter_json_records(path)):
        samples.append(_record_to_sample(rec, i))
    if not samples:
        raise ValueError(f"{path}: no samples found.")
    return samples


def load_safe_prompts(path: str, sample_id_prefix: str = "safe") -> List[GuardChatSample]:
    """Load a list of safe (benign) prompts as ``GuardChatSample`` with
    all-zero label vectors. Used to mix DiffusionDB safe prompts into
    Task 1 training (per the paper's 9k harmful + 5k safe recipe).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Safe prompts file not found: {path!r}")
    samples: List[GuardChatSample] = []
    for i, rec in enumerate(_iter_json_records(path)):
        prompt = rec.get("prompt") or rec.get("enhanced_prompt") or rec.get("text") or ""
        if not prompt:
            continue
        samples.append(GuardChatSample(
            sample_id=f"{sample_id_prefix}-{i}",
            enhanced_prompt=str(prompt),
            conversation=[],
            label_vector=[0] * NUM_CATEGORIES,
            source=rec.get("source"),
            raw=rec,
        ))
    return samples


# --------------------- Convenience helpers --------------------------- #

def split_texts_and_labels(
    samples: Sequence[GuardChatSample],
    kind: str = "single",
) -> Tuple[List[str], List[List[int]]]:
    """Vectorise a list of samples for the recognition head."""
    texts = [s.text_for(kind=kind) for s in samples]
    labels = [list(s.label_vector) for s in samples]
    return texts, labels
