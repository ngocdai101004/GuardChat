"""Score a Task-2 result file for semantic preservation with SBERT.

Reads the JSON that :mod:`src.utils.task2_eval` wrote, embeds each
``(original_text, rewritten_text)`` pair, and returns per-sample scores
plus a summary. Nothing here re-runs a rewriter - this is a pure offline
metric pass over ``rewritten_text``, exactly as the Task-2 docstrings
promised, so it costs no API budget and can be re-run whenever the metric
definition changes.

Two similarity numbers per conversation
---------------------------------------
``similarity``
    One embedding for the whole flattened dialogue, on both sides. This
    is the direct analogue of the prompt-level score and the number that
    goes in the table.
``similarity_per_turn``
    Turn *i* against turn *i*, averaged over turns. Defined only when
    both sides have the same number of turns, which is the normal case
    (the rewriters are required to preserve turn count).

The second exists because GuardChat conversations average ~440 words -
past ``all-mpnet-base-v2``'s 384-token window - so the whole-dialogue
score reads a prefix of long samples. Per-turn scoring sidesteps that:
individual turns are short. Where the two agree, truncation did not
distort the result; where they diverge, the per-turn number is the one to
trust. Both are reported rather than silently picking one.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.utils.metrics import summarise_similarity
from src.utils.task2_eval import REWRITE_KINDS


def load_result_file(path: str) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """Open a ``<slug>_task2_<kind>.json`` and return ``(kind, rewrites, meta)``."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    kinds = [k for k in REWRITE_KINDS if k in payload]
    if not kinds:
        raise ValueError(
            f"{path} has no Task-2 representation block "
            f"(expected one of {REWRITE_KINDS}, found {sorted(payload)})."
        )
    kind = kinds[0]
    block = payload[kind]
    return kind, list(block.get("rewrites", [])), dict(payload.get("meta", {}))


def _is_scorable(record: Dict[str, Any]) -> bool:
    """A row is scorable when the rewriter produced usable text.

    Mirrors :attr:`RewriteRecord.is_usable`. Blocked and refused rows
    have an empty ``rewritten_text``; embedding "" would return the
    model's bias vector and a meaningless-but-nonzero similarity, so they
    are excluded from the mean and penalised separately.
    """
    return (record.get("status") == "ok"
            and bool(str(record.get("rewritten_text", "")).strip())
            and bool(str(record.get("original_text", "")).strip()))


def _skip_reason(record: Dict[str, Any]) -> str:
    """Why a row could not be scored, in the summary's vocabulary.

    ``status`` alone is not enough: a row can be ``"ok"`` and still carry
    an empty rewrite, and reporting that as ``"ok"`` in a list of skip
    reasons reads as a bug rather than a finding.
    """
    status = str(record.get("status") or "unknown")
    if status != "ok":
        return status
    if not str(record.get("original_text", "")).strip():
        return "empty_original"
    return "empty_rewrite"


def _turn_pairs(record: Dict[str, Any]) -> Optional[List[Tuple[str, str]]]:
    """Aligned ``(original_turn, rewritten_turn)`` pairs, or ``None``.

    Returns ``None`` when the record is not a conversation, when a
    rewriter dropped or added turns (nothing to align against), or when
    any turn is empty on either side.
    """
    src = record.get("original_turns")
    dst = record.get("rewritten_turns")
    if not src or not dst or len(src) != len(dst):
        return None
    pairs = [(str(a), str(b)) for a, b in zip(src, dst)]
    if any(not a.strip() or not b.strip() for a, b in pairs):
        return None
    return pairs


def score_records(
    encoder,
    records: Sequence[Dict[str, Any]],
    per_turn: bool = True,
    batch_size: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Attach an SBERT similarity to every record.

    Every text is embedded in one flat pass rather than per record, so
    the GPU sees full batches regardless of how the samples are shaped.
    Unscorable rows keep a ``None`` similarity and carry the reason.
    """
    out: List[Dict[str, Any]] = []

    # Collect first, embed once, scatter back. Each entry remembers where
    # its texts landed in the flat list via the temporary `_doc_base` /
    # `_turn_base` offsets, which `_clean` strips before serialising.
    texts: List[str] = []

    for rec in records:
        entry: Dict[str, Any] = {
            "sample_id": rec.get("sample_id"),
            "status": rec.get("status"),
            "error_kind": rec.get("error_kind"),
            "gold_category": rec.get("gold_category"),
            "similarity": None,
        }
        if not _is_scorable(rec):
            entry["skip_reason"] = _skip_reason(rec)
            out.append(entry)
            continue

        entry["_doc_base"] = len(texts)
        texts.append(str(rec["original_text"]))
        texts.append(str(rec["rewritten_text"]))

        if per_turn:
            pairs = _turn_pairs(rec)
            if pairs:
                entry["_turn_base"] = len(texts)
                entry["num_turns_scored"] = len(pairs)
                for a, b in pairs:
                    texts.append(a)
                    texts.append(b)
        out.append(entry)

    if not texts:
        return [_clean(e) for e in out]

    embeddings = encoder.encode(texts, batch_size=batch_size)
    lengths = encoder.token_counts(texts)
    limit = encoder.max_seq_length

    for entry in out:
        base = entry.pop("_doc_base", None)
        if base is None:
            continue
        a, b = embeddings[base], embeddings[base + 1]
        entry["similarity"] = float(encoder.cosine_similarity(
            a.unsqueeze(0), b.unsqueeze(0)).item())
        entry["original_tokens"] = lengths[base]
        entry["rewritten_tokens"] = lengths[base + 1]
        entry["truncated"] = bool(lengths[base] > limit or lengths[base + 1] > limit)

        t_base = entry.pop("_turn_base", None)
        if t_base is None:
            continue
        n = int(entry["num_turns_scored"])
        sims = [
            float(encoder.cosine_similarity(
                embeddings[t_base + 2 * k].unsqueeze(0),
                embeddings[t_base + 2 * k + 1].unsqueeze(0)).item())
            for k in range(n)
        ]
        entry["similarity_per_turn"] = sum(sims) / n
        entry["min_turn_similarity"] = min(sims)
        entry["turn_truncated"] = any(
            lengths[t_base + j] > limit for j in range(2 * n)
        )

    return [_clean(e) for e in out]


def _clean(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Drop the bookkeeping keys and round the floats for a readable file."""
    entry.pop("_doc_base", None)
    entry.pop("_turn_base", None)
    for key in ("similarity", "similarity_per_turn", "min_turn_similarity"):
        if entry.get(key) is not None:
            entry[key] = round(float(entry[key]), 6)
    return entry


def summarise_scores(
    scored: Sequence[Dict[str, Any]],
    encoder_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Aggregate per-sample scores into the numbers that go in the table."""
    sims = [e["similarity"] for e in scored if e.get("similarity") is not None]
    summary: Dict[str, Any] = dict(encoder_meta or {})
    summary.update(summarise_similarity(sims, num_total=len(scored)))

    skips: Dict[str, int] = {}
    for entry in scored:
        reason = entry.get("skip_reason")
        if reason:
            skips[reason] = skips.get(reason, 0) + 1
    if skips:
        summary["unscorable_reasons"] = skips

    truncated = [e for e in scored if e.get("truncated") is not None]
    if truncated:
        n_trunc = sum(1 for e in truncated if e["truncated"])
        summary["num_truncated"] = n_trunc
        summary["fraction_truncated"] = n_trunc / len(truncated)

    per_turn = [e["similarity_per_turn"] for e in scored
                if e.get("similarity_per_turn") is not None]
    if per_turn:
        turn_stats = summarise_similarity(per_turn, num_total=len(scored))
        summary["per_turn"] = {
            "num_aligned": len(per_turn),
            "mean_similarity": turn_stats["mean_similarity"],
            "median_similarity": turn_stats["median_similarity"],
            "std_similarity": turn_stats["std_similarity"],
            "mean_similarity_penalised": turn_stats["mean_similarity_penalised"],
            "mean_worst_turn": sum(
                e["min_turn_similarity"] for e in scored
                if e.get("min_turn_similarity") is not None
            ) / len(per_turn),
        }

    by_cat: Dict[str, List[float]] = {}
    for entry in scored:
        if entry.get("similarity") is None:
            continue
        cat = str(entry.get("gold_category") or "unknown")
        by_cat.setdefault(cat, []).append(float(entry["similarity"]))
    if len(by_cat) > 1:
        summary["mean_similarity_by_category"] = {
            cat: round(sum(v) / len(v), 6) for cat, v in sorted(by_cat.items())
        }

    return _round_floats(summary)


def _round_floats(obj: Any, places: int = 6) -> Any:
    if isinstance(obj, float):
        return round(obj, places)
    if isinstance(obj, dict):
        return {k: _round_floats(v, places) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, places) for v in obj]
    return obj


def output_path(output_dir: str, source_path: str, suffix: str = "sbert") -> str:
    """``<output_dir>/<source stem>_<suffix>.json``.

    The suffix carries the encoder, so an SBERT pass and a CLIP pass over
    the same run land side by side instead of overwriting each other.
    """
    stem = os.path.splitext(os.path.basename(source_path))[0]
    return os.path.join(output_dir, f"{stem}_{suffix}.json")


__all__ = [
    "load_result_file",
    "output_path",
    "score_records",
    "summarise_scores",
]
