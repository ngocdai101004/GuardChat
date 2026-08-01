"""Shared evaluation loop for the zero-shot Task-1 baselines.

:mod:`src.ShieldGemma.eval_recognition` and
:mod:`src.LlamaGuard.eval_recognition` differ only in which model they
load; the surrounding machinery - resumable checkpointing, restoring
dataset order, computing the metric bundle, writing one JSON file per
input representation - is identical and lives here.

Output layout, one file per representation::

    <output_dir>/<slug>_task1_prompt.json
    <output_dir>/<slug>_task1_raw_prompt.json
    <output_dir>/<slug>_task1_conversation.json

Each file keeps the schema shared with the supervised baselines::

    { "<kind>": { "metrics": {...}, "predictions": [...] },
      "meta":   {...} }

Long runs are checkpointed to ``<output>.partial.jsonl`` after every
sample, so a killed process loses nothing and ``resume=True`` picks up
where it stopped.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Sequence

from .data import CATEGORIES, GuardChatSample, normalise_text_kind
from .metrics import summarise_recognition


def output_path(output_dir: str, slug: str, kind: str) -> str:
    return os.path.join(output_dir, f"{slug}_task1_{kind}.json")


def checkpoint_path(out_path: str) -> str:
    return out_path + ".partial.jsonl"


def load_checkpoint(path: str) -> Dict[str, dict]:
    """Read a ``.partial.jsonl`` checkpoint into ``{sample_id: record}``."""
    done: Dict[str, dict] = {}
    if not os.path.isfile(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated last line from a killed run
            sid = rec.get("sample_id")
            if sid is not None:
                done[str(sid)] = rec
    return done


def evaluate_kind(
    predict: Callable[[Sequence[GuardChatSample], str, Callable[[Any], None]], List[Any]],
    samples: Sequence[GuardChatSample],
    kind: str,
    output_dir: str,
    slug: str,
    resume: bool = False,
) -> Dict[str, Any]:
    """Score every sample under one representation, with checkpointing.

    ``predict(pending, kind, on_prediction)`` must return a list of
    prediction objects exposing ``sample_id`` and ``to_dict()``, and call
    ``on_prediction`` once per finished sample.
    """
    kind = normalise_text_kind(kind)
    out_path = output_path(output_dir, slug, kind)
    ckpt_path = checkpoint_path(out_path)

    done = load_checkpoint(ckpt_path) if resume else {}
    if done:
        print(f"  resuming: {len(done)} sample(s) already scored in {ckpt_path}")
    if not resume and os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    pending = [s for s in samples if str(s.sample_id) not in done]

    os.makedirs(output_dir, exist_ok=True)
    ckpt_file = open(ckpt_path, "a", encoding="utf-8")

    def _append(pred: Any) -> None:
        ckpt_file.write(json.dumps(pred.to_dict(), ensure_ascii=False) + "\n")
        ckpt_file.flush()

    try:
        fresh = predict(pending, kind, _append)
    finally:
        ckpt_file.close()

    for pred in fresh:
        done[str(pred.sample_id)] = pred.to_dict()

    # Restore the original dataset order.
    scored = [s for s in samples if str(s.sample_id) in done]
    records = [done[str(s.sample_id)] for s in scored]

    y_true = [list(s.label_vector) for s in scored]
    # Index by CATEGORIES rather than the dict's own order - checkpoint
    # records round-trip through JSON and must stay column-aligned.
    y_pred = [[int(r["multi_label"].get(c, 0)) for c in CATEGORIES] for r in records]
    metrics = summarise_recognition(y_true, y_pred)

    return {"metrics": metrics, "predictions": records, "path": out_path}


def save_kind(
    result: Dict[str, Any],
    kind: str,
    meta: Dict[str, Any],
    keep_checkpoint: bool = False,
) -> str:
    """Write the per-kind JSON file and drop its checkpoint."""
    out_path = str(result["path"])
    payload = {
        kind: {"metrics": result["metrics"], "predictions": result["predictions"]},
        "meta": {**meta, "text_kind": kind},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    if not keep_checkpoint:
        ckpt = checkpoint_path(out_path)
        if os.path.exists(ckpt):
            os.remove(ckpt)
    return out_path


def print_metrics(metrics: Dict[str, float]) -> None:
    for key in ("macro_f1", "recall_binary", "asr"):
        if key in metrics:
            print(f"  {key:>14}: {metrics[key]:.4f}")
    for key, val in metrics.items():
        if key.startswith("f1_"):
            print(f"  {key:>14}: {val:.4f}")


__all__ = [
    "output_path",
    "checkpoint_path",
    "load_checkpoint",
    "evaluate_kind",
    "save_kind",
    "print_metrics",
]
