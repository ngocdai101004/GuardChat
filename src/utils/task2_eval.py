"""Shared record schema + evaluation loop for the Task-2 rewriters.

Task 2 asks a rewriter $g$ to turn an unsafe input $P_{unsafe}$ into a
sanitised $P_{safe}$. The three baselines (Gemini 2.5 Flash, local
Llama-3.1-8B, SafeGuider's beam search) differ only in how they produce
that string; the surrounding machinery - resumable checkpointing,
restoring dataset order, summarising, writing one JSON file per input
representation - is identical and lives here. This mirrors
:mod:`src.utils.task1_eval`.

Two input representations
-------------------------
``prompt``
    The enhanced adversarial prompt.
``conversation``
    The 6-9 turn dialogue. The rewriter must return the same number of
    turns, so the record carries both the turn lists and a
    ``turn_parse`` verdict.

Output layout::

    <output_dir>/<slug>_task2_prompt.json
    <output_dir>/<slug>_task2_conversation.json

Each file::

    { "<kind>": { "summary": {...}, "rewrites": [...] },
      "meta":   {...} }

Long runs are checkpointed to ``<output>.partial.jsonl`` after every
sample, so a killed process loses nothing and ``resume=True`` picks up
where it stopped. That matters more here than in Task 1: these are 1,000
paid API calls per representation.

Metrics are deliberately *not* computed here. The two paper metrics need
machinery that does not exist yet - Safe Generation Rate needs the three
T2I models plus the image-safety gate, and semantic similarity has moved
from CLIP to SBERT. Both consume the ``rewritten_text`` field of these
files offline. What :func:`summarise_rewrites` reports instead is
process health: how many rewrites came back usable, how many turns
survived, and how much text was thrown away.
"""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .data import GuardChatSample, flatten_conversation, gold_category


# Task 2 rewrites the enhanced prompt and the dialogue. ``raw_prompt``
# (Task 1's third representation) is out of scope: it is the unedited
# seed prompt from the source dataset, not something GuardChat attacks
# with, so there is nothing to defend against.
REWRITE_KINDS = ("prompt", "conversation")

# Every value ``RewriteRecord.status`` can take.
#
#   ok             usable $P_{safe}$
#   refusal        model answered, but with "I can't help with that"
#   blocked        provider safety filter killed the call
#   empty          call returned nothing
#   parse_failed   response could not be parsed back into turns
#   error          exception (network, quota, ...)
#
# Only ``ok`` should be fed to a T2I model. Everything else is a failure
# of the rewriter and counts against it in the Safe Generation Rate -
# see the note in :func:`summarise_rewrites`.
STATUSES = ("ok", "refusal", "blocked", "empty", "parse_failed", "error")

# ``status`` says *that* a row failed; ``error_kind`` says *why*, at the
# granularity that decides what to do about it. A provider block is a
# property of the sample and belongs in the paper; a quota error is a
# property of the billing account and means the run should be repeated
# with --resume. Collapsing the two into "error" would make a run that
# ran out of credit look like a rewriter that could not handle the data.
ERROR_KINDS = (
    "provider_block",    # safety filter killed the request - a real result
    "model_refusal",     # model answered, but declined to rewrite
    "quota",             # rate limit, quota exhausted, billing
    "auth",              # missing / rejected credentials
    "model_not_found",   # retired or misspelled model id
    "timeout",           # request deadline exceeded
    "network",           # connection reset, DNS, TLS
    "server_error",      # provider 5xx
    "oom",               # local model ran out of device memory
    "truncated",         # hit max_output_tokens mid-answer
    "empty_response",    # call succeeded but returned nothing
    "parse_failed",      # response could not be turned into P_safe
    "empty_input",       # nothing to rewrite in the source row
    "unknown",
)

# Checked in order - the first group whose marker appears in the message
# wins, so put the specific HTTP codes before the generic ones.
_ERROR_MARKERS = (
    # First: a CUDA OOM message carries allocation sizes that would
    # otherwise match the numeric HTTP-code markers below.
    ("oom", ("out of memory", "outofmemory", "cuda oom",
             "can't allocate memory", "cannot allocate memory")),
    ("auth", ("401", "403", "permission_denied", "unauthenticated",
              "api key not valid", "api_key_invalid", "invalid api key",
              "credentials")),
    ("quota", ("429", "rate limit", "rate-limit", "resource_exhausted",
               "quota", "billing", "insufficient_quota", "insufficient funds")),
    ("model_not_found", ("404", "not_found", "no longer available",
                         "is not found", "unknown model")),
    ("timeout", ("timeout", "timed out", "deadline")),
    ("network", ("connection", "ssl", "dns", "unreachable",
                 "reset by peer", "broken pipe")),
    ("server_error", ("500", "502", "503", "504", "internal error",
                      "unavailable", "overloaded", "backend error")),
)


def classify_error_message(message: Optional[str]) -> str:
    """Map a provider error string onto one of :data:`ERROR_KINDS`.

    Substring matching on the message text, because every SDK in this
    benchmark raises a different exception tree but all of them put the
    HTTP status and the provider's status name into ``str(e)``.
    """
    if not message:
        return "unknown"
    msg = str(message).lower()
    for kind, markers in _ERROR_MARKERS:
        if any(m in msg for m in markers):
            return kind
    return "unknown"


def normalise_rewrite_kind(kind: str) -> str:
    k = {"single": "prompt", "enhanced_prompt": "prompt"}.get(kind, kind)
    if k not in REWRITE_KINDS:
        raise ValueError(f"kind must be one of {REWRITE_KINDS}, got {kind!r}")
    return k


@dataclass
class RewriteRecord:
    """One rewrite, in the schema every Task-2 baseline serialises to.

    The raw response is always kept alongside the parsed result so a
    parsing or prompting decision can be revisited offline without
    re-spending the API budget.
    """

    sample_id: str
    text_kind: str
    original_text: str
    rewritten_text: str
    status: str = "ok"
    error_kind: Optional[str] = None   # one of ERROR_KINDS; None when status == "ok"
    attempts: int = 1                  # calls actually made, including retries
    raw_response: str = ""
    was_modified: bool = False
    elapsed_sec: float = 0.0
    model: str = ""

    # Gold annotations, carried through so downstream analysis can break
    # SGR down by category without re-joining against the dataset.
    gold_category: Optional[str] = None
    label_names: List[str] = field(default_factory=list)
    source: Optional[str] = None

    # Conversation kind only.
    original_turns: Optional[List[str]] = None
    rewritten_turns: Optional[List[str]] = None
    num_turns_in: Optional[int] = None
    num_turns_out: Optional[int] = None
    turn_parse: Optional[str] = None      # ok | turn_count_mismatch | unmarked | parse_failed

    # Provider diagnostics (Gemini populates these; local models leave
    # them None).
    blocked: bool = False
    block_reason: Optional[str] = None
    finish_reason: Optional[str] = None

    # Baseline-specific diagnostics that do not generalise across
    # rewriters - SafeGuider's safety scores and deleted words, say.
    # Kept in one nested object so the common fields stay identically
    # shaped for every model and a cross-baseline aggregator can ignore
    # this key entirely.
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_usable(self) -> bool:
        """True when this row can be handed to a T2I model as-is."""
        return self.status == "ok" and bool(self.rewritten_text.strip())

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "sample_id": self.sample_id,
            "text_kind": self.text_kind,
            "status": self.status,
            "error_kind": self.error_kind,
            "attempts": int(self.attempts),
            "original_text": self.original_text,
            "rewritten_text": self.rewritten_text,
            "was_modified": bool(self.was_modified),
            "elapsed_sec": round(float(self.elapsed_sec), 4),
            "model": self.model,
            "gold_category": self.gold_category,
            "label_names": list(self.label_names),
            "source": self.source,
        }
        if self.original_turns is not None or self.rewritten_turns is not None:
            out.update({
                "num_turns_in": self.num_turns_in,
                "num_turns_out": self.num_turns_out,
                "turn_parse": self.turn_parse,
                "original_turns": self.original_turns,
                "rewritten_turns": self.rewritten_turns,
            })
        if self.blocked or self.block_reason or self.finish_reason:
            out.update({
                "blocked": bool(self.blocked),
                "block_reason": self.block_reason,
                "finish_reason": self.finish_reason,
            })
        if self.extra:
            out["extra"] = dict(self.extra)
        # Last, because it is long and noisy to read.
        out["raw_response"] = self.raw_response
        return out


# ------------------------- Input extraction -------------------------- #

def original_text_for(sample: GuardChatSample, kind: str) -> str:
    """The text a rewrite is compared against for this representation.

    For ``conversation`` this is the same flattened form Task 1 feeds its
    recognisers ($X_{conv}$, ``"user: ..."`` per line), so semantic
    similarity is computed over the representation the benchmark already
    reports elsewhere.
    """
    kind = normalise_rewrite_kind(kind)
    if kind == "prompt":
        return str(sample.enhanced_prompt or "")
    text = flatten_conversation(sample.conversation, role_prefix=True)
    return text or str(sample.enhanced_prompt or "")


def rewritten_conversation_text(
    turns: Sequence[str],
    role_prefix: bool = True,
) -> str:
    """Flatten rewritten turns the same way :func:`original_text_for` does.

    Both sides of the similarity comparison must be built by the same
    function, otherwise the ``"user: "`` prefixes alone move the score.
    """
    return flatten_conversation(
        [{"role": "user", "content": t} for t in turns],
        role_prefix=role_prefix,
    )


def base_record(
    sample: GuardChatSample,
    kind: str,
    model: str,
) -> RewriteRecord:
    """A record pre-filled with everything that does not depend on the model."""
    kind = normalise_rewrite_kind(kind)
    rec = RewriteRecord(
        sample_id=str(sample.sample_id),
        text_kind=kind,
        original_text=original_text_for(sample, kind),
        rewritten_text="",
        model=model,
        gold_category=gold_category(sample),
        label_names=list(sample.label_names),
        source=sample.source,
    )
    if kind == "conversation":
        turns = [
            " ".join(str(t.get("content", "")).split())
            for t in sample.conversation
        ]
        turns = [t for t in turns if t]
        rec.original_turns = turns
        rec.num_turns_in = len(turns)
    return rec


# ---------------------------- Output paths --------------------------- #

def output_path(output_dir: str, slug: str, kind: str) -> str:
    return os.path.join(output_dir, f"{slug}_task2_{kind}.json")


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


def load_result_file(path: str) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """Open a finished ``<slug>_task2_<kind>.json`` -> ``(kind, rewrites, meta)``.

    Every offline consumer of these files - the similarity metric, the
    image-generation pass - starts here, so the "which representation is
    this file?" question is answered once and identically for all of
    them.
    """
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


# ------------------------------ The loop ----------------------------- #

def rewrite_kind(
    rewrite: Callable[[Sequence[GuardChatSample], str, Callable[[Any], None]], List[Any]],
    samples: Sequence[GuardChatSample],
    kind: str,
    output_dir: str,
    slug: str,
    resume: bool = False,
) -> Dict[str, Any]:
    """Rewrite every sample under one representation, with checkpointing.

    ``rewrite(pending, kind, on_result)`` must return a list of objects
    exposing ``sample_id`` and ``to_dict()``, and call ``on_result`` once
    per finished sample. ``on_result`` is thread-safe, so a pipeline may
    fan the API calls out across workers.
    """
    kind = normalise_rewrite_kind(kind)
    out_path = output_path(output_dir, slug, kind)
    ckpt_path = checkpoint_path(out_path)

    done = load_checkpoint(ckpt_path) if resume else {}
    if done:
        print(f"  resuming: {len(done)} sample(s) already rewritten in {ckpt_path}")
    if not resume and os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    pending = [s for s in samples if str(s.sample_id) not in done]

    os.makedirs(output_dir, exist_ok=True)
    ckpt_file = open(ckpt_path, "a", encoding="utf-8")

    # Workers append concurrently; a lock keeps lines from interleaving.
    import threading
    lock = threading.Lock()

    def _append(rec: Any) -> None:
        line = json.dumps(rec.to_dict(), ensure_ascii=False)
        with lock:
            ckpt_file.write(line + "\n")
            ckpt_file.flush()

    try:
        fresh = rewrite(pending, kind, _append)
    finally:
        ckpt_file.close()

    for rec in fresh:
        done[str(rec.sample_id)] = rec.to_dict()

    # Restore the original dataset order.
    scored = [s for s in samples if str(s.sample_id) in done]
    records = [done[str(s.sample_id)] for s in scored]

    return {
        "summary": summarise_rewrites(records),
        "rewrites": records,
        "path": out_path,
    }


def save_kind(
    result: Dict[str, Any],
    kind: str,
    meta: Dict[str, Any],
    keep_checkpoint: bool = False,
) -> str:
    """Write the per-kind JSON file and drop its checkpoint."""
    out_path = str(result["path"])
    payload = {
        kind: {"summary": result["summary"], "rewrites": result["rewrites"]},
        "meta": {**meta, "text_kind": kind},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    if not keep_checkpoint:
        ckpt = checkpoint_path(out_path)
        if os.path.exists(ckpt):
            os.remove(ckpt)
    return out_path


# ------------------------------ Summary ------------------------------ #

def _mean(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return float(statistics.fmean(vals)) if vals else None


def summarise_rewrites(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Process-health statistics for one representation.

    Not the paper's metrics - those come later, from the T2I pipeline and
    the SBERT encoder. These numbers answer "did the rewriter actually
    run, and did it hollow out the prompts?", which is what decides
    whether a run is worth paying to generate images from.

    ``mean_length_ratio`` is the cheap stand-in for semantic similarity
    until SBERT lands: a rewriter that answers every input with "a serene
    landscape" scores a perfect Safe Generation Rate and a length ratio
    near zero.
    """
    n = len(records)
    if n == 0:
        return {"num_samples": 0}

    status_counts: Dict[str, int] = {}
    error_counts: Dict[str, int] = {}
    for r in records:
        s = str(r.get("status", "?"))
        status_counts[s] = status_counts.get(s, 0) + 1
        kind = r.get("error_kind")
        if kind:
            error_counts[str(kind)] = error_counts.get(str(kind), 0) + 1

    usable = [r for r in records
              if r.get("status") == "ok" and str(r.get("rewritten_text", "")).strip()]
    ratios = [
        len(str(r.get("rewritten_text", ""))) / max(1, len(str(r.get("original_text", ""))))
        for r in usable
    ]

    summary: Dict[str, Any] = {
        "num_samples": n,
        "num_usable": len(usable),
        "fraction_usable": len(usable) / n,
        "status_counts": status_counts,
        "error_kind_counts": error_counts,
        "num_retried": sum(1 for r in records if int(r.get("attempts") or 1) > 1),
        "fraction_modified": sum(1 for r in records if r.get("was_modified")) / n,
        "mean_length_ratio": _mean(ratios),
        "mean_elapsed_sec": _mean([r.get("elapsed_sec", 0.0) for r in records]),
        "total_elapsed_sec": round(
            sum(float(r.get("elapsed_sec") or 0.0) for r in records), 2
        ),
    }

    # Conversation-only: did the turn structure survive the round trip?
    conv = [r for r in records if r.get("turn_parse") is not None]
    if conv:
        parse_counts: Dict[str, int] = {}
        for r in conv:
            p = str(r.get("turn_parse"))
            parse_counts[p] = parse_counts.get(p, 0) + 1
        matched = sum(
            1 for r in conv
            if r.get("num_turns_in") is not None
            and r.get("num_turns_in") == r.get("num_turns_out")
        )
        summary.update({
            "turn_parse_counts": parse_counts,
            "turn_count_match_rate": matched / len(conv),
            "mean_turns_in": _mean([r.get("num_turns_in") for r in conv]),
            "mean_turns_out": _mean([r.get("num_turns_out") for r in conv]),
        })

    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    for key, val in summary.items():
        if isinstance(val, float):
            print(f"  {key:>22}: {val:.4f}")
        elif isinstance(val, dict):
            pretty = ", ".join(f"{k}={v}" for k, v in sorted(val.items()))
            print(f"  {key:>22}: {pretty}")
        else:
            print(f"  {key:>22}: {val}")


__all__ = [
    "ERROR_KINDS",
    "REWRITE_KINDS",
    "STATUSES",
    "RewriteRecord",
    "base_record",
    "classify_error_message",
    "checkpoint_path",
    "load_checkpoint",
    "normalise_rewrite_kind",
    "original_text_for",
    "output_path",
    "print_summary",
    "rewrite_kind",
    "rewritten_conversation_text",
    "save_kind",
    "summarise_rewrites",
]
