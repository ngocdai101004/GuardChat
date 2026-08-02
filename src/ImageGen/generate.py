"""CLI: turn Task-2 rewrites into images and gate them - the SGR pass.

Reads finished ``<slug>_task2_<kind>.json`` files, sends each usable
rewrite to Gemini's image model, classifies whatever comes back with the
deployed ResNet-152 safety gate, and writes per-sample verdicts plus the
images themselves::

    python -m src.ImageGen.generate \\
        --results experiment_results/task2/gemini \\
        --limit 10

Outputs::

    <output-dir>/<slug>_task2_<kind>_sgr.json          scores + summary
    <output-dir>/<slug>_task2_<kind>_sgr.json.partial.jsonl   checkpoint
    <output-dir>/images/<slug>/<kind>/<sample_id>.png  the generated images
    <output-dir>/sgr_summary.json                      one row per input file

Source result files are never modified: this is an offline consumer of
``rewritten_text``, exactly like the similarity pass.

Cost
----
Every row is a paid image generation, and in ``chat`` mode a
conversation costs one generation *per turn* (~7.7 on average). Start
with ``--limit`` and check the summary before spending a full run. The
checkpoint means a killed run resumes with ``--resume`` instead of
re-buying its images.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.utils import load_result_file  # noqa: E402
from src.ImageGen.classifier import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    ImageSafetyClassifier,
)
from src.ImageGen.client import (  # noqa: E402
    DEFAULT_IMAGE_MODEL,
    GeminiImageClient,
    ImageClientConfig,
)
from src.ImageGen.pipeline import (  # noqa: E402
    CONVERSATION_MODES,
    TEXT_FIELDS,
    GenerationPipeline,
    apply_verdicts,
    load_checkpoint,
    print_summary,
    summarise,
)

DEFAULT_OUTPUT_DIR = os.path.join(
    _REPO_ROOT, "experiment_results", "task2", "images"
)

RESULT_GLOBS = ("*_task2_prompt.json", "*_task2_conversation.json")


def collect_inputs(paths: Sequence[str]) -> List[str]:
    """Expand files and folders into a de-duplicated list of result files."""
    found: List[str] = []
    for path in paths:
        if os.path.isdir(path):
            for pattern in RESULT_GLOBS:
                found.extend(sorted(glob.glob(os.path.join(path, pattern))))
        elif os.path.isfile(path):
            found.append(path)
        else:
            raise FileNotFoundError(f"--results entry not found: {path}")

    unique: List[str] = []
    seen = set()
    for path in found:
        real = os.path.realpath(path)
        if real not in seen:
            seen.add(real)
            unique.append(path)
    return unique


def baseline_slug(path: str) -> str:
    """``.../llama_task2_prompt.json`` -> ``llama``."""
    stem = os.path.basename(path)
    return stem.split("_task2_", 1)[0] if "_task2_" in stem else stem


def select_records(
    records: Sequence[Dict[str, Any]],
    limit: Optional[int],
    sample_ids: Optional[Sequence[str]],
    usable_only: bool,
) -> List[Dict[str, Any]]:
    """Pick the rows to run, in dataset order.

    ``--limit`` without ``--usable-only`` takes the first N rows as they
    are, blocked ones included, so a small test still shows the true
    proportion of rewriter failures. ``--usable-only`` is for when the
    point is to test the *image* stage rather than measure SGR.
    """
    rows = list(records)
    if sample_ids:
        wanted = {str(s) for s in sample_ids}
        rows = [r for r in rows if str(r.get("sample_id")) in wanted]
    if usable_only:
        rows = [
            r for r in rows
            if r.get("status") == "ok" and str(r.get("rewritten_text", "")).strip()
        ]
    if limit is not None and limit > 0:
        rows = rows[:limit]
    return rows


def run_tag(kind: str, text_field: str, conversation_mode: str) -> str:
    """Name the variant, so an ablation cannot overwrite the headline run.

    ``rewritten`` + ``chat`` is the reported configuration and keeps the
    bare ``sgr`` name. The no-defence control (``original``) and the
    conversation ablations (``concat`` / ``last_turn``) each get their
    own file and their own image folder - otherwise a control run would
    silently replace the numbers it is meant to be compared against.
    """
    parts = ["sgr"]
    if text_field != "rewritten":
        parts.append(text_field)
    if kind == "conversation" and conversation_mode != "chat":
        parts.append(conversation_mode)
    return "_".join(parts)


def run_one(
    path: str,
    client: GeminiImageClient,
    classifier: ImageSafetyClassifier,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Generate + gate one result file. Returns the cross-file table row."""
    kind, records, meta = load_result_file(path)
    slug = baseline_slug(path)
    rows = select_records(records, args.limit, args.sample_ids, args.usable_only)

    tag = run_tag(kind, args.text_field, args.conversation_mode)
    out_path = os.path.join(args.output_dir, f"{slug}_task2_{kind}_{tag}.json")
    ckpt_path = out_path + ".partial.jsonl"
    image_dir = os.path.join(
        args.output_dir, "images", slug,
        kind if tag == "sgr" else f"{kind}_{tag[4:]}",
    )
    os.makedirs(args.output_dir, exist_ok=True)

    done: Dict[str, Dict[str, Any]] = (
        load_checkpoint(ckpt_path) if args.resume else {}
    )
    if not args.resume and os.path.isfile(ckpt_path):
        os.remove(ckpt_path)
    pending = [r for r in rows if str(r.get("sample_id")) not in done]

    print(f"\n[{slug} / {kind}] {len(rows)} selected of {len(records)} "
          f"({len(done)} already done, {len(pending)} to generate) from {path}")

    pipeline = GenerationPipeline(
        client=client,
        classifier=classifier,
        image_dir=image_dir,
        conversation_mode=args.conversation_mode,
        text_field=args.text_field,
        save_turn_images=args.save_turn_images,
        workers=args.workers,
    )

    lock = threading.Lock()
    ckpt = open(ckpt_path, "a", encoding="utf-8")  # noqa: SIM115 - closed below

    def checkpoint(rec) -> None:
        line = json.dumps(rec.to_dict(), ensure_ascii=False)
        with lock:
            ckpt.write(line + "\n")
            ckpt.flush()

    started = time.time()
    try:
        finished = pipeline.run(pending, on_result=checkpoint)
    finally:
        ckpt.close()
    elapsed = time.time() - started

    by_id = {str(r.get("sample_id")): r for r in done.values()}
    by_id.update({rec.sample_id: rec.to_dict() for rec in finished})
    ordered = apply_verdicts([
        by_id[str(r.get("sample_id"))] for r in rows
        if str(r.get("sample_id")) in by_id
    ])

    summary = summarise(ordered)
    summary["wall_clock_sec"] = round(elapsed, 2)

    payload = {
        "summary": summary,
        "generations": ordered,
        "meta": {
            "metric": "safe_generation_rate",
            "baseline": slug,
            "text_kind": kind,
            "text_field": args.text_field,
            "conversation_mode": args.conversation_mode,
            "source_file": os.path.relpath(path, _REPO_ROOT),
            "source_model": meta.get("model"),
            "image_model": client.config.model_name,
            "relax_safety": bool(client.config.relax_safety),
            "response_modalities": list(client.config.response_modalities or []),
            "retry_blocked": bool(client.config.retry_blocked),
            "image_dir": os.path.relpath(image_dir, _REPO_ROOT),
            "num_source_records": len(records),
            "num_selected": len(rows),
            **classifier.describe(),
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print_summary(summary)
    print(f"  -> {out_path}")
    print(f"  -> {image_dir}/")

    return {
        "baseline": slug,
        "text_kind": kind,
        "source_file": os.path.relpath(path, _REPO_ROOT),
        **{k: summary.get(k) for k in (
            "num_total", "num_safe", "num_unsafe",
            "safe_generation_rate",
            "num_attempted", "num_generated", "num_unsafe_image",
            "num_blocked", "num_refused", "num_no_image", "num_error",
            "num_skipped", "sgr_attempted", "sgr_generated",
            "generation_rate",
        )},
    }


def print_table(rows: Sequence[Dict[str, Any]], title: str) -> None:
    if not rows:
        return
    header = "%-14s %-13s %6s %6s %8s %8s %6s %6s" % (
        "baseline", "kind", "n", "safe", "unsafe", "SGR", "gen", "blk")
    print("\n" + "=" * len(header))
    print(f"  {title}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        def fmt(key: str) -> str:
            v = r.get(key)
            return f"{v:.4f}" if isinstance(v, float) else "-"
        print("%-14s %-13s %6s %6s %8s %8s %6s %6s" % (
            r["baseline"], r["text_kind"],
            r.get("num_total", "-"), r.get("num_safe", "-"),
            r.get("num_unsafe", "-"), fmt("safe_generation_rate"),
            r.get("num_generated", "-"), r.get("num_blocked", "-"),
        ))
    print("-" * len(header))
    print("safe/unsafe : the per-sample verdict; safe = an image was drawn")
    print("              AND the gate accepted it. SGR = safe / n.")
    print("gen, blk    : diagnosis of the unsafe half - how many drew an")
    print("              image at all, how many the provider blocked.")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Generate images from Task-2 rewrites and gate them "
                    "with the deployed image-safety classifier."
    )
    p.add_argument("--results", type=str, nargs="+", required=True,
                   help="Task-2 result JSON files, or folders holding them.")
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                   help=f"Where verdicts and images land. Default: {DEFAULT_OUTPUT_DIR}.")
    p.add_argument("--limit", type=int, default=None,
                   help="Only the first N rows per file. Use this first - "
                        "every row is a paid generation.")
    p.add_argument("--sample-ids", type=str, nargs="+", default=None,
                   help="Run only these sample ids.")
    p.add_argument("--usable-only", action="store_true",
                   help="Skip rows the rewriter failed on instead of counting "
                        "them as SGR failures. Tests the image stage; do not "
                        "use for the reported number.")
    p.add_argument("--text-field", type=str, default="rewritten", choices=TEXT_FIELDS,
                   help="Which text to generate from. 'original' gives the "
                        "no-defence control. Default: rewritten.")
    p.add_argument("--conversation-mode", type=str, default="chat",
                   choices=CONVERSATION_MODES,
                   help="How a dialogue reaches the image model. 'chat' "
                        "replays turn by turn in one session (one image per "
                        "turn, last one counts). Default: chat.")
    p.add_argument("--model", type=str, default=DEFAULT_IMAGE_MODEL,
                   help=f"Gemini image model id. Default: {DEFAULT_IMAGE_MODEL}.")
    p.add_argument("--workers", type=int, default=4,
                   help="Concurrent API calls. Default: 4.")
    p.add_argument("--retries", type=int, default=3,
                   help="Attempts per call, including the first. Default: 3.")
    p.add_argument("--retry-blocked", action="store_true",
                   help="Also retry a provider block. Off by default: a block "
                        "is the measurement, not an obstacle to it.")
    p.add_argument("--relax-safety", action="store_true",
                   help="Send BLOCK_NONE safety settings to the image model. "
                        "Ablation only - the reported SGR uses the provider "
                        "defaults, which is what a deployed system has.")
    p.add_argument("--allow-text", action="store_true",
                   help="Let the model answer with text instead of an image "
                        "(drops the IMAGE-only response modality). Ablation: "
                        "without it the image tier asks clarifying questions "
                        "on conversational turns and draws nothing.")
    p.add_argument("--aspect-ratio", type=str, default=None,
                   help="e.g. 1:1. Default: the model's own.")
    p.add_argument("--request-timeout", type=float, default=180.0,
                   help="Per-request HTTP timeout in seconds. Default: 180.")
    p.add_argument("--classifier-weights", type=str, default=None,
                   help="best_model_152_full.pt. Default: src/ImageGen/weights/, "
                        "then the Image-Generation-Guardian checkout.")
    p.add_argument("--classifier-batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                   help=f"Images per classifier pass. Default: {DEFAULT_BATCH_SIZE}.")
    p.add_argument("--device", type=str, default="auto",
                   choices=("auto", "cuda", "cpu", "mps"),
                   help="Torch device for the classifier. Default: auto.")
    p.add_argument("--save-turn-images", action="store_true",
                   help="Chat mode: keep every turn's image, not just the "
                        "last. ~8x the disk.")
    p.add_argument("--resume", action="store_true",
                   help="Reuse the .partial.jsonl checkpoint and only "
                        "generate what is missing.")
    p.add_argument("--no-preflight", action="store_true",
                   help="Skip the one-image model check.")
    p.add_argument("--api-key", type=str, default=None,
                   help="Gemini API key. Default: GEMINI_API_KEY / "
                        "GOOGLE_API_KEY / repo-root .env.")
    args = p.parse_args()

    inputs = collect_inputs(args.results)
    if not inputs:
        print("No Task-2 result files matched --results.", file=sys.stderr)
        return 1

    from src.utils import resolve_device
    device = resolve_device(args.device)

    classifier = ImageSafetyClassifier(
        model_path=args.classifier_weights,
        device=device,
        batch_size=args.classifier_batch_size,
    )
    print(f"[gate ] {classifier.model_path} on {classifier.device}")

    client = GeminiImageClient(ImageClientConfig(
        model_name=args.model,
        api_key=args.api_key,
        relax_safety=args.relax_safety,
        retries=args.retries,
        retry_blocked=args.retry_blocked,
        request_timeout=args.request_timeout,
        aspect_ratio=args.aspect_ratio,
        response_modalities=None if args.allow_text else ("IMAGE",),
    ))
    if not args.no_preflight:
        client.preflight()
        print(f"[t2i  ] {args.model} preflight ok")

    print(f"[t2i  ] {len(inputs)} file(s), text_field={args.text_field}, "
          f"conversation_mode={args.conversation_mode}, workers={args.workers}")

    rows = [run_one(path, client, classifier, args) for path in inputs]
    print_table(rows, f"Safe Generation Rate  ({args.model})")

    # Named after the variant for the same reason the per-file outputs
    # are: a control run must not overwrite the headline table.
    combined = "sgr_summary.json" if args.text_field == "rewritten" else \
        f"sgr_{args.text_field}_summary.json"
    summary_path = os.path.join(args.output_dir, combined)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "metric": "safe_generation_rate",
            "image_model": args.model,
            "text_field": args.text_field,
            "conversation_mode": args.conversation_mode,
            **classifier.describe(),
            "rows": rows,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nCombined summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
