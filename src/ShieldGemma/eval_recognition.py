"""CLI: evaluate ShieldGemma-2B on GuardChat Task 1 (zero-shot).

    python -m src.ShieldGemma.eval_recognition \
        --test build_dataset/dataset/final_df_test.json \
        --weights google/shieldgemma-2b \
        --mode guardchat \
        --text-kind all \
        --output-dir experiment_results/task1/shieldgemma

``--text-kind all`` writes one JSON file per input representation:

    shieldgemma_task1_prompt.json        enhanced prompt   (X_single)
    shieldgemma_task1_raw_prompt.json    original seed prompt
    shieldgemma_task1_conversation.json  concatenated turns (X_conv)

Each file follows the schema shared with the other Task-1 baselines::

    { "<kind>": { "metrics": {...}, "predictions": [...] },
      "meta":   {...} }

``metrics`` is computed with the default threshold as a convenience;
every prediction also carries the raw per-policy ``P(Yes)`` scores, so
the reported numbers can be recomputed under a different threshold or a
different policy-to-category mapping without touching the GPU again.

Long runs are checkpointed to ``<output>.partial.jsonl`` after every
sample; re-running with ``--resume`` skips whatever is already there.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

# Repo-root sys.path bootstrap.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.utils import CATEGORIES, load_guardchat, summarise_recognition  # noqa: E402
from src.ShieldGemma.model import DEFAULT_LOCAL_DIR, DEFAULT_MODEL_NAME  # noqa: E402
from src.ShieldGemma.recognition import (  # noqa: E402
    TEXT_KINDS,
    RecognitionPipeline,
    RecognitionPrediction,
    normalise_kind,
)
from src.ShieldGemma.taxonomy import MODES, policies_for_mode  # noqa: E402


DEFAULT_OUTPUT_DIR = os.path.join(_REPO_ROOT, "experiment_results", "task1", "shieldgemma")


def _output_path(output_dir: str, kind: str) -> str:
    return os.path.join(output_dir, f"shieldgemma_task1_{kind}.json")


# --------------------------- Checkpointing --------------------------- #

def _load_checkpoint(path: str) -> Dict[str, dict]:
    """Read ``<output>.partial.jsonl`` into ``{sample_id: record}``."""
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


def _evaluate_one_kind(
    pipe: RecognitionPipeline,
    samples,
    kind: str,
    output_dir: str,
    resume: bool,
) -> Dict[str, object]:
    kind = normalise_kind(kind)
    out_path = _output_path(output_dir, kind)
    ckpt_path = out_path + ".partial.jsonl"

    done = _load_checkpoint(ckpt_path) if resume else {}
    if done:
        print(f"  resuming: {len(done)} sample(s) already scored in {ckpt_path}")
    if not resume and os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    pending = [s for s in samples if str(s.sample_id) not in done]

    ckpt_file = open(ckpt_path, "a", encoding="utf-8")

    def _append(pred: RecognitionPrediction) -> None:
        ckpt_file.write(json.dumps(pred.to_dict(), ensure_ascii=False) + "\n")
        ckpt_file.flush()

    try:
        fresh = pipe.predict_samples(pending, kind=kind, on_prediction=_append)
    finally:
        ckpt_file.close()

    for pred in fresh:
        done[str(pred.sample_id)] = pred.to_dict()

    # Restore the original dataset order.
    records = [done[str(s.sample_id)] for s in samples if str(s.sample_id) in done]

    y_true = [list(s.label_vector) for s in samples if str(s.sample_id) in done]
    # Index by CATEGORIES rather than the dict's own order - checkpoint
    # records round-trip through JSON and must stay column-aligned.
    y_pred = [[int(r["multi_label"].get(c, 0)) for c in CATEGORIES] for r in records]
    metrics = summarise_recognition(y_true, y_pred)
    return {"metrics": metrics, "predictions": records, "path": out_path}


def main() -> int:
    p = argparse.ArgumentParser(
        description="Evaluate ShieldGemma-2B on GuardChat Task 1 (zero-shot)."
    )
    p.add_argument("--test", type=str,
                   default=os.path.join(_REPO_ROOT, "build_dataset", "dataset",
                                        "final_df_test.json"),
                   help="Local JSON/JSONL path or a HuggingFace repo id. "
                        "Default: build_dataset/dataset/final_df_test.json.")
    p.add_argument("--split", type=str, default="test",
                   help="HF split when --test is a repo id. Default: test.")
    p.add_argument("--weights", type=str, default=DEFAULT_LOCAL_DIR,
                   help="Local snapshot dir (downloaded on first use) or an HF "
                        f"id. Default: src/ShieldGemma/weights/shieldgemma-2b, "
                        f"auto-populated from {DEFAULT_MODEL_NAME}.")
    p.add_argument("--no-auto-download", action="store_true",
                   help="Fail instead of fetching the snapshot when the local "
                        "weights folder is empty.")
    p.add_argument("--mode", type=str, default="guardchat", choices=list(MODES),
                   help="'guardchat' = six GuardChat-aligned policies (class "
                        "counts match). 'native' = ShieldGemma's four published "
                        "policies + a lossy mapping (shocking never fires).")
    p.add_argument("--dtype", type=str, default="auto",
                   choices=["auto", "bfloat16", "float16", "float32", "int8", "nf4"],
                   help="Weight dtype. 'auto' = float32 on CPU, bfloat16 "
                        "elsewhere. int8/nf4 need bitsandbytes (CUDA only).")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cuda", "mps", "cpu"],
                   help="Compute device. 'auto' prefers cuda > mps > cpu.")
    p.add_argument("--text-kind", type=str, default="all",
                   choices=list(TEXT_KINDS) + ["single", "all"],
                   help="Input representation. 'all' runs prompt, raw_prompt "
                        "and conversation in one go (three output files).")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="P(Yes) cut-off for a policy to fire. Default: 0.5. "
                        "Raw scores are always stored, so this can be revisited "
                        "offline.")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Policy prompts scored per forward pass. Default: 4.")
    p.add_argument("--max-length", type=int, default=4096,
                   help="Tokenizer truncation cap. Default: 4096.")
    p.add_argument("--no-role-prefix", action="store_true",
                   help="Concatenate conversation turns without the 'user: ' "
                        "prefix. Default keeps the prefix, matching the other "
                        "Task-1 baselines.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap the number of samples (smoke tests).")
    p.add_argument("--token", type=str, default=None,
                   help="HuggingFace token override. Default reads HF_TOKEN "
                        "from the environment or the repo-root .env file.")
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                   help=f"Where the per-kind JSON files land. "
                        f"Default: {DEFAULT_OUTPUT_DIR}.")
    p.add_argument("--resume", action="store_true",
                   help="Reuse any '<output>.partial.jsonl' checkpoint instead "
                        "of re-scoring from scratch.")
    p.add_argument("--keep-checkpoint", action="store_true",
                   help="Keep the .partial.jsonl file after a successful run.")
    args = p.parse_args()

    print(f"Loading test split from {args.test} (split={args.split})")
    samples = load_guardchat(args.test, split=args.split)
    if args.limit:
        samples = samples[: int(args.limit)]
    print(f"  -> {len(samples)} samples")

    kinds: List[str]
    if args.text_kind == "all":
        kinds = list(TEXT_KINDS)
    else:
        kinds = [normalise_kind(args.text_kind)]

    policies = policies_for_mode(args.mode)
    print(f"Mode '{args.mode}': {len(policies)} policies "
          f"({', '.join(policies)}) -> {len(samples) * len(policies)} forward "
          f"passes per input representation")

    pipe = RecognitionPipeline.from_pretrained(
        weights=args.weights,
        mode=args.mode,
        device=args.device,
        dtype=args.dtype,
        threshold=args.threshold,
        token=args.token,
        batch_size=args.batch_size,
        max_length=args.max_length,
        role_prefix=not args.no_role_prefix,
        auto_download=not args.no_auto_download,
    )
    print(f"Loaded {args.weights} on {pipe.model.device} "
          f"({pipe.model.dtype_name})")

    unreachable = pipe.unreachable_categories
    if unreachable:
        print(f"  NOTE: mode '{args.mode}' has no policy for "
              f"{unreachable} - those categories can never be predicted "
              f"and will score F1 = 0.")

    os.makedirs(args.output_dir, exist_ok=True)

    meta: Dict[str, object] = {
        "model": args.weights,
        "mode": args.mode,
        "policies": policies,
        "threshold": args.threshold,
        "dtype": pipe.model.dtype_name,
        "device": str(pipe.model.device),
        "test": args.test,
        "num_samples": len(samples),
        "role_prefix": not args.no_role_prefix,
        "unreachable_categories": unreachable,
    }

    written: List[str] = []
    for kind in kinds:
        print(f"\n=== {kind} ===")
        res = _evaluate_one_kind(
            pipe, samples, kind, args.output_dir, resume=args.resume
        )
        metrics = res["metrics"]
        out_path = str(res["path"])

        payload = {
            kind: {"metrics": metrics, "predictions": res["predictions"]},
            "meta": {**meta, "text_kind": kind},
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        written.append(out_path)
        print(f"Saved -> {out_path}")

        for key in ["macro_f1", "recall_binary", "asr"]:
            print(f"  {key:>14}: {metrics[key]:.4f}")
        for k, v in metrics.items():
            if k.startswith("f1_"):
                print(f"  {k:>14}: {v:.4f}")

        if not args.keep_checkpoint:
            ckpt = out_path + ".partial.jsonl"
            if os.path.exists(ckpt):
                os.remove(ckpt)

    print("\nDone. Files written:")
    for pth in written:
        print(f"  {pth}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
