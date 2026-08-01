"""CLI: evaluate Llama-Guard-3-8B on GuardChat Task 1 (zero-shot).

    python -m src.LlamaGuard.eval_recognition \
        --test build_dataset/dataset/final_df_test.json \
        --mode guardchat \
        --text-kind all \
        --output-dir experiment_results/task1/llamaguard

``--text-kind all`` writes one JSON file per input representation:

    llamaguard_task1_prompt.json        enhanced prompt   (X_single)
    llamaguard_task1_raw_prompt.json    original seed prompt
    llamaguard_task1_conversation.json  multi-turn dialogue (X_conv)

Output schema is identical to the other Task-1 baselines
(``shieldgemma_task1_*.json``, ``safeguider_task1.json``, ...) so a
single aggregator can compose Table 1 across every row.

There is no ``train_recognition.py`` for this baseline - Llama-Guard is
used strictly zero-shot.

Long runs are checkpointed to ``<output>.partial.jsonl`` after every
sample; re-running with ``--resume`` skips whatever is already there.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

# Repo-root sys.path bootstrap.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.utils import (  # noqa: E402
    DTYPE_CHOICES,
    TEXT_KINDS,
    evaluate_kind,
    load_guardchat,
    normalise_text_kind,
    print_metrics,
    save_kind,
)
from src.LlamaGuard.model import DEFAULT_LOCAL_DIR, DEFAULT_MODEL_NAME  # noqa: E402
from src.LlamaGuard.recognition import CONV_FORMATS, RecognitionPipeline  # noqa: E402
from src.LlamaGuard.taxonomy import MODES  # noqa: E402


SLUG = "llamaguard"
DEFAULT_OUTPUT_DIR = os.path.join(_REPO_ROOT, "experiment_results", "task1", SLUG)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Evaluate Llama-Guard-3-8B on GuardChat Task 1 (zero-shot)."
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
                        "id. Default: src/LlamaGuard/weights/Llama-Guard-3-8B, "
                        f"auto-populated from {DEFAULT_MODEL_NAME}.")
    p.add_argument("--no-auto-download", action="store_true",
                   help="Fail instead of fetching the snapshot when the local "
                        "weights folder is empty.")
    p.add_argument("--mode", type=str, default="guardchat",
                   choices=list(MODES) + ["custom"],
                   help="'guardchat' = render the prompt with GuardChat's six "
                        "categories in place of S1-S14 (class counts match; "
                        "'custom' is a deprecated alias). 'native' = the shipped "
                        "S1-S14 taxonomy + a lossy mapping (shocking never fires).")
    p.add_argument("--dtype", type=str, default="auto", choices=list(DTYPE_CHOICES),
                   help="Weight dtype. 'auto' = float32 on CPU, bfloat16 "
                        "elsewhere. int8/nf4 need bitsandbytes (CUDA only).")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cuda", "mps", "cpu"],
                   help="Compute device. 'auto' prefers cuda > mps > cpu.")
    p.add_argument("--text-kind", type=str, default="all",
                   choices=list(TEXT_KINDS) + ["single", "all"],
                   help="Input representation. 'all' runs prompt, raw_prompt "
                        "and conversation in one go (three output files).")
    p.add_argument("--conv-format", type=str, default="concat",
                   choices=list(CONV_FORMATS),
                   help="How the dialogue is fed for --text-kind conversation. "
                        "'concat' flattens it into one user message, matching "
                        "the other baselines' X_conv. 'turns' forwards the real "
                        "turn list, merging consecutive same-role turns because "
                        "the chat template demands strict alternation.")
    p.add_argument("--no-role-prefix", action="store_true",
                   help="Concatenate conversation turns without the 'user: ' "
                        "prefix. Only affects the recorded text and "
                        "--conv-format concat.")
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
        kinds = [normalise_text_kind(args.text_kind)]

    pipe = RecognitionPipeline.from_pretrained(
        weights=args.weights,
        mode=args.mode,
        device=args.device,
        dtype=args.dtype,
        token=args.token,
        role_prefix=not args.no_role_prefix,
        conv_format=args.conv_format,
        auto_download=not args.no_auto_download,
    )
    print(f"Loaded {args.weights} on {pipe.model.device} ({pipe.model.dtype_name})")
    print(f"Mode '{pipe.mode}', conversation format '{pipe.conv_format}'")

    unreachable = pipe.unreachable_categories
    if unreachable:
        print(f"  NOTE: mode '{pipe.mode}' has no S-code for {unreachable} - "
              f"those categories can never be predicted and will score F1 = 0.")

    meta: Dict[str, object] = {
        "model": args.weights,
        "mode": pipe.mode,
        "dtype": pipe.model.dtype_name,
        "device": str(pipe.model.device),
        "test": args.test,
        "num_samples": len(samples),
        "conv_format": pipe.conv_format,
        "role_prefix": not args.no_role_prefix,
        "unreachable_categories": unreachable,
    }

    written: List[str] = []
    for kind in kinds:
        print(f"\n=== {kind} ===")
        res = evaluate_kind(
            lambda pending, k, cb: pipe.predict_samples(pending, kind=k,
                                                        on_prediction=cb),
            samples, kind, args.output_dir, SLUG, resume=args.resume,
        )
        out_path = save_kind(res, kind, meta, keep_checkpoint=args.keep_checkpoint)
        written.append(out_path)
        print(f"Saved -> {out_path}")

        # An "unsafe" verdict carrying a code outside the active taxonomy
        # collapses to an all-zero vector, i.e. it is scored as safe.
        # Surface it instead of letting it quietly depress recall.
        unmapped = [r for r in res["predictions"] if r.get("unmapped_scodes")]
        if unmapped:
            codes = sorted({c for r in unmapped for c in r["unmapped_scodes"]})
            print(f"  WARNING: {len(unmapped)}/{len(res['predictions'])} unsafe "
                  f"verdicts used codes outside mode '{pipe.mode}' ({codes}) and "
                  f"were counted as safe. See 'unmapped_scodes' in the output.")
        print_metrics(res["metrics"])

    print("\nDone. Files written:")
    for pth in written:
        print(f"  {pth}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
