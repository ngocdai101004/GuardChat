"""CLI: fetch ``sentence-transformers/all-mpnet-base-v2`` into ``src/SBERT/weights/``.

Open access - no licence to accept and no token needed. The snapshot is
~440 MB once the redundant TensorFlow / Flax / ONNX / OpenVINO copies are
filtered out (see :data:`src.SBERT.model.IGNORE_PATTERNS`); the full repo
is several times that.

    python -m src.SBERT.download_weights

Running it ahead of time is optional - :func:`src.SBERT.model.load_encoder`
populates the same folder on first use - but it keeps a slow download out
of the middle of an evaluation run.
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.utils import ensure_local_snapshot, resolve_hf_token  # noqa: E402
from src.SBERT.model import (  # noqa: E402
    DEFAULT_LOCAL_DIR,
    DEFAULT_MODEL_NAME,
    IGNORE_PATTERNS,
    read_max_seq_length,
)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Download the SBERT sentence encoder to a local folder."
    )
    p.add_argument("--repo-id", type=str, default=DEFAULT_MODEL_NAME,
                   help=f"HuggingFace repo id. Default: {DEFAULT_MODEL_NAME}.")
    p.add_argument("--local-dir", type=str, default=DEFAULT_LOCAL_DIR,
                   help=f"Destination folder. Default: {DEFAULT_LOCAL_DIR}.")
    p.add_argument("--token", type=str, default=None,
                   help="HuggingFace token. Not required for this repo; "
                        "read from the environment or repo-root .env when "
                        "present, which helps behind a rate-limited IP.")
    p.add_argument("--force", action="store_true",
                   help="Re-download even when the folder already holds a "
                        "snapshot.")
    args = p.parse_args()

    if args.force:
        marker = os.path.join(args.local_dir, "config.json")
        if os.path.isfile(marker):
            os.remove(marker)

    print(f"[sbert] repo: https://huggingface.co/{args.repo_id}")
    print(f"[sbert] ignore patterns: {IGNORE_PATTERNS}")

    ensure_local_snapshot(
        args.local_dir,
        repo_id=args.repo_id,
        token=resolve_hf_token(args.token),
        ignore_patterns=IGNORE_PATTERNS,
        log_prefix="sbert",
    )

    # Confirm the sentence-transformers side-cars survived the filter -
    # without them we would fall back to a guessed sequence length and
    # could not verify the pooling mode.
    seq = read_max_seq_length(args.local_dir)
    if seq is None:
        print("[sbert] WARNING: sentence_bert_config.json missing; "
              "max_seq_length will fall back to the built-in default.")
    else:
        print(f"[sbert] max_seq_length from checkpoint: {seq}")

    print(f"[sbert] done. Local folder: {args.local_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
