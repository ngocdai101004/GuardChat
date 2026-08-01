"""CLI: download Llama-Guard-3-8B weights into ``src/LlamaGuard/weights/``.

Llama-Guard-3-8B is gated on HuggingFace. Before running this script:

    1. Visit https://huggingface.co/meta-llama/Llama-Guard-3-8B and
       request access (Meta's licence agreement). Approval is usually
       quick.
    2. Provide a token - either ``huggingface-cli login``, or
       ``export HF_TOKEN=hf_...``, or a ``HF_TOKEN=...`` line in the
       repo-root ``.env`` file (see :mod:`src.utils.hf_token`).
    3. Download::

           python -m src.LlamaGuard.download_weights

       which writes a complete snapshot to
       ``src/LlamaGuard/weights/Llama-Guard-3-8B/``. Subsequent loads
       are fully offline.

By default we skip the optional ``original/`` folder shipped by Meta
(it duplicates the weights as raw ``.consolidated.pth`` files - the
HuggingFace ``.safetensors`` weights are sufficient for the
``transformers`` loader and save ~16 GB of disk).

Running this ahead of time is optional - the benchmark populates the same
folder on its first run - but doing it separately keeps the download step
distinct from the evaluation step on a slow link.
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.utils import (  # noqa: E402
    DEFAULT_IGNORE_PATTERNS,
    HF_TOKEN_ENV_KEYS as ENV_KEYS,
    ensure_local_snapshot,
    resolve_hf_token,
)
from src.LlamaGuard.model import DEFAULT_LOCAL_DIR, DEFAULT_MODEL_NAME  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(
        description="Download Llama-Guard-3-8B weights to a local folder."
    )
    p.add_argument("--repo-id", type=str, default=DEFAULT_MODEL_NAME,
                   help=f"HuggingFace repo id. Default: {DEFAULT_MODEL_NAME}.")
    p.add_argument("--local-dir", type=str, default=DEFAULT_LOCAL_DIR,
                   help=f"Destination folder. Default: {DEFAULT_LOCAL_DIR}.")
    p.add_argument("--include-original", action="store_true",
                   help="Also fetch the duplicated raw .pth weights under "
                        "`original/`. Doubles the disk footprint.")
    p.add_argument("--token", type=str, default=None,
                   help="Override the HuggingFace token. By default reads "
                        f"{' / '.join(ENV_KEYS)} from the environment or the "
                        "repo-root .env file.")
    p.add_argument("--force", action="store_true",
                   help="Re-download even when the folder already holds a "
                        "snapshot.")
    args = p.parse_args()

    ignore = list(DEFAULT_IGNORE_PATTERNS)
    if args.include_original:
        ignore.remove("original/*")

    token = resolve_hf_token(args.token)
    print("[LlamaGuard] gated repo: make sure you have accepted the licence at")
    print(f"             https://huggingface.co/{args.repo_id}")
    print(f"[LlamaGuard] token: {'provided' if token else 'none (using HF cache login)'}")
    print(f"[LlamaGuard] ignore patterns: {ignore}")

    if args.force:
        marker = os.path.join(args.local_dir, "config.json")
        if os.path.isfile(marker):
            os.remove(marker)

    ensure_local_snapshot(
        args.local_dir,
        repo_id=args.repo_id,
        token=token,
        ignore_patterns=ignore,
        log_prefix="LlamaGuard",
    )
    print(f"[LlamaGuard] done. Local folder: {args.local_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
