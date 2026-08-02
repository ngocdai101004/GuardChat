"""CLI: download Llama-3.1-8B-Instruct weights into ``src/Llama/weights/``.

``meta-llama/Llama-3.1-8B-Instruct`` is **gated** on HuggingFace (same
access flow as Llama-Guard-3-8B). Before the first run:

    1. Visit https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
       and accept Meta's licence. Approval is usually instant.
    2. Provide a token - any one of:
       ``huggingface-cli login``, ``export HF_TOKEN=hf_...``, or a
       ``HF_TOKEN=...`` line in the repo-root ``.env`` (see
       :mod:`src.utils.hf_token`).
    3. Download::

           python -m src.Llama.download_weights

       which writes a complete snapshot (~16 GB, 4 safetensors shards)
       to ``src/Llama/weights/Llama-3.1-8B-Instruct/``. Subsequent loads
       are fully offline.

Running this ahead of time is optional - the benchmark populates the
same folder on its first run - but doing it separately keeps the
download step distinct from the evaluation step on a slow link, and
surfaces a licence problem before the GPU is booked.
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
from src.Llama.model import DEFAULT_LOCAL_DIR, DEFAULT_MODEL_NAME  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(
        description="Download Llama-3.1-8B-Instruct weights to a local folder."
    )
    p.add_argument("--repo-id", type=str, default=DEFAULT_MODEL_NAME,
                   help=f"HuggingFace repo id. Default: {DEFAULT_MODEL_NAME}.")
    p.add_argument("--local-dir", type=str, default=DEFAULT_LOCAL_DIR,
                   help=f"Destination folder. Default: {DEFAULT_LOCAL_DIR}.")
    p.add_argument("--include-original", action="store_true",
                   help="Also fetch the duplicated raw .pth weights under "
                        "`original/`. Doubles the disk footprint and nothing "
                        "in this repo reads them.")
    p.add_argument("--token", type=str, default=None,
                   help="Override the HuggingFace token. By default reads "
                        f"{' / '.join(ENV_KEYS)} from the environment or the "
                        "repo-root .env file. Required: this repo is gated.")
    p.add_argument("--force", action="store_true",
                   help="Re-download even when the folder already holds a "
                        "snapshot.")
    args = p.parse_args()

    ignore = list(DEFAULT_IGNORE_PATTERNS)
    if args.include_original:
        ignore.remove("original/*")

    token = resolve_hf_token(args.token)
    print(f"[Llama-3.1] gated repo: https://huggingface.co/{args.repo_id}")
    print("[Llama-3.1] accept Meta's licence on that page first, or the "
          "download fails with 401/403.")
    print(f"[Llama-3.1] token: {'provided' if token else 'NONE - this will fail'}")
    print(f"[Llama-3.1] ignore patterns: {ignore}")

    if args.force:
        marker = os.path.join(args.local_dir, "config.json")
        if os.path.isfile(marker):
            os.remove(marker)

    ensure_local_snapshot(
        args.local_dir,
        repo_id=args.repo_id,
        token=token,
        ignore_patterns=ignore,
        log_prefix="Llama-3.1",
    )
    print(f"[Llama-3.1] done. Local folder: {args.local_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
