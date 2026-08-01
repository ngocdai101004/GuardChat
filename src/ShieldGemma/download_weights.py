"""CLI: download ShieldGemma-2B weights into ``src/ShieldGemma/weights/``.

``google/shieldgemma-2b`` is gated on HuggingFace. Before running:

    1. Accept the Gemma licence at
       https://huggingface.co/google/shieldgemma-2b
    2. Provide a token - either ``huggingface-cli login``, or
       ``export HF_TOKEN=hf_...``, or a ``HF_TOKEN=...`` line in the
       repo-root ``.env`` file (see :mod:`src.utils.hf_token`).
    3. Download::

           python -m src.ShieldGemma.download_weights

       which writes a snapshot to ``src/ShieldGemma/weights/shieldgemma-2b/``.
       Subsequent loads are fully offline.

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
    HF_TOKEN_ENV_KEYS as ENV_KEYS,
    ensure_local_snapshot,
    resolve_hf_token,
)
from src.ShieldGemma.model import DEFAULT_LOCAL_DIR, DEFAULT_MODEL_NAME  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(
        description="Download ShieldGemma-2B weights to a local folder."
    )
    p.add_argument("--repo-id", type=str, default=DEFAULT_MODEL_NAME,
                   help=f"HuggingFace repo id. Default: {DEFAULT_MODEL_NAME}.")
    p.add_argument("--local-dir", type=str, default=DEFAULT_LOCAL_DIR,
                   help=f"Destination folder. Default: {DEFAULT_LOCAL_DIR}.")
    p.add_argument("--token", type=str, default=None,
                   help="Override the HuggingFace token. By default reads "
                        f"{' / '.join(ENV_KEYS)} from the environment or the "
                        "repo-root .env file.")
    p.add_argument("--force", action="store_true",
                   help="Re-download even when the folder already holds a "
                        "snapshot.")
    args = p.parse_args()

    token = resolve_hf_token(args.token)
    print("[ShieldGemma] gated repo: accept the Gemma licence at")
    print(f"              https://huggingface.co/{args.repo_id}")
    print(f"[ShieldGemma] token: {'provided' if token else 'none (using HF cache login)'}")

    if args.force:
        marker = os.path.join(args.local_dir, "config.json")
        if os.path.isfile(marker):
            os.remove(marker)

    ensure_local_snapshot(
        args.local_dir,
        repo_id=args.repo_id,
        token=token,
        log_prefix="ShieldGemma",
    )
    print(f"[ShieldGemma] done. Local folder: {args.local_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
