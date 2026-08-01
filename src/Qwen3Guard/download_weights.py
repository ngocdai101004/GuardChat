"""CLI: download Qwen3Guard-Gen-8B weights into ``src/Qwen3Guard/weights/``.

``Qwen/Qwen3Guard-Gen-8B`` is Apache-2.0 and **not gated**, so no licence
click-through is needed and a token is optional (an anonymous download
works, it is just rate-limited harder). To use one anyway: run
``huggingface-cli login``, ``export HF_TOKEN=hf_...``, or put a
``HF_TOKEN=...`` line in the repo-root ``.env`` file (see
:mod:`src.utils.hf_token`).

    python -m src.Qwen3Guard.download_weights

writes a complete snapshot (~16 GB, 5 safetensors shards) to
``src/Qwen3Guard/weights/Qwen3Guard-Gen-8B/``. Subsequent loads are fully
offline.

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
from src.Qwen3Guard.model import DEFAULT_LOCAL_DIR, DEFAULT_MODEL_NAME  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(
        description="Download Qwen3Guard-Gen-8B weights to a local folder."
    )
    p.add_argument("--repo-id", type=str, default=DEFAULT_MODEL_NAME,
                   help=f"HuggingFace repo id. Default: {DEFAULT_MODEL_NAME}.")
    p.add_argument("--local-dir", type=str, default=DEFAULT_LOCAL_DIR,
                   help=f"Destination folder. Default: {DEFAULT_LOCAL_DIR}.")
    p.add_argument("--token", type=str, default=None,
                   help="Override the HuggingFace token. By default reads "
                        f"{' / '.join(ENV_KEYS)} from the environment or the "
                        "repo-root .env file. Optional - the repo is public.")
    p.add_argument("--force", action="store_true",
                   help="Re-download even when the folder already holds a "
                        "snapshot.")
    args = p.parse_args()

    ignore = list(DEFAULT_IGNORE_PATTERNS)

    token = resolve_hf_token(args.token)
    print(f"[Qwen3Guard] public repo (Apache-2.0): https://huggingface.co/{args.repo_id}")
    print(f"[Qwen3Guard] token: {'provided' if token else 'none (anonymous download)'}")
    print(f"[Qwen3Guard] ignore patterns: {ignore}")

    if args.force:
        marker = os.path.join(args.local_dir, "config.json")
        if os.path.isfile(marker):
            os.remove(marker)

    ensure_local_snapshot(
        args.local_dir,
        repo_id=args.repo_id,
        token=token,
        ignore_patterns=ignore,
        log_prefix="Qwen3Guard",
    )
    print(f"[Qwen3Guard] done. Local folder: {args.local_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
