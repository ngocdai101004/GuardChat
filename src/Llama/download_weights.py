"""CLI: download Llama-3.1-8B-Instruct weights into ``src/Llama/weights/``.

Llama-3.1-8B-Instruct is gated on HuggingFace (same access flow as
Llama-Guard-3-8B). Before the first run:

    1. Visit https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
       and request access (Meta's licence).
    2. Authenticate from the CLI:

           huggingface-cli login

       or export ``HF_TOKEN=hf_...``.

    3. Download::

           python -m src.Llama.download_weights

       which writes a complete snapshot to
       ``src/Llama/weights/Llama-3.1-8B-Instruct/``. Subsequent loads
       are fully offline.
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.Llama.model import DEFAULT_LOCAL_DIR, DEFAULT_MODEL_NAME  # noqa: E402


DEFAULT_IGNORE_PATTERNS = [
    "original/*",
    "*.bin",
    "*.gguf",
    "*.h5", "*.msgpack", "tf_model.h5", "flax_model.msgpack",
]


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
                        "`original/`. Doubles the disk footprint.")
    p.add_argument("--token", type=str, default=None,
                   help="Override HF token. Reads HF_TOKEN env var by default.")
    args = p.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is required (pip install huggingface_hub)."
        ) from e

    ignore = list(DEFAULT_IGNORE_PATTERNS)
    if args.include_original:
        ignore.remove("original/*")

    os.makedirs(args.local_dir, exist_ok=True)
    print(f"[Llama-3.1] downloading {args.repo_id!r} -> {args.local_dir}")
    print("[Llama-3.1] this is a gated repo; make sure you have accepted")
    print("            the licence at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
    print(f"[Llama-3.1] ignore patterns: {ignore}")

    snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        token=args.token,
        ignore_patterns=ignore,
    )
    print(f"[Llama-3.1] done. Local folder: {args.local_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
