"""CLI: download Llama-Guard-3-8B weights into ``src/LlamaGuard/weights/``.

Llama-Guard-3-8B is gated on HuggingFace. Before running this script:

    1. Visit https://huggingface.co/meta-llama/Llama-Guard-3-8B and
       request access (Meta's licence agreement). Approval is usually
       quick.
    2. Authenticate from the CLI:

           huggingface-cli login

       or export ``HF_TOKEN=hf_...`` in your shell.

    3. Download::

           python -m src.LlamaGuard.download_weights

       which writes a complete snapshot to
       ``src/LlamaGuard/weights/Llama-Guard-3-8B/``. Subsequent loads
       are fully offline.

By default we skip the optional ``original/`` folder shipped by Meta
(it duplicates the weights as raw ``.consolidated.pth`` files - the
HuggingFace ``.safetensors`` weights are sufficient for the
``transformers`` loader and save ~16 GB of disk).
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.LlamaGuard.model import DEFAULT_LOCAL_DIR, DEFAULT_MODEL_NAME  # noqa: E402


# Patterns to skip during snapshot_download. We keep the
# ``.safetensors`` weights, tokenizer files, and config; everything
# else is optional and bloats the local copy.
DEFAULT_IGNORE_PATTERNS = [
    "original/*",
    "*.bin",            # legacy pytorch_model.bin (we use safetensors)
    "*.gguf",           # llama.cpp weights
    "*.h5", "*.msgpack", "tf_model.h5", "flax_model.msgpack",
]


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
                   help="Override HuggingFace token. By default reads from "
                        "`HF_TOKEN` env var or `huggingface-cli login`.")
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
    print(f"[LlamaGuard] downloading {args.repo_id!r} -> {args.local_dir}")
    print("[LlamaGuard] this is a gated repo; make sure you have accepted")
    print("             the licence at https://huggingface.co/meta-llama/Llama-Guard-3-8B")
    print(f"[LlamaGuard] ignore patterns: {ignore}")

    snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        token=args.token,
        ignore_patterns=ignore,
    )
    print(f"[LlamaGuard] done. Local folder: {args.local_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
