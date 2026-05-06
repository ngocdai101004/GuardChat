"""CLI: download Qwen2.5-7B-Instruct weights into ``src/Qwen/weights/``.

Qwen2.5-7B-Instruct is **not** gated on HuggingFace - no licence
acceptance is required. You can run this script directly:

    python -m src.Qwen.download_weights

which writes a complete snapshot to
``src/Qwen/weights/Qwen2.5-7B-Instruct/``. Subsequent loads are fully
offline.

Roughly ~15 GB of disk (BF16/FP16 ``.safetensors`` shards). We skip
flax/tf duplicates and the 4-bit GPTQ checkpoint (a separate repo)
to keep the local copy lean.
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.Qwen.model import DEFAULT_LOCAL_DIR, DEFAULT_MODEL_NAME  # noqa: E402


DEFAULT_IGNORE_PATTERNS = [
    "*.bin",            # legacy pytorch_model.bin (we use safetensors)
    "*.gguf",           # llama.cpp weights
    "*.h5", "*.msgpack", "tf_model.h5", "flax_model.msgpack",
]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Download Qwen2.5-7B-Instruct weights to a local folder."
    )
    p.add_argument("--repo-id", type=str, default=DEFAULT_MODEL_NAME,
                   help=f"HuggingFace repo id. Default: {DEFAULT_MODEL_NAME}.")
    p.add_argument("--local-dir", type=str, default=DEFAULT_LOCAL_DIR,
                   help=f"Destination folder. Default: {DEFAULT_LOCAL_DIR}.")
    p.add_argument("--token", type=str, default=None,
                   help="HuggingFace token (only needed for rate-limited pulls "
                        "or private mirrors). Reads HF_TOKEN env var by default.")
    args = p.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is required (pip install huggingface_hub)."
        ) from e

    os.makedirs(args.local_dir, exist_ok=True)
    print(f"[Qwen] downloading {args.repo_id!r} -> {args.local_dir}")
    print(f"[Qwen] ignore patterns: {DEFAULT_IGNORE_PATTERNS}")

    snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        token=args.token,
        ignore_patterns=DEFAULT_IGNORE_PATTERNS,
    )
    print(f"[Qwen] done. Local folder: {args.local_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
