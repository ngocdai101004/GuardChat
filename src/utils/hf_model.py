"""Shared HuggingFace loading helpers for the zero-shot guard baselines.

:mod:`src.ShieldGemma` and :mod:`src.LlamaGuard` both need the same four
things before they can score anything: decide whether ``--weights`` names
a Hub repo or a folder, populate that folder on first use, pick a device,
and turn a dtype string into either a ``torch.dtype`` or a
``BitsAndBytesConfig``. They live here so the two baselines cannot drift
apart.

Weights are deliberately fetched into the module's own ``weights/``
folder rather than the shared ``~/.cache/huggingface`` tree: one folder
per experiment is far easier to inspect, move between machines, or
delete.
"""

from __future__ import annotations

import os
from typing import Optional

import torch


DTYPE_CHOICES = ("auto", "bfloat16", "float16", "float32", "int8", "nf4")


# ----------------------------- Paths -------------------------------- #

def is_hub_id(path: str) -> bool:
    """True when ``path`` reads as ``owner/name`` rather than a folder.

    A Hub id has exactly one ``/``, is not absolute, and its first
    segment is not an existing directory - which is what separates
    ``google/shieldgemma-2b`` from a relative ``weights/shieldgemma-2b``.
    """
    if os.path.isabs(path) or path.startswith(".") or path.count("/") != 1:
        return False
    return not os.path.isdir(os.path.dirname(path))


def snapshot_is_present(local_dir: str) -> bool:
    """True when ``local_dir`` already holds a usable snapshot."""
    return os.path.isfile(os.path.join(local_dir, "config.json"))


# Duplicated or framework-specific artefacts we never need: the
# safetensors weights plus tokenizer/config files are enough.
DEFAULT_IGNORE_PATTERNS = [
    "original/*",       # Meta ships a second copy of Llama weights here
    "*.bin",            # legacy pytorch_model.bin
    "*.gguf",           # llama.cpp weights
    "*.h5", "*.msgpack", "tf_model.h5", "flax_model.msgpack",
]


def ensure_local_snapshot(
    local_dir: str,
    repo_id: str,
    token: Optional[str] = None,
    ignore_patterns=None,
    log_prefix: str = "weights",
) -> str:
    """Download ``repo_id`` into ``local_dir`` unless it is already there."""
    if snapshot_is_present(local_dir):
        return local_dir

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:  # pragma: no cover - hard runtime dep
        raise RuntimeError(
            "huggingface_hub is required to fetch model weights "
            "(pip install huggingface_hub)."
        ) from e

    # realpath, not local_dir: `weights/<model>` is often a symlink onto a
    # bigger volume, and makedirs(exist_ok=True) still raises
    # FileExistsError on a symlink whose target does not exist yet.
    os.makedirs(os.path.realpath(local_dir), exist_ok=True)
    print(f"[{log_prefix}] snapshot not found in {local_dir}")
    print(f"[{log_prefix}] downloading {repo_id!r} (one time)...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        token=token,
        ignore_patterns=list(
            DEFAULT_IGNORE_PATTERNS if ignore_patterns is None else ignore_patterns
        ),
    )
    print(f"[{log_prefix}] weights ready: {local_dir}")
    return local_dir


def resolve_weights_path(
    path: str,
    repo_id: str,
    token: Optional[str] = None,
    auto_download: bool = True,
    ignore_patterns=None,
    log_prefix: str = "weights",
) -> str:
    """Return a loadable path, fetching the snapshot when needed.

    Hub ids pass straight through. A local folder that is missing or only
    partially populated (e.g. a killed download) is filled in when
    ``auto_download`` is set, and otherwise raises with the command that
    would fix it.
    """
    if is_hub_id(path):
        return path
    if snapshot_is_present(path):
        return path
    if not auto_download:
        raise FileNotFoundError(
            f"Weights not found at {path!r}. Run the module's "
            f"`download_weights` CLI to populate it, or pass a HuggingFace "
            f"id via --weights."
        )
    return ensure_local_snapshot(
        path, repo_id=repo_id, token=token,
        ignore_patterns=ignore_patterns, log_prefix=log_prefix,
    )


# ---------------------------- Device -------------------------------- #

def resolve_device(device: Optional[str] = None) -> torch.device:
    """Pick a torch device. ``None`` / ``'auto'`` -> cuda > mps > cpu."""
    if device and device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_dtype_for(device: torch.device) -> str:
    """fp32 on CPU (bf16 matmul is slow there), bf16 on an accelerator.

    float16 is never a default: Gemma-2 overflows in fp16, and Llama in
    fp16 is only marginally cheaper than bf16 on supported hardware.
    """
    return "float32" if device.type == "cpu" else "bfloat16"


# ----------------------------- Dtype -------------------------------- #

def resolve_torch_dtype(name: str) -> torch.dtype:
    n = name.lower()
    if n in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if n in {"float16", "fp16"}:
        return torch.float16
    if n in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(
        f"Unsupported torch dtype {name!r}. Use one of {DTYPE_CHOICES}."
    )


def bnb_config(name: str):
    """Build a ``BitsAndBytesConfig`` for int8 / nf4, else ``None``."""
    n = name.lower()
    if n not in {"int8", "8bit", "nf4", "4bit"}:
        return None
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as e:  # pragma: no cover - hard runtime dep
        raise RuntimeError(
            "Quantised loading needs transformers with the "
            "BitsAndBytesConfig API plus bitsandbytes >= 0.43 (CUDA only)."
        ) from e
    if n in {"int8", "8bit"}:
        return BitsAndBytesConfig(load_in_8bit=True)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def from_pretrained_with_dtype(auto_class, path: str, dtype_name: str, **kwargs):
    """``auto_class.from_pretrained`` that works either side of the
    ``torch_dtype`` -> ``dtype`` rename in transformers 4.56."""
    kwargs = dict(kwargs)
    kwargs["dtype"] = resolve_torch_dtype(dtype_name)
    try:
        return auto_class.from_pretrained(path, **kwargs)
    except TypeError:
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        return auto_class.from_pretrained(path, **kwargs)


__all__ = [
    "DTYPE_CHOICES",
    "DEFAULT_IGNORE_PATTERNS",
    "is_hub_id",
    "snapshot_is_present",
    "ensure_local_snapshot",
    "resolve_weights_path",
    "resolve_device",
    "default_dtype_for",
    "resolve_torch_dtype",
    "bnb_config",
    "from_pretrained_with_dtype",
]
