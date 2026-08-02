"""HuggingFace access-token resolution for the gated guard-model repos.

Both zero-shot baselines pull gated weights - ``google/shieldgemma-2b``
(Gemma licence) and ``meta-llama/Llama-Guard-3-8B`` (Meta licence) - so
every download / load needs a token. Resolution order (first hit wins):

1. an explicit ``--token`` CLI argument,
2. the ``HF_TOKEN`` / ``HUGGINGFACE_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN``
   environment variables,
3. the same keys in a ``.env`` file at the repo root (git-ignored),
4. a prior ``huggingface-cli login`` (handled inside ``huggingface_hub``;
   this module simply returns ``None`` and lets the library take over).

Put the token in ``.env`` to change it later without touching code::

    HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Sequence


ENV_KEYS = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN")

_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)

DEFAULT_ENV_FILE = os.path.join(_REPO_ROOT, ".env")


def _parse_env_file(path: str) -> Dict[str, str]:
    """Minimal ``KEY=value`` reader - no python-dotenv dependency."""
    values: Dict[str, str] = {}
    if not os.path.isfile(path):
        return values
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            val = val.strip().strip('"').strip("'")
            if val:
                values[key.strip()] = val
    return values


def resolve_env_value(
    keys: Sequence[str],
    explicit: Optional[str] = None,
    env_file: Optional[str] = None,
) -> Optional[str]:
    """First non-empty value among ``explicit``, the environment, ``.env``.

    Generic version of :func:`resolve_hf_token`, used for the other
    credentials the benchmark needs - ``GEMINI_API_KEY`` for the Task-2
    Gemini rewriter, and later the image-generation keys - so they all
    live in the same git-ignored ``.env`` at the repo root.
    """
    if explicit:
        return explicit
    for key in keys:
        val = os.environ.get(key)
        if val:
            return val
    parsed = _parse_env_file(env_file or DEFAULT_ENV_FILE)
    for key in keys:
        val = parsed.get(key)
        if val:
            return val
    return None


def resolve_hf_token(
    explicit: Optional[str] = None,
    env_file: Optional[str] = None,
) -> Optional[str]:
    """Return the first token found, or ``None`` to defer to the HF cache."""
    return resolve_env_value(ENV_KEYS, explicit=explicit, env_file=env_file)


__all__ = [
    "ENV_KEYS",
    "DEFAULT_ENV_FILE",
    "resolve_env_value",
    "resolve_hf_token",
]
