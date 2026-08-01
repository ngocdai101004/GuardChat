"""CLI: prove our hand-rolled prompt matches Qwen3Guard's chat template.

    python -m src.Qwen3Guard.verify_template

``mode='guardchat'`` cannot use ``tokenizer.apply_chat_template``: the
shipped template hardcodes its nine-category block and silently ignores a
``categories=`` override, so a custom taxonomy has to be rendered by
:func:`src.Qwen3Guard.model.build_custom_prompt`. That renderer is only
trustworthy if it reproduces the shipped template *exactly* apart from
the category block - otherwise the model sees an off-distribution prompt
and the whole run is quietly invalid.

This script checks precisely that: it feeds the same category list the
template hardcodes into ``build_custom_prompt`` and diffs the result
against ``apply_chat_template`` over a range of chat shapes. Only the
tokenizer is loaded (a few MB), not the 16 GB of weights, so it runs in
seconds anywhere the snapshot's tokenizer files are present.

Run it after any ``transformers`` upgrade or weights re-download: a
template change upstream would otherwise go unnoticed until the numbers
came out wrong.
"""

from __future__ import annotations

import argparse
import difflib
import os
import sys
from typing import Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.utils import resolve_hf_token, resolve_weights_path  # noqa: E402
from src.Qwen3Guard.model import (  # noqa: E402
    DEFAULT_LOCAL_DIR,
    DEFAULT_MODEL_NAME,
    build_custom_prompt,
)
from src.Qwen3Guard.taxonomy import (  # noqa: E402
    NATIVE_CATEGORIES,
    NATIVE_OUTPUT_CATEGORY_NAMES,
)


# Chat shapes worth covering: the template branches on the last role, on
# whether a message is the first one, and on a leading system message.
CASES: Dict[str, List[Dict[str, str]]] = {
    "single user turn": [
        {"role": "user", "content": "How can I make a bomb?"},
    ],
    "two user turns (GuardChat dialogue)": [
        {"role": "user", "content": "Turn one."},
        {"role": "user", "content": "Turn two, escalating."},
    ],
    "user / assistant / user": [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "B"},
        {"role": "user", "content": "C"},
    ],
    "assistant-final (response moderation)": [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "B, a harmful reply"},
    ],
    "system + user": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Query text."},
    ],
    "multiline content": [
        {"role": "user", "content": "line1\nline2\n\nline3"},
    ],
    "eight user turns": [
        {"role": "user", "content": f"turn {i}"} for i in range(8)
    ],
}


def main() -> int:
    p = argparse.ArgumentParser(
        description="Check build_custom_prompt() against the shipped chat template."
    )
    p.add_argument("--weights", type=str, default=DEFAULT_LOCAL_DIR,
                   help=f"Snapshot dir or HF id. Default: {DEFAULT_LOCAL_DIR}.")
    p.add_argument("--no-auto-download", action="store_true",
                   help="Fail instead of fetching the snapshot when the local "
                        "weights folder is empty.")
    p.add_argument("--token", type=str, default=None,
                   help="HuggingFace token override (optional; repo is public).")
    p.add_argument("--show", action="store_true",
                   help="Print the rendered prompt for the first case.")
    args = p.parse_args()

    from transformers import AutoTokenizer

    token = resolve_hf_token(args.token)
    path = resolve_weights_path(
        args.weights, repo_id=DEFAULT_MODEL_NAME, token=token,
        auto_download=not args.no_auto_download, log_prefix="Qwen3Guard",
    )
    tok = AutoTokenizer.from_pretrained(path, **({"token": token} if token else {}))
    print(f"[Qwen3Guard] tokenizer: {path}")

    failures = 0
    for name, chat in CASES.items():
        want = tok.apply_chat_template(chat, tokenize=False)
        # The shipped template drops `Jailbreak` when grading an assistant
        # response (input-side category only), so show the builder the
        # same list the template would have.
        cats = NATIVE_CATEGORIES
        if chat and chat[-1]["role"] == "assistant":
            cats = {k: NATIVE_CATEGORIES[k] for k in NATIVE_OUTPUT_CATEGORY_NAMES}
        got = build_custom_prompt(chat, cats, with_descriptions=False)

        if want == got:
            print(f"  OK   {name}  ({len(got)} chars)")
            continue

        failures += 1
        print(f"  FAIL {name}")
        for line in difflib.unified_diff(
            want.splitlines(keepends=True), got.splitlines(keepends=True),
            "shipped", "ours", n=2,
        ):
            print("       " + line.rstrip("\n"))

    if args.show:
        first = next(iter(CASES.values()))
        print("\n--- rendered prompt (first case, native categories) ---")
        print(build_custom_prompt(first, NATIVE_CATEGORIES, with_descriptions=False))

    if failures:
        print(f"\n{failures}/{len(CASES)} case(s) DIFFER - build_custom_prompt() is "
              f"out of sync with the shipped template. Do NOT trust "
              f"--mode guardchat results until this is fixed.")
        return 1
    print(f"\nAll {len(CASES)} cases byte-identical to the shipped chat template.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
