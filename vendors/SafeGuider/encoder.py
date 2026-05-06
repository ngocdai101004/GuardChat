"""
Text encoder wrapper for SafeGuider input guard.

Important — ensure that the embedding generated here MATCHES the embedding that the recognizer
was trained on. In the original repo, the recognizer was trained on embeddings produced
by `FrozenCLIPEmbedder` from LDM (see `ldm/modules/encoders/modules.py`), which essentially only does:

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    transformer = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    batch_encoding = tokenizer(text, truncation=True, max_length=77,
                               padding="max_length", return_tensors="pt")
    z = transformer(input_ids=batch_encoding["input_ids"]).last_hidden_state

The `CLIPEncoder` class here replicates this logic exactly (DO NOT pass
`attention_mask`, since CLIP's pad_token_id is the same as eos_token_id = 49407,
and LDM also does not pass attention_mask during training).

Supports OpenCLIP (SD-V2.1, dim=1024) and T5 (Flux, dim=4096) via
the `--encoder-model` flag. Default is `openai/clip-vit-large-patch14`
(SD-V1.4).
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


# CLIP eos = pad = 49407. When tokenizer uses pad="max_length", PAD is also 49407,
# so the actual EOS position is the FIRST occurrence of 49407 in input_ids.
CLIP_EOS_TOKEN_ID = 49407

# Default cache dir = <input_guard_only>/weights/. Each HuggingFace model is downloaded into a
# subfolder `<weights>/<model_basename>/` (for example `weights/clip-vit-large-patch14/`)
# for easier management — does not interfere with HuggingFace's default cache at ~/.cache/huggingface.
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")


# Module logger — kept separate from the root logger so it can be enabled/disabled
# without affecting the global logging config. Only prints if `CLIPEncoder(verbose=True)`.
log = logging.getLogger("safeguider.encoder")
if not log.handlers:
    _h = logging.StreamHandler(stream=sys.stderr)
    _h.setFormatter(logging.Formatter("[encoder] %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False


def _is_loadable_dir(path: str) -> bool:
    """Checks if a directory contains sufficient weights + config for `from_pretrained(path)` to load offline."""
    if not os.path.isdir(path):
        return False
    if not os.path.isfile(os.path.join(path, "config.json")):
        return False
    # At least one weight file (safe for pytorch_model.bin and .safetensors)
    for fn in os.listdir(path):
        low = fn.lower()
        if low.endswith((".safetensors", ".bin", ".pt", ".pth", ".msgpack")):
            return True
    return False


def resolve_encoder_path(
    model_name: str,
    cache_dir: str = DEFAULT_CACHE_DIR,
    force_download: bool = False,
) -> str:
    """Local-first resolver for a HuggingFace model id.

    Logic:
        1. If `model_name` is already a valid local folder → return as is.
        2. If `<cache_dir>/<basename>` is valid → return (load offline).
        3. Otherwise: `snapshot_download(model_name, local_dir=<cache_dir>/<basename>)`
           and return.

    Args:
        model_name:     "openai/clip-vit-large-patch14" or a local path.
        cache_dir:      parent directory, default = `<input_guard_only>/weights/`.
        force_download: if True, always re-download (even if local exists).

    Returns:
        Path to the local folder that `transformers.from_pretrained(...)` can load.
    """
    # Case 1: user already provides a valid local folder
    if _is_loadable_dir(model_name):
        return model_name

    basename = model_name.rstrip("/").split("/")[-1]
    target = os.path.join(cache_dir, basename)

    # Case 2: already cached locally
    if not force_download and _is_loadable_dir(target):
        return target

    # Case 3: download
    os.makedirs(cache_dir, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "Requires `huggingface_hub` to auto-download. "
            "Install with `pip install huggingface_hub`, or manually copy the model "
            f"from HuggingFace into {target!r}."
        ) from e

    print(f"[encoder] downloading {model_name!r} -> {target}")
    snapshot_download(
        repo_id=model_name,
        local_dir=target,
        # Do not download flax/tf files to save space
        ignore_patterns=["*.msgpack", "*.h5", "tf_model.h5", "flax_model.msgpack"],
    )
    if not _is_loadable_dir(target):
        raise RuntimeError(
            f"After download, {target!r} is still not valid. "
            "Check network or file write permissions."
        )
    return target


@dataclass
class EncodeResult:
    """Result of encoding a batch of prompts.

    Attributes:
        last_hidden_state: tensor (B, L, D) — full padded sequence embedding.
        eos_positions:     tensor (B,) int64 — EOS position in each prompt.
        eos_embedding:     tensor (B, D) — embedding at EOS, used for classifier.
        input_ids:         tensor (B, L) — token ids (padded).
    """

    last_hidden_state: Tensor
    eos_positions: Tensor
    eos_embedding: Tensor
    input_ids: Tensor


class CLIPEncoder:
    """Wrapper for text encoder + tokenizer (local-first).

    Args:
        model_name:    HuggingFace id `openai/clip-vit-large-patch14` or a local path.
                       Default = SD-V1.4 encoder.
        cache_dir:     directory for downloaded models. Default
                       `<input_guard_only>/weights/`. Each model lives in its own subfolder
                       (e.g. `weights/clip-vit-large-patch14/`).
        device:        `cuda` | `cpu` | None (auto-detect).
        max_length:    77 for CLIP. Change if using different encoder.
        eos_token_id:  49407 for CLIP. Change for different tokenizers (e.g. T5).
        dtype:         torch dtype for model weights (default float32).
        force_download: if True, force re-download even if local cache exists.

    Loading logic:
        1) If model_name is a valid local folder → load directly.
        2) If `<cache_dir>/<basename>` exists → load offline from there.
        3) Otherwise → snapshot_download from HF to `<cache_dir>/<basename>` then load.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        cache_dir: str = DEFAULT_CACHE_DIR,
        device: Optional[str] = None,
        max_length: int = 77,
        eos_token_id: int = CLIP_EOS_TOKEN_ID,
        dtype: torch.dtype = torch.float32,
        force_download: bool = False,
        verbose: bool = False,
    ) -> None:
        # Lazy imports so that CLIPEncoder doesn't require transformers if only using
        # other utilities (e.g. offline .pt loading).
        from transformers import AutoTokenizer, AutoModel, CLIPTextModel, CLIPTokenizer
        from transformers import logging as hf_logging

        # Snapshot `openai/clip-vit-large-patch14` contains FULL CLIP (text + vision +
        # 2 projection heads + logit_scale). We only load `CLIPTextModel` (text-only),
        # so transformers will warn about UNEXPECTED keys for `vision_model.*`,
        # `visual_projection`, `text_projection`, `logit_scale`. This is expected
        # (text encoder still loads correctly), not an error → silence warnings.
        hf_logging.set_verbosity_error()

        self.device = torch.device(device) if device else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.max_length = max_length
        self.eos_token_id = eos_token_id
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.verbose = bool(verbose)

        # Local-first resolve: returns a valid local folder, NEVER uses HuggingFace default cache.
        local_path = resolve_encoder_path(model_name, cache_dir=cache_dir,
                                          force_download=force_download)
        self.local_path = local_path

        # Prefer CLIPTokenizer / CLIPTextModel, fallback to Auto* if unavailable.
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(local_path, local_files_only=True)
            self.text_encoder = CLIPTextModel.from_pretrained(
                local_path, local_files_only=True,
            ).to(device=self.device, dtype=dtype)
        except (OSError, ValueError):
            self.tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
            self.text_encoder = AutoModel.from_pretrained(
                local_path, local_files_only=True,
            ).to(device=self.device, dtype=dtype)

        self.text_encoder.eval()
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        # Infer hidden_size — for classifier initialization.
        self.hidden_size: int = int(self.text_encoder.config.hidden_size)

        if self.verbose:
            log.info(
                f"loaded {model_name!r} from {local_path} | "
                f"device={self.device} dtype={dtype} | "
                f"hidden_size={self.hidden_size} max_length={self.max_length} "
                f"eos_token_id={self.eos_token_id}"
            )

    # ----------------------------- Core API ----------------------------- #

    @torch.no_grad()
    def encode(self, prompts: Sequence[str]) -> EncodeResult:
        """Encode a list of prompts → EncodeResult.

        Matches logic from `FrozenCLIPEmbedder.forward` in LDM: tokenize with
        `padding="max_length"` and DO NOT pass attention_mask.

        If `verbose=True`, logs:
          - raw_tokens:  number of tokens generated by tokenizer (includes BOS + EOS),
                         BEFORE truncation. If > max_length, will be truncated.
          - tokens:      number of "real" tokens in sequence after truncation
                         (BOS + content + EOS, does not count PAD).
          - truncated:   number of truncated tokens (= max(0, raw - max_length)).
          - shape of `last_hidden_state` (B, L, D).
          - shape of `eos_embedding` (B, D) — the vector used in the classifier.
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        # Pre-tokenize WITHOUT trunc/pad to get raw token count (only if verbose
        # — to avoid slowdowns during beam search).
        raw_token_counts: Optional[List[int]] = None
        if self.verbose:
            raw = self.tokenizer(
                list(prompts),
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )
            raw_token_counts = [len(ids) for ids in raw["input_ids"]]

        batch = self.tokenizer(
            list(prompts),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            return_attention_mask=False,
        )
        input_ids = batch["input_ids"].to(self.device)

        outputs = self.text_encoder(input_ids=input_ids)
        last_hidden_state = outputs.last_hidden_state  # (B, L, D)

        eos_positions = self._eos_positions(input_ids)
        eos_embedding = last_hidden_state[
            torch.arange(last_hidden_state.size(0), device=self.device), eos_positions
        ]

        if self.verbose:
            self._log_encode(prompts, eos_positions, last_hidden_state,
                             eos_embedding, raw_token_counts)

        return EncodeResult(
            last_hidden_state=last_hidden_state,
            eos_positions=eos_positions,
            eos_embedding=eos_embedding,
            input_ids=input_ids,
        )

    @torch.no_grad()
    def eos_embedding(self, prompts: Sequence[str]) -> Tensor:
        """Utility: returns only the EOS embedding, shape (B, D)."""
        return self.encode(prompts).eos_embedding

    # ----------------------------- Helpers ------------------------------ #

    def _eos_positions(self, input_ids: Tensor) -> Tensor:
        """Finds the EOS position for each sequence.

        Because CLIP has pad_token_id == eos_token_id, the real EOS position is
        the FIRST occurrence of eos_token_id in the padded sequence.
        If not found (rare with aggressive truncation), fall back to `max_length - 1`.
        """
        is_eos = (input_ids == self.eos_token_id).int()
        if is_eos.any(dim=1).all():
            return is_eos.argmax(dim=1)

        # Safe fallback for sequences with no EOS
        positions = []
        L = input_ids.size(1)
        for row in is_eos:
            if row.any():
                positions.append(int(row.argmax().item()))
            else:
                positions.append(L - 1)
        return torch.tensor(positions, device=input_ids.device, dtype=torch.long)

    @staticmethod
    def cosine_similarity(a: Tensor, b: Tensor) -> Tensor:
        """Cosine similarity between two batch embeddings (B, D)."""
        return F.cosine_similarity(a, b, dim=-1)

    def _log_encode(
        self,
        prompts: Sequence[str],
        eos_positions: Tensor,
        last_hidden_state: Tensor,
        eos_embedding: Tensor,
        raw_token_counts: Optional[List[int]] = None,
    ) -> None:
        """Print token-count + shape when `verbose=True`.

        - `raw_tokens`: number of tokens generated by tokenizer (BOS + content + EOS),
          BEFORE truncation.
        - `tokens`:     actual token count after truncation in padded sequence
          = `eos_position + 1` (since pad_token_id == eos_token_id for CLIP).
        - `truncated`:  `max(0, raw_tokens - max_length)`.
        - When `truncated > 0`, EOS is kept at the last position of max_length;
          content between BOS and EOS may be truncated from the tail.
        """
        for i, (p, pos) in enumerate(zip(prompts, eos_positions.tolist())):
            n_tokens = int(pos) + 1
            n_pad = self.max_length - n_tokens
            preview = p if len(p) <= 60 else p[:57] + "..."

            if raw_token_counts is not None:
                raw = raw_token_counts[i]
                truncated = max(0, raw - self.max_length)
                trunc_tag = f" TRUNCATED!" if truncated > 0 else ""
                log.info(
                    f"prompt[{i}]={preview!r} | raw_tokens={raw} -> "
                    f"tokens={n_tokens} (pad={n_pad}, truncated={truncated}){trunc_tag} | "
                    f"eos_pos={pos}"
                )
            else:
                log.info(
                    f"prompt[{i}]={preview!r} | tokens={n_tokens} (pad={n_pad}) | "
                    f"eos_pos={pos}"
                )
        log.info(
            f"shapes: last_hidden_state={tuple(last_hidden_state.shape)} "
            f"-> eos_embedding={tuple(eos_embedding.shape)} "
            f"(classifier input dim={self.hidden_size})"
        )
