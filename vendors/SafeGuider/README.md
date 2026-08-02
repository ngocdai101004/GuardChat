# vendors/SafeGuider — trimmed extract

Upstream: <https://github.com/pgqihere/safeguider> — *SafeGuider: Robust
and Practical Content Safety Control for Text-to-Image Models*, ACM CCS
2025.

A **minimal extract** of the SafeGuider input guard: only what
`src/SafeGuider/` imports.

| File | Provides | Extracted from |
|------|----------|----------------|
| `classifier.py` | `ThreeLayerClassifier` (binary safety MLP) | `stable-diffusion-1.4/tools/classifier.py` |
| `encoder.py` | `CLIPEncoder` (CLIP text encoder + EOS embedding) | `recognizer.py` / LDM's `FrozenCLIPEmbedder` |
| `beam_search.py` | `SafetyAwareBeamSearch`, `BeamSearchResult`, thresholds | `stable-diffusion-1.4/scripts/safeguider_gene.py` |
| `weights/README.md` | How to obtain the recognizer checkpoint | — |

Upstream ships the rewriter as ~200 lines inlined into a Stable
Diffusion sampling loop, entangled with LDM model loading, DDIM
sampling and PNG writing — there is no importable rewrite module. The
extraction pulls that logic out and swaps `model.get_learned_conditioning`
for a standalone CLIP text encoder, so the rewriter runs without Stable
Diffusion checked out. The embedding is the same one either way: CLIP
`last_hidden_state` at the EOS position, no attention mask, padded to 77.

## Deviations from upstream

Two, both behaviour-preserving, both spelled out at the top of
`beam_search.py`:

* **Batched scoring.** Upstream encodes one candidate per forward pass.
  Candidates within a beam step are independent and the tokenizer pads
  to a fixed 77, so batching them is numerically identical — it changes
  the cost, not the result. Without it a 1,000-sample GuardChat run
  takes hours.
* **Termination.** Upstream keeps iterating depths after a step expands
  nothing; here that breaks the loop.

The thresholds (0.80 / 0.10), beam width 6, depth `min(25, len(words)-1)`,
the sort keys, the tie-breaks, the fallback rule and the "class 1 = safe"
convention all match `safeguider_gene.py` line for line.

Upstream's training and empirical-study code (`tools/train.py`,
`tools/json2embedding.py`, `Emperical_Study/`) and the whole
`stable-diffusion-1.4/` tree are **not** vendored — the GuardChat
benchmark does not use them. Refer to the upstream release if you need
them. The three modules above are imported by `src/SafeGuider/` via a
`sys.path` shim; see `src/SafeGuider/__init__.py`.

## Weights

`vendors/SafeGuider/weights/` is **populated locally** and gitignored:

- `SD1.4_safeguider.pt` — required for Task 2 (beam-search rewriter).
  Obtain from the upstream SafeGuider release.
- `clip-vit-large-patch14/` — auto-downloaded by `encoder.py` on first
  use (also pre-fetchable via `bash scripts/download_weights.sh
  safeguider`).
- `SD2.1_safeguider.pt`, `Flux_safeguider.pt` — alternative encoder
  backbones (SD-V2.1, FLUX). Optional; not used by default.

See `weights/README.md` for details.
