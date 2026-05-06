# vendors/SafeGuider — trimmed copy

This folder is a **minimal extract** of the upstream SafeGuider input
guard. Only the modules that `src/SafeGuider/` actually imports are
kept here:

| File | Provides |
|------|----------|
| `classifier.py` | `ThreeLayerClassifier` (binary safety MLP, used by Task 2) |
| `encoder.py` | `CLIPEncoder` (CLIP text-encoder + EOS embedding helper) |
| `beam_search.py` | `SafetyAwareBeamSearch`, `BeamSearchResult`, default thresholds |
| `weights/README.md` | How to obtain the binary safety classifier checkpoint |

The upstream SafeGuider repository ships a much larger CLI / training
pipeline (`recognizer.py`, `pipeline.py`, `prepare_embeddings.py`,
`train.py`, `examples/`). Those entry points are **not used** by the
GuardChat benchmark code in this repository, so they have been removed
from the vendored copy to keep the repo lean.

If you need the full SafeGuider implementation, refer to the upstream
release. The three modules above are imported by `src/SafeGuider/` via
a `sys.path` shim — see `src/SafeGuider/__init__.py` for the bootstrap
logic.

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
