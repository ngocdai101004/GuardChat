# SafeGuider — GuardChat Evaluation Code

Evaluation code that adapts the vendored **SafeGuider** (`vendors/SafeGuider/`)
to the two GuardChat benchmark tasks defined in the paper:

| Task | Description | Metrics |
|------|-------------|---------|
| **1. Recognition** | Multi-label unsafe text recognition over six NSFW categories (sexual, illegal, shocking, violence, self-harm, harassment), supporting both single-turn prompts and multi-turn conversations. | Macro-F1, Recall, ASR |
| **2. Rewrite** | NSFW concept removal via prompt rewriting (safety-aware beam search), over both the enhanced prompt and the multi-turn conversation. | Safe Generation Rate + SBERT semantic similarity, both computed downstream |

This folder adds GuardChat-specific data loading, a 6-way multi-label
classifier head, training/eval scripts, and metric utilities. The vendored
beam search carries two deliberate, behaviour-preserving deviations from
upstream (batched encoding and earlier loop termination) — both are
documented at the top of `vendors/SafeGuider/beam_search.py`.

---

## 1. Layout

```
src/SafeGuider/
├── __init__.py                # bootstraps vendors/SafeGuider/ on sys.path
├── data.py                    # GuardChatSample loader; single-turn + conversation
├── classifier.py              # MultiLabelClassifier (6 sigmoid heads)
├── recognition.py             # RecognitionPipeline + RecognitionTrainer (Task 1)
├── rewrite.py                 # RewritePipeline (Task 2, shared record schema)
├── metrics.py                 # Macro-F1, Recall, ASR, CLIP cosine similarity
├── train_recognition.py       # CLI: train Task 1 head
├── eval_recognition.py        # CLI: evaluate Task 1
├── eval_rewrite.py            # CLI: run Task 2 rewriting + CLIP-sim
└── configs/
    ├── recognition.yaml
    └── rewrite.yaml
```

---

## 2. Data format

GuardChat samples are loaded via `data.load_guardchat(path)` and accept
`.json` (list / `{"data": [...]}`) or `.jsonl`. Each record is normalised to:

```jsonc
{
  "id": "0001",
  "enhanced_prompt": "...",
  "conversation": [
    {"turn_id": 1, "role": "user", "content": "..."},
    ...
  ],
  "label_vector": [1, 0, 0, 1, 0, 0],     // canonical 6-dim
  "source": "I2P"                           // optional
}
```

Labels can also be supplied as a list of names (`["sexual", "violence"]`)
or as a `{"sexual": 1, ...}` dict — the loader normalises everything to
the canonical order:

```
["sexual", "illegal", "shocking", "violence", "self-harm", "harassment"]
```

Benign prompts (e.g. DiffusionDB safe) are loaded with
`data.load_safe_prompts(path)` and assigned an all-zero label vector.

---

## 3. Task 1 — Recognition

### Train (paper recipe: 9k harmful + 5k safe, AdamW, lr=2e-5, 10 epochs)

```bash
python -m src.SafeGuider.train_recognition \
    --train data/guardchat/train.jsonl \
    --safe  data/diffusiondb_safe.json \
    --output src/SafeGuider/weights/recognition_multilabel.pt \
    --text-kind conversation \
    --epochs 10 --batch-size 32 --lr 2e-5 --weight-decay 1e-2
```

### Evaluate (single-turn ASR + multi-turn ASR + Macro-F1, as in Table 1)

```bash
python -m src.SafeGuider.eval_recognition \
    --test data/guardchat/test.jsonl \
    --weights src/SafeGuider/weights/recognition_multilabel.pt \
    --text-kind both \
    --output results/safeguider_task1.json
```

`--text-kind both` runs the model twice — once on `enhanced_prompt`
(produces single-turn ASR) and once on the flattened conversation
(produces multi-turn ASR / Macro-F1). The output JSON keeps each
representation under its own key so downstream tables can pick them up
without re-running inference.

### Python API

```python
from src.SafeGuider.recognition import RecognitionPipeline
from src.utils import load_guardchat

pipe = RecognitionPipeline.from_pretrained(
    weights="src/SafeGuider/weights/recognition_multilabel.pt",
)
samples = load_guardchat("data/guardchat/test.jsonl")
preds = pipe.predict_samples(samples, kind="conversation")
```

---

## 4. Task 2 — NSFW Concept Removal (Rewriting)

The rewriter is the **upstream** system: the published binary recognizer
checkpoint (`SD1.4_safeguider.pt`, unchanged) plus the safety-aware beam
search lifted out of `stable-diffusion-1.4/scripts/safeguider_gene.py`.

```bash
python -m src.SafeGuider.eval_rewrite \
    --test build_dataset/dataset/final_df_test.json \
    --text-kind all \
    --output-dir experiment_results/task2/safeguider
```

or `bash scripts/benchmark_task2_safeguider.sh`.

Two files are written, one per input representation, in the shared
Task-2 record schema (`src/utils/task2_eval.py`) that Gemini and Llama
also use:

```
safeguider_task2_prompt.json        enhanced prompt -> P_safe
safeguider_task2_conversation.json  dialogue        -> P_safe
```

### How it differs from the LLM baselines

SafeGuider does not *generate*. It **deletes words** from the input until
the recognizer's `P[safe]` crosses 0.80 while the CLIP-EOS cosine to the
original stays above 0.10. Three consequences shape how the numbers
should be read:

1. **Subsequence only.** The output is always a subsequence of the
   input. Where Gemini turns *"a mutilated body"* into *"a weathered
   statue"*, SafeGuider can only drop *"mutilated"*. Its length ratio is
   therefore a hard ceiling on retained content, not a soft signal.
2. **The gate is part of the system.** Upstream skips the rewrite when
   the recognizer calls a prompt safe, and `--gate recognizer` (the
   default) reproduces that. Prompts the recognizer misses reach the T2I
   model exactly as the attacker wrote them — a genuine SafeGuider
   failure mode that forcing a rewrite on every row would hide. Use
   `--gate always` for the ablation that measures the search alone. The
   CLI prints the pass-through rate for both cases.
3. **CLIP sees 77 tokens.** GuardChat's enhanced prompts average 99
   *words*, so a large part of a typical prompt sits outside the encoder
   window. A word out there cannot move the safety score, so the search
   can never find it. Every record carries `num_tokens` and `truncated`,
   and the CLI reports the fraction affected — quantify the reach rather
   than assuming it.

### Conversations

The dialogue is rewritten **turn by turn**, then reassembled. Running
the search over the flattened conversation would fail twice over: CLIP
would see only the first turn and part of the second, and word deletion
could eat the `user:` prefixes that define turn boundaries. Per-turn
search preserves the turn count by construction, so `turn_parse` is
always `"ok"` here — unlike the LLM baselines, where it is the verdict
of parsing a generated `[Tn]` block and can legitimately fail.

### Runtime

A 99-word prompt expands into thousands of deletion candidates. The
vendored search batches candidates through the encoder and memoises by
candidate string (see `--batch-size`), which is what makes a
1,000-sample run practical; upstream's one-string-per-forward-pass loop
is not. Each record reports `num_encoded`, the number of distinct
strings actually pushed through CLIP.

**Safe Generation Rate (SGR)** and **SBERT semantic similarity** are
*not* computed here. Both consume the `rewritten_text` field offline —
SGR by feeding it to FLUX.1 / Gemini Flash Image / DALL-E 3 and judging
the images with the safety gate. This keeps `eval_rewrite.py` free of
proprietary API dependencies.

### Python API

```python
from src.SafeGuider.rewrite import RewritePipeline
from src.utils import load_guardchat

pipe = RewritePipeline.from_weights(
    weights="vendors/SafeGuider/weights/SD1.4_safeguider.pt",
    gate="recognizer",
)
samples = load_guardchat("build_dataset/dataset/final_df_test.json")
records = pipe.rewrite_samples(samples, kind="prompt")
for r in records[:3]:
    print(r.status, r.extra["outcome"], r.rewritten_text)
```

---

## 5. Output schemas

### Task 1 — `results/safeguider_task1.json`

```jsonc
{
  "single": {
    "metrics": {
      "macro_f1": 0.487,
      "recall_binary": 0.945,
      "asr": 0.055,
      "f1_sexual": 0.62, "f1_illegal": 0.41, ...
    },
    "predictions": [
      {
        "sample_id": "0001",
        "text": "...",
        "probs": {"sexual": 0.93, ...},
        "multi_label": {"sexual": 1, ...},
        "predicted_categories": ["sexual"],
        "binary_pred": 1,
        "label_vector_true": {"sexual": 1, ...}
      },
      ...
    ]
  },
  "conversation": { ... same shape ... }
}
```

### Task 2 — `experiment_results/task2/safeguider/safeguider_task2_prompt.json`

The shared Task-2 schema. `extra` is the SafeGuider-specific block; every
other key is identical across the three baselines.

```jsonc
{
  "prompt": {
    "summary": {
      "num_samples": 1000,
      "num_usable": 1000,
      "fraction_usable": 1.0,
      "status_counts": {"ok": 1000},
      "error_kind_counts": {},
      "fraction_modified": 0.91,
      "mean_length_ratio": 0.63,
      "mean_elapsed_sec": 2.4
    },
    "rewrites": [
      {
        "sample_id": "0001",
        "text_kind": "prompt",
        "status": "ok",
        "error_kind": null,
        "original_text": "...",
        "rewritten_text": "...",
        "was_modified": true,
        "model": "SafeGuider-beam-search",
        "gold_category": "violence",
        "extra": {
          "gated_safe": false,
          "original_safety": 0.014,
          "modified_safety": 0.962,
          "beam_similarity": 0.688,
          "removed_tokens": ["mutilated", "gory"],
          "num_removed": 2,
          "outcome": "qualified",
          "depth_reached": 2,
          "num_tokens": 131,
          "truncated": true,
          "num_encoded": 812
        }
      }
    ]
  },
  "meta": {"gate": "recognizer", "beam_width": 6, "...": "..."}
}
```

`safeguider_task2_conversation.json` has the same shape plus
`original_turns` / `rewritten_turns` / `turn_parse`, and its `extra`
aggregates over turns (`num_turns_gated_safe`, `num_turns_modified`,
`num_turns_truncated`, `mean_modified_safety`, …) while keeping the
per-turn detail under `extra.turns`.

`outcome` records how the search ended:

| Value | Meaning |
|-------|---------|
| `qualified` | a candidate cleared both thresholds |
| `fallback` | none did; took the safest above `similarity_floor` |
| `unchanged` | nothing cleared `similarity_floor`; original kept |
| `single_word` | too short to delete from |
| `gated_safe` | recognizer judged it safe; no search run |

---

## 6. Notes

* The recognition head is multi-label (6 sigmoid outputs) where the
  vendored ``ThreeLayerClassifier`` is binary (2-way softmax). The
  multi-label weights are NOT interchangeable with the binary
  ``SD1.4_safeguider.pt`` checkpoint; train new weights with
  `train_recognition.py`.
* Multi-turn input flattens the conversation as
  ``role: content\nrole: content\n...`` before encoding. Long
  conversations may be truncated by CLIP's 77-token limit; the encoder
  will log a `TRUNCATED!` warning when `--verbose` is enabled.
* Task 2 keeps the upstream recognizer gate by default, so prompts the
  binary classifier judges safe are passed through unmodified. Pass
  `--gate always` to rewrite every row regardless — useful as an
  ablation, but no longer the published system.
* Task 2 no longer computes CLIP similarity inline. Semantic similarity
  moved to SBERT and is computed downstream from `rewritten_text`,
  together with Safe Generation Rate.
