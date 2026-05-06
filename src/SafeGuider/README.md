# SafeGuider — GuardChat Evaluation Code

Evaluation code that adapts the vendored **SafeGuider** (`vendors/SafeGuider/`)
to the two GuardChat benchmark tasks defined in the paper:

| Task | Description | Metrics |
|------|-------------|---------|
| **1. Recognition** | Multi-label unsafe text recognition over six NSFW categories (sexual, illegal, shocking, violence, self-harm, harassment), supporting both single-turn prompts and multi-turn conversations. | Macro-F1, Recall, ASR |
| **2. Rewrite** | NSFW concept removal via prompt rewriting (safety-aware beam search). | CLIP cosine similarity (here) + Safe Generation Rate via T2I models (external) |

The vendored library is **not modified** — this folder only adds GuardChat-specific
data loading, a 6-way multi-label classifier head, training/eval scripts, and
metric utilities.

---

## 1. Layout

```
src/SafeGuider/
├── __init__.py                # bootstraps vendors/SafeGuider/ on sys.path
├── data.py                    # GuardChatSample loader; single-turn + conversation
├── classifier.py              # MultiLabelClassifier (6 sigmoid heads)
├── recognition.py             # RecognitionPipeline + RecognitionTrainer (Task 1)
├── rewrite.py                 # RewritePipeline (Task 2)
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
from src.SafeGuider.data import load_guardchat

pipe = RecognitionPipeline.from_pretrained(
    weights="src/SafeGuider/weights/recognition_multilabel.pt",
)
samples = load_guardchat("data/guardchat/test.jsonl")
preds = pipe.predict_samples(samples, kind="conversation")
```

---

## 4. Task 2 — NSFW Concept Removal (Rewriting)

The rewriter uses the **original** SafeGuider binary safety classifier
(unchanged from the published checkpoint, e.g. `SD1.4_safeguider.pt`) and
the vendored safety-aware beam search.

```bash
python -m src.SafeGuider.eval_rewrite \
    --test data/guardchat/test.jsonl \
    --weights vendors/SafeGuider/weights/SD1.4_safeguider.pt \
    --output results/safeguider_task2.json
```

The output records include `rewritten_prompt`, `was_modified`,
`removed_tokens`, the SafeGuider in-loop safety score, and the
post-hoc `clip_similarity` between the original and rewritten prompts.

**Safe Generation Rate (SGR)** against FLUX.1 / Gemini / DALL-E 3 is *not*
computed here — feed `rewritten_prompt` to the relevant T2I system in a
separate pipeline, judge the resulting images, and compute SGR
externally. This keeps `eval_rewrite.py` free of proprietary API
dependencies.

### Python API

```python
from src.SafeGuider.rewrite import RewritePipeline
from src.SafeGuider.data import load_guardchat

pipe = RewritePipeline(weights="vendors/SafeGuider/weights/SD1.4_safeguider.pt")
samples = load_guardchat("data/guardchat/test.jsonl")
results = pipe.rewrite_samples(samples)
for r in results[:3]:
    print(r.to_dict())
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

### Task 2 — `results/safeguider_task2.json`

```jsonc
{
  "summary": {
    "num_samples": 1000,
    "fraction_modified": 0.97,
    "mean_clip_similarity": 0.351,
    "mean_safeguider_similarity": 0.41,
    "mean_modified_safety": 0.92
  },
  "rewrites": [
    {
      "sample_id": "0001",
      "original_prompt": "...",
      "rewritten_prompt": "...",
      "was_modified": true,
      "removed_tokens": ["violent", "gory"],
      "original_safety": 0.014,
      "modified_safety": 0.962,
      "safeguider_similarity": 0.688,
      "clip_similarity": 0.412,
      "elapsed_sec": 1.83,
      "label_names": ["violence"],
      "source": "I2P"
    },
    ...
  ]
}
```

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
* The rewriting pipeline always rewrites the enhanced prompt, regardless
  of the in-loop binary verdict, because GuardChat test entries are
  oracle-verified adversarial samples by construction.
