# BERT — GuardChat Task 1 Baseline

PyTorch + HuggingFace fine-tuning of BERT for the GuardChat
multi-label unsafe text recognition task. CLI surface, output JSON
schema, and metric definitions are kept identical to
`src/SafeGuider/` and `src/BiLSTM/` so a single benchmark aggregator
composes Table 1 across all three baselines.

The reference notebook (`vendors/BERT/mental_bert.py`) trained a
single-label softmax head over four classes on a sentiment dataset; we
reuse the encoder fine-tuning recipe but replace the head with
`AutoModelForSequenceClassification(problem_type="multi_label_classification")`,
which switches the loss to BCE-with-logits and exposes six independent
sigmoid outputs over the canonical GuardChat categories.

---

## 1. Layout

```
src/BERT/
├── __init__.py
├── README.md
├── model.py                # BERTClassifier wrapping AutoModelForSequenceClassification
├── recognition.py          # RecognitionPipeline + RecognitionTrainer
├── train_recognition.py    # CLI: fine-tune
├── eval_recognition.py     # CLI: evaluate
└── configs/
    └── recognition.yaml
```

Shared GuardChat conventions live in `src/utils/` (CATEGORIES,
GuardChatSample, summarise_recognition, etc.) — same as SafeGuider /
BiLSTM.

---

## 2. Architecture

| Component | Spec |
|-----------|------|
| Backbone | `bert-base-uncased` (configurable via `--model-name`) |
| Head | `Linear(hidden, 6)` with `problem_type="multi_label_classification"` |
| Loss | `BCEWithLogitsLoss` (handled internally by HF when labels are float) |
| Optimiser | AdamW, `lr=2e-5`, `weight_decay=1e-2` |
| Batch / Epochs | 32 / 10 |
| Max length | 256 (configurable) |

Differences vs the reference notebook:

1. Head replaced with multi-label sigmoid (6 outputs) instead of softmax
   over 4–8 classes.
2. The notebook used `mental/mental-bert-base-uncased`; we default to
   `bert-base-uncased` to match the paper citation. Either works via
   `--model-name`.
3. Single-label one-hot encoder removed — labels come straight from the
   shared `GuardChatSample.label_vector` (6-dim 0/1).

---

## 3. Fine-tune

```bash
python -m src.BERT.train_recognition \
    --train data/guardchat/train.jsonl \
    --safe  data/diffusiondb_safe.json \
    --output src/BERT/weights/bert_multilabel \
    --text-kind conversation \
    --epochs 10 --batch-size 32 --lr 2e-5 --weight-decay 1e-2
```

`--output` is a **directory**, not a single `.pt` file: HuggingFace's
`save_pretrained` writes the model + config + tokenizer there, plus a
small `recognition_meta.json` (`num_classes`, `max_length`, categories,
backbone) used by the inference loader.

The trainer:

1. Loads GuardChat samples + DiffusionDB safe prompts via shared utils.
2. Tokenises with the BERT tokenizer (`max_length=256`, padded /
   truncated).
3. Optimises with AdamW per the paper recipe, BCE-with-logits over six
   sigmoid heads.
4. Tracks Macro-F1 on validation (or 10% holdout if `--val` is omitted)
   and saves whenever it improves.

---

## 4. Evaluate

```bash
python -m src.BERT.eval_recognition \
    --test data/guardchat/test.jsonl \
    --weights src/BERT/weights/bert_multilabel \
    --text-kind both \
    --output results/bert_task1.json
```

Reports the same metric bundle as SafeGuider / BiLSTM:

* Macro-F1 over six categories.
* Per-category F1.
* Binary Recall and ASR (`asr = 1 - recall`).
* Single-turn vs multi-turn ASR via `--text-kind both`.

Output JSON shape:

```jsonc
{
  "single":       { "metrics": {...}, "predictions": [...] },
  "conversation": { "metrics": {...}, "predictions": [...] }
}
```

---

## 5. Python API

```python
from src.utils import load_guardchat
from src.BERT.recognition import RecognitionPipeline

pipe = RecognitionPipeline.from_pretrained("src/BERT/weights/bert_multilabel")
samples = load_guardchat("data/guardchat/test.jsonl")
preds = pipe.predict_samples(samples, kind="conversation")
for p in preds[:3]:
    print(p.to_dict())
```

---

## 6. Notes

* The notebook's `clean_text` step (URL / HTML / punctuation / emoji
  strip) is **not** applied here — BERT's WordPiece tokenizer is robust
  to those artefacts, and stripping them would distort the multi-turn
  conversational signal we are explicitly trying to evaluate.
* `--text-kind conversation` flattens the dialogue as
  `role: content\nrole: content\n...` before tokenisation. Long inputs
  are truncated to `--max-length` tokens (default 256), matching the
  CLIP / BiLSTM truncation budget.
* For experiments that swap BERT for a different encoder
  (e.g. RoBERTa), pass `--model-name roberta-base` — the
  `RecognitionTrainer` only requires that the HuggingFace AutoModel
  supports `problem_type="multi_label_classification"`.
