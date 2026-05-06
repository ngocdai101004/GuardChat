# BiLSTM — GuardChat Task 1 Baseline

PyTorch port of the stacked Bidirectional-LSTM in
`vendors/BiLSTM/sentiment_analysis.py`, adapted to the GuardChat
multi-label unsafe text recognition task. The CLI surface, output JSON
schema, and metric definitions are kept identical to
`src/SafeGuider/` so a single benchmark aggregator can compose Table 1
across both baselines.

---

## 1. Layout

```
src/BiLSTM/
├── __init__.py
├── README.md
├── tokenizer.py            # preprocess_text + word vocab (port of notebook)
├── model.py                # PyTorch BiLSTM (multi-label sigmoid head)
├── recognition.py          # RecognitionPipeline + RecognitionTrainer
├── train_recognition.py    # CLI: train
├── eval_recognition.py     # CLI: evaluate
└── configs/
    └── recognition.yaml
```

Shared GuardChat conventions live in `src/utils/`:

* `CATEGORIES` — canonical six-category ordering.
* `GuardChatSample`, `load_guardchat`, `load_safe_prompts`.
* `summarise_recognition`, `macro_f1`, `recall_score`, `attack_success_rate`.

---

## 2. Architecture

| Layer | Spec |
|-------|------|
| Embedding | `vocab_size -> 100`, `padding_idx=0` |
| BiLSTM #1 | hidden=128, `bidirectional=True`, returns full sequence |
| BatchNorm1d + Dropout(0.5) | over `2·128 = 256` channels |
| BiLSTM #2 | hidden=64, `bidirectional=True`, takes final hidden state |
| BatchNorm1d + Dropout(0.5) | over `2·64 = 128` channels |
| Dense | `128 -> 64`, ReLU + Dropout(0.5) |
| Output | `64 -> 6` logits (sigmoid head, BCEWithLogits loss) |

Differences vs the reference notebook:

1. Final layer is **6-way sigmoid** (multi-label) instead of 4-way
   softmax (single-label). The binary verdict is recovered as
   `any(prob ≥ threshold)`, matching the SafeGuider convention.
2. Keras' L2 kernel regularisation is approximated by AdamW
   `weight_decay` (paper recipe: `1e-2`).
3. `torchtext` is replaced with a stdlib word tokenizer + a JSON-
   serialisable `Vocab`, so the package has zero brittle dependencies.

---

## 3. Training

```bash
python -m src.BiLSTM.train_recognition \
    --train data/guardchat/train.jsonl \
    --safe  data/diffusiondb_safe.json \
    --output src/BiLSTM/weights/bilstm_multilabel.pt \
    --text-kind conversation \
    --epochs 10 --batch-size 32 --lr 2e-5 --weight-decay 1e-2
```

The trainer:

1. Loads GuardChat samples + DiffusionDB safe prompts via the shared
   `src.utils.load_guardchat` / `load_safe_prompts`.
2. Builds a vocab from the **training corpus only** (no test leakage).
3. Encodes texts to fixed-length integer sequences with `Vocab.encode_text`.
4. Optimises with **AdamW (lr=2e-5, weight_decay=1e-2)**, batch 32,
   10 epochs, BCEWithLogits over six categories.
5. Saves a single `.pt` bundling state dict, `BiLSTMConfig`, `Vocab.itos`,
   and `max_len` so downstream evaluation needs only one path.

---

## 4. Evaluation

```bash
python -m src.BiLSTM.eval_recognition \
    --test data/guardchat/test.jsonl \
    --weights src/BiLSTM/weights/bilstm_multilabel.pt \
    --text-kind both \
    --output results/bilstm_task1.json
```

Reports the same metric bundle as SafeGuider:

* Macro-F1 over six categories.
* Per-category F1 (`f1_sexual`, `f1_illegal`, ...).
* Binary Recall and Attack Success Rate (`asr = 1 - recall`).
* Single-turn vs multi-turn ASR via `--text-kind both`.

The output JSON schema matches `results/safeguider_task1.json` exactly:

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
from src.BiLSTM.recognition import RecognitionPipeline

pipe = RecognitionPipeline.from_pretrained(
    "src/BiLSTM/weights/bilstm_multilabel.pt",
)
samples = load_guardchat("data/guardchat/test.jsonl")
preds = pipe.predict_samples(samples, kind="conversation")
for p in preds[:3]:
    print(p.to_dict())
```

---

## 6. Notes

* The vocab is built only on the training corpus, so OOV words at test
  time map to `<unk>` (id=1). The `<pad>` id is fixed at 0 and matches
  `nn.Embedding(padding_idx=0)`.
* `--text-kind conversation` flattens the dialogue as
  `role: content\nrole: content\n...` before tokenisation. Long inputs
  are truncated to `--max-len` tokens (default 256).
* The vendored notebook used Keras' L2 regulariser; PyTorch's AdamW
  applies decoupled weight decay equivalently for the dense and dense-
  recurrent weight matrices, but does **not** regularise the biases or
  the BatchNorm gain/bias - this matches the standard PyTorch idiom.
