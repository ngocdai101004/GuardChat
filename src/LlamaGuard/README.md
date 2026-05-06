# Llama-Guard-3-8B — GuardChat Task 1 Baseline (zero-shot)

Zero-shot evaluation of `meta-llama/Llama-Guard-3-8B` on the GuardChat
multi-label unsafe text recognition task. CLI surface, output JSON
schema, and metric definitions are kept identical to
`src/SafeGuider/`, `src/BiLSTM/`, and `src/BERT/` so a single benchmark
aggregator can compose Table 1 across all baselines.

There is **no training** for this baseline - the model is used purely
zero-shot.

---

## 1. Layout

```
src/LlamaGuard/
├── __init__.py
├── README.md
├── taxonomy.py             # S1-S14 ↔ GuardChat 6-category mapping
├── model.py                # LlamaGuardModel: load + apply chat template + generate
├── recognition.py          # RecognitionPipeline (no trainer)
├── download_weights.py     # CLI: snapshot_download into weights/
├── eval_recognition.py     # CLI: evaluate
├── configs/
│   └── recognition.yaml
└── weights/
    └── README.md           # how to populate weights/Llama-Guard-3-8B/
```

Shared GuardChat conventions (`CATEGORIES`, `GuardChatSample`,
`summarise_recognition`, ...) live in `src/utils/`.

---

## 2. Library versions (tested)

| Package | Minimum | Why |
|---------|---------|-----|
| `torch` | `>= 2.1` | required by `transformers >= 4.43` |
| `transformers` | `>= 4.43` | Llama 3.1 architecture support |
| `accelerate` | `>= 0.26` | `device_map='auto'` |
| `huggingface_hub` | `>= 0.20` | gated `snapshot_download` |
| `bitsandbytes` | `>= 0.43` (optional) | `--dtype int8` / `--dtype nf4` |

If `transformers < 4.43`, `from_pretrained` fails with
`KeyError: 'llama'` or `ValueError`. Upgrade with
`pip install -U "transformers>=4.43"`.

---

## 3. Download weights

Llama-Guard-3-8B is **gated**. Steps:

1. Accept the licence at
   https://huggingface.co/meta-llama/Llama-Guard-3-8B
2. Authenticate: `huggingface-cli login` or `export HF_TOKEN=hf_...`
3. Download:

```bash
python -m src.LlamaGuard.download_weights \
    --local-dir src/LlamaGuard/weights/Llama-Guard-3-8B
```

Disk: ~16 GB (`.safetensors`, fp16). The optional `original/*.pth`
duplicates are skipped by default; pass `--include-original` to keep
them.

Subsequent loads are fully offline.

---

## 4. Evaluate

```bash
python -m src.LlamaGuard.eval_recognition \
    --test data/guardchat/test.jsonl \
    --weights src/LlamaGuard/weights/Llama-Guard-3-8B \
    --mode native \
    --dtype bfloat16 \
    --text-kind both \
    --output results/llamaguard_task1.json
```

Outputs the same Macro-F1 / Recall / ASR metric bundle as the other
baselines, separately for `single` and `conversation` text kinds.

### Memory-constrained GPUs

```bash
# 8-bit weights  (~9 GB GPU)
--dtype int8

# 4-bit NF4      (~5 GB GPU)
--dtype nf4
```

Both require `bitsandbytes >= 0.43`. CPU-only inference works but is
extremely slow (~minutes per sample).

---

## 5. Two taxonomy modes

### `--mode native` (default)

Llama-Guard-3 was fine-tuned on Meta's hazard taxonomy with codes
`S1`-`S14` (Violent Crimes, Sex-Related Crimes, Hate, ...). Each S-code
is mapped to a GuardChat category via
`taxonomy.SCODE_TO_GUARDCHAT`:

| S-code | Label | → GuardChat |
|--------|-------|-------------|
| S1     | Violent Crimes | violence |
| S2     | Non-Violent Crimes | illegal |
| S3     | Sex-Related Crimes | sexual |
| S4     | Child Sexual Exploitation | sexual |
| S5     | Defamation | harassment |
| S6     | Specialized Advice | illegal |
| S7     | Privacy | illegal |
| S8     | Intellectual Property | illegal |
| S9     | Indiscriminate Weapons | violence |
| S10    | Hate | harassment |
| S11    | Suicide & Self-Harm | self-harm |
| S12    | Sexual Content | sexual |
| S13    | Elections | illegal |
| S14    | Code Interpreter Abuse | illegal |

The Llama-Guard taxonomy has **no analogue for GuardChat's `shocking`**
(gore / body-horror), so this mode never fires that class. Expect
`f1_shocking ≈ 0` in the reported metrics.

### `--mode custom`

Pass GuardChat's six categories straight into the chat template
(`tokenizer.apply_chat_template(..., categories=...)`). The model then
predicts over `S1=Sexual, S2=Illegal, S3=Shocking, S4=Violence,
S5=Self-harm, S6=Harassment` directly. This recovers the `shocking`
class but typically degrades accuracy on the natively-trained
classes. Useful as an upper-bound comparison; native is the canonical
mode for the Table 1 row.

---

## 6. Python API

```python
from src.utils import load_guardchat
from src.LlamaGuard.recognition import RecognitionPipeline

pipe = RecognitionPipeline.from_pretrained(
    weights="src/LlamaGuard/weights/Llama-Guard-3-8B",
    mode="native",
    dtype="bfloat16",
)
samples = load_guardchat("data/guardchat/test.jsonl")
preds = pipe.predict_samples(samples, kind="conversation")
for p in preds[:3]:
    print(p.to_dict())
```

Each prediction record carries the model's raw text output and the
parsed S-codes alongside the GuardChat-aligned label vector:

```jsonc
{
  "sample_id": "0001",
  "text": "user: ...\nuser: ...",
  "multi_label": {"sexual": 1, "illegal": 0, "shocking": 0,
                  "violence": 0, "self-harm": 0, "harassment": 0},
  "predicted_categories": ["sexual"],
  "binary_pred": 1,
  "raw_response": "\n\nunsafe\nS3",
  "scodes": ["S3"],
  "label_vector_true": {"sexual": 1, ...}
}
```

Note: there is no `probs` field - Llama-Guard emits discrete S-codes
rather than per-category probabilities.

---

## 7. Notes

* Llama-Guard moderates the **most recent user turn** in the context
  of preceding turns. For `--text-kind conversation` we forward the
  multi-turn dialogue verbatim as a list of `{"role", "content"}`
  dicts (rather than concatenating into one string), which matches the
  model's training distribution.
* Evaluation is single-sample per call by default; batching gives
  marginal speedups for an 8B causal LM and complicates the chat
  template handling, so we keep the loop simple. With a single A100
  80GB and `bfloat16`, evaluating the 1k-sample test set takes ~3-5
  minutes for `single` and ~5-10 minutes for `conversation`.
* `RecognitionPipeline` does **not** expose a `RecognitionTrainer` -
  Llama-Guard is zero-shot only in this benchmark. To fine-tune it
  on GuardChat, use a separate LoRA training pipeline (out of scope
  for this evaluation code).
