# GuardChat — Benchmark Code

Reference evaluation code for the paper
**"GuardChat: Benchmarking Multi-Turn Jailbreak Attacks in T2I Systems"**.

GuardChat ships 10,000 prompt–conversation pairs across six NSFW
categories and defines two safety tasks:

| | Task 1 — Recognition | Task 2 — Rewriting |
|---|---|---|
| Goal | Multi-label classify whether a single prompt or a multi-turn conversation is unsafe, across `{sexual, illegal, shocking, violence, self-harm, harassment}`. | Rewrite an unsafe T2I prompt to remove NSFW concepts while preserving benign visual intent. |
| Metrics | **Macro-F1**, **Recall**, **ASR** = `1 – Recall` (single-turn vs multi-turn) | **CLIP cosine similarity** (this repo) + **Safe Generation Rate (SGR)** via external T2I models (out of scope here) |

This repository implements **5 baselines for Task 1** and **3 baselines
for Task 2**, all sharing a single output schema so a downstream
aggregator can compose Tables 1 and 2 of the paper.

---

## 1. Baselines

### Task 1 — Recognition

| Baseline | Type | Folder | Notes |
|----------|------|--------|-------|
| BiLSTM | supervised | `src/BiLSTM/` | PyTorch port of `vendors/BiLSTM/sentiment_analysis.py`, 6-way sigmoid head |
| BERT | supervised | `src/BERT/` | `bert-base-uncased` + multi-label head (`problem_type="multi_label_classification"`) |
| SafeGuider | supervised | `src/SafeGuider/` | CLIP-EOS embedding + 3-layer MLP, multi-label sigmoid |
| Llama-Guard 3 | zero-shot | `src/LlamaGuard/` | `meta-llama/Llama-Guard-3-8B`, native S1–S14 taxonomy mapped to GuardChat 6 |
| Qwen2.5-7B | zero-shot | `src/Qwen/` | `Qwen/Qwen2.5-7B-Instruct`, custom prompt enforcing 6-key JSON output |

### Task 2 — Rewriting

| Baseline | Type | Folder | Notes |
|----------|------|--------|-------|
| SafeGuider | inference | `src/SafeGuider/` (`rewrite.py`) | Safety-aware beam-search over CLIP-EOS scores |
| Llama-3.1-8B | zero-shot | `src/Llama/` | `meta-llama/Llama-3.1-8B-Instruct` with the shared rewrite prompt |
| Gemini 2.5 Flash | zero-shot (API) | `src/Gemini/` | `google-genai` SDK, `gemini-2.5-flash` |

Shared GuardChat conventions live in `src/utils/` (canonical category
order, `GuardChatSample`, all metrics, the rewrite prompt).

---

## 2. Repository layout

```
.
├── README.md                  ← you are here
├── requirements.txt
├── scripts/                   ← bash entry points (download / train / benchmark)
│   ├── env.sh
│   ├── download_weights.sh
│   ├── train_task1_supervised.sh
│   ├── benchmark_task1.sh
│   ├── benchmark_task2.sh
│   ├── benchmark_all.sh
│   └── README.md
├── src/
│   ├── utils/                 ← shared data loader + metrics + rewrite prompt
│   ├── BiLSTM/                ← Task 1 supervised
│   ├── BERT/                  ← Task 1 supervised
│   ├── SafeGuider/            ← Task 1 supervised + Task 2 beam-search rewriter
│   ├── LlamaGuard/            ← Task 1 zero-shot (Llama-Guard-3-8B)
│   ├── Qwen/                  ← Task 1 zero-shot (Qwen2.5-7B-Instruct)
│   ├── Llama/                 ← Task 2 zero-shot (Llama-3.1-8B-Instruct)
│   └── Gemini/                ← Task 2 API (Gemini 2.5 Flash)
└── vendors/                   ← upstream reference code (SafeGuider, BiLSTM, BERT)
    ├── SafeGuider/
    ├── BiLSTM/
    └── BERT/
```

Each baseline package has the same shape:

```
src/<baseline>/
├── __init__.py
├── README.md                  ← per-baseline usage, prompt details, etc.
├── model.py                   ← model wrapper (or classifier.py for SafeGuider)
├── recognition.py | rewrite.py
├── train_recognition.py       ← only for supervised Task-1 baselines
├── eval_recognition.py | eval_rewrite.py
├── download_weights.py        ← only for LLM baselines
├── configs/
│   └── recognition.yaml | rewrite.yaml
└── weights/                   ← created on first download / train
```

Per-baseline READMEs cover prompt design, taxonomy mapping, and
quirks; this top-level guide covers the end-to-end flow.

---

## 3. Setup

### 3.1. Python environment

Tested with Python 3.10 / 3.11 / 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
# Optional: install a CUDA-matched PyTorch wheel first.
# pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

`requirements.txt` covers every baseline. `bitsandbytes` is conditional
(`platform_system != "Darwin"`) — macOS users must skip 4-bit / 8-bit
quantisation and run with `bfloat16` / `float16`.

### 3.2. Authentication

| Resource | When you need it | How |
|----------|------------------|-----|
| HuggingFace token | Downloading the gated Meta repos (`Llama-Guard-3-8B`, `Llama-3.1-8B-Instruct`) | `huggingface-cli login` (writes `~/.cache/huggingface/token`) **or** `export HF_TOKEN=hf_...` |
| Gemini API key | Running the Gemini Task-2 baseline | `export GEMINI_API_KEY=...` (free key: <https://aistudio.google.com/>) |

You also have to **accept the Meta licences** at
[Llama-Guard-3-8B](https://huggingface.co/meta-llama/Llama-Guard-3-8B)
and
[Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
before the gated downloads succeed.

### 3.3. Data

Place the GuardChat splits and the DiffusionDB safe prompts under
`./data/` (override with `DATA_DIR=...`):

```
data/
├── guardchat/
│   ├── train.jsonl        ← 9,000 GuardChat training samples
│   └── test.jsonl         ← 1,000 oracle-verified test samples
└── diffusiondb_safe.json  ← 5,000 benign prompts (label = 0) for Task-1 training
```

Each GuardChat record is a dict with at least:

```jsonc
{
  "id": "0001",
  "enhanced_prompt": "...",
  "conversation": [
    {"turn_id": 1, "role": "user", "content": "..."},
    ...
  ],
  "labels": ["sexual", "violence"],   // OR a 6-dim 0/1 label_vector
  "source": "I2P"                      // optional
}
```

`src.utils.load_guardchat` accepts:
- `.json` list, `.json` `{"data": [...]}`, or `.jsonl` (one record per line)
- `labels` as list of names, dict (`{"sexual": 1, ...}`), or 6-dim vector
- snake_case / underscore aliases (`self_harm`, `nudity`, ...)

### 3.4. Weights

Run the downloader for whichever baselines you plan to evaluate:

```bash
# all gated + open-access weights
bash scripts/download_weights.sh

# subset
bash scripts/download_weights.sh llamaguard qwen
```

Disk footprint:

| Baseline | Path | Size |
|----------|------|------|
| Llama-Guard-3-8B | `src/LlamaGuard/weights/Llama-Guard-3-8B/` | ~16 GB |
| Llama-3.1-8B-Instruct | `src/Llama/weights/Llama-3.1-8B-Instruct/` | ~16 GB |
| Qwen2.5-7B-Instruct | `src/Qwen/weights/Qwen2.5-7B-Instruct/` | ~15 GB |
| CLIP ViT-L/14 (SafeGuider) | `vendors/SafeGuider/weights/clip-vit-large-patch14/` | ~600 MB |

The SafeGuider Task-2 binary classifier (`SD1.4_safeguider.pt`) is **not
on HuggingFace**. Obtain it from the upstream SafeGuider release and
place it at `vendors/SafeGuider/weights/SD1.4_safeguider.pt`. The
downloader will warn if it is missing.

The BiLSTM, BERT, and SafeGuider Task-1 multi-label heads have **no
pre-trained checkpoints** — they are trained from scratch on
GuardChat. See §4.

---

## 4. Train (Task 1, supervised baselines only)

Reproduces the recipe in Section 6.1 of the paper:
9,000 GuardChat conversational + 5,000 DiffusionDB safe samples,
AdamW (lr = 2 × 10⁻⁵, weight decay = 0.01), batch 32, 10 epochs,
BCE-with-logits over six categories.

```bash
# train all three (BiLSTM, BERT, SafeGuider)
bash scripts/train_task1_supervised.sh

# train one
bash scripts/train_task1_supervised.sh bilstm

# tweak hyperparameters from the shell
EPOCHS=15 BATCH_SIZE=64 LR=5e-5 \
  bash scripts/train_task1_supervised.sh safeguider
```

Outputs:

| Baseline | Checkpoint |
|----------|------------|
| BiLSTM | `src/BiLSTM/weights/bilstm_multilabel.pt` (single bundle: state-dict + vocab + `max_len`) |
| BERT | `src/BERT/weights/bert_multilabel/` (HuggingFace `save_pretrained` directory) |
| SafeGuider | `src/SafeGuider/weights/recognition_multilabel.pt` (multi-label MLP state-dict) |

Per-epoch metrics are written to `${RESULTS_DIR}/<baseline>_train_history.json`.

The five-baseline Python CLIs are still callable directly:

```bash
python -m src.SafeGuider.train_recognition \
    --train data/guardchat/train.jsonl \
    --safe  data/diffusiondb_safe.json \
    --output src/SafeGuider/weights/recognition_multilabel.pt \
    --text-kind conversation \
    --epochs 10 --batch-size 32 --lr 2e-5 --weight-decay 1e-2
```

(See `src/<baseline>/README.md` for the full flag list.)

---

## 5. Benchmark Task 1 — Recognition

```bash
bash scripts/benchmark_task1.sh                   # all 5 baselines
bash scripts/benchmark_task1.sh llamaguard qwen   # zero-shot subset
```

The script writes one JSON per baseline to `${RESULTS_DIR}/`:

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
        "probs": {"sexual": 0.93, ...},        // omitted for LLM baselines
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

### Memory-constrained GPUs

For the LLM baselines, drop weight precision via `DTYPE`:

```bash
# 8-bit weights  (~9 GB GPU)
DTYPE=int8 bash scripts/benchmark_task1.sh llamaguard

# 4-bit NF4      (~5 GB GPU)
DTYPE=nf4  bash scripts/benchmark_task1.sh qwen
```

Both `int8` and `nf4` require `bitsandbytes>=0.43` (Linux/Windows GPU only).

### Llama-Guard taxonomy mode

```bash
# default: native S1-S14 taxonomy + post-hoc mapping (no `shocking` predictions)
LLAMAGUARD_MODE=native bash scripts/benchmark_task1.sh llamaguard

# alternative: pass GuardChat 6 categories directly into the chat template
LLAMAGUARD_MODE=custom bash scripts/benchmark_task1.sh llamaguard
```

See `src/LlamaGuard/README.md` for the trade-offs.

---

## 6. Benchmark Task 2 — Rewriting

```bash
# all 3 rewriters
bash scripts/benchmark_task2.sh

# one
bash scripts/benchmark_task2.sh gemini
```

Outputs per baseline land at `${RESULTS_DIR}/<baseline>_task2.json`:

```jsonc
{
  "summary": {
    "num_samples": 1000,
    "fraction_modified": 0.97,
    "mean_clip_similarity": 0.351,
    "model": "Llama-3.1-8B-Instruct",
    "clip_encoder": "openai/clip-vit-large-patch14"
    // (Gemini also reports "fraction_blocked")
  },
  "rewrites": [
    {
      "sample_id": "0001",
      "original_prompt": "...",
      "rewritten_prompt": "...",
      "was_modified": true,
      "raw_response": "...",
      "elapsed_sec": 1.83,
      "label_names": ["violence"],
      "source": "I2P",
      "clip_similarity": 0.412
      // SafeGuider also reports {removed_tokens, original_safety,
      //                          modified_safety, safeguider_similarity}
      // Gemini also reports     {blocked, block_reason, finish_reason,
      //                          model_name}
    },
    ...
  ]
}
```

CLIP cosine similarity uses the **same vendored encoder for all
baselines** (`openai/clip-vit-large-patch14`), so similarity columns
are directly comparable.

### Safe Generation Rate (SGR)

SGR is **not computed in this repo** — it requires running the
`rewritten_prompt` field through external T2I systems (FLUX.1, Gemini
Image, DALL-E 3) and judging the resulting images. Use the JSON
outputs above as the input to that downstream pipeline.

### Gemini-specific knobs

```bash
# Required:
export GEMINI_API_KEY=AIzaSy...

# Optional:
GEMINI_MODEL=gemini-2.5-flash  bash scripts/benchmark_task2.sh gemini
```

By default the client relaxes Gemini's safety thresholds to
`BLOCK_NONE` so the model is allowed to *read* adversarial GuardChat
inputs and emit a sanitised rewrite. Pass `--no-relax-safety` (via the
direct CLI) to keep default thresholds and measure block rate.

---

## 7. End-to-end pipeline

```bash
# One-shot: download all weights, train supervised Task-1 baselines,
# evaluate both tasks across every baseline.
bash scripts/download_weights.sh
bash scripts/benchmark_all.sh

# Or skip training and reuse existing weights:
SKIP_TRAIN=1 bash scripts/benchmark_all.sh
```

`benchmark_all.sh` simply chains `train_task1_supervised.sh`,
`benchmark_task1.sh`, and `benchmark_task2.sh` — read it for a quick
mental model of the whole flow.

---

## 8. Library version pins

The shared baseline:

```
torch              >= 2.1
transformers       >= 4.43         (Llama 3.1 / Qwen2 / multi-label HF)
accelerate         >= 0.26
huggingface_hub    >= 0.20
safetensors        >= 0.4.2
numpy              >= 1.23
tqdm               >= 4.60
```

Optional, only when actually used:

```
google-genai       >= 0.3          (src/Gemini/)
bitsandbytes       >= 0.43         (--dtype int8 / nf4 on Linux/Windows)
```

If `transformers < 4.43`, `from_pretrained` fails with `KeyError:
'llama'` / `KeyError: 'qwen2'`. Upgrade with
`pip install -U "transformers>=4.43"`.

---

## 9. Output schema cheatsheet

```
results/
├── bilstm_train_history.json     ← per-epoch loss / macro_f1
├── bert_train_history.json
├── safeguider_train_history.json
│
├── bilstm_task1.json             ← Task 1 metrics + per-sample predictions
├── bert_task1.json
├── safeguider_task1.json
├── llamaguard_task1.json
├── qwen_task1.json
│
├── safeguider_task2.json         ← Task 2 rewrites + summary
├── llama_task2.json
└── gemini_task2.json
```

All Task-1 JSONs share the same `{"single": {...}, "conversation": {...}}`
structure; all Task-2 JSONs share the same `{"summary": {...}, "rewrites": [...]}`
structure with optional baseline-specific extra fields.

---

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `KeyError: 'llama'` / `KeyError: 'qwen2'` during `from_pretrained` | `transformers < 4.43` | `pip install -U "transformers>=4.43"` |
| `OSError: You are trying to access a gated repo` | HF token missing or licence not accepted | `huggingface-cli login`, accept licence on the model page |
| `RuntimeError: ... safetensors ... non-contiguous tensor` (BERT save) | Older `safetensors` plus a backbone that produces non-contiguous weights | Already mitigated in `src/BERT/model.py:save_pretrained` (uses `safe_serialization=False`); upgrade `safetensors` if it still bites |
| `RuntimeError: ... bitsandbytes` | `--dtype int8 / nf4` on macOS or no CUDA | Switch to `DTYPE=bfloat16` |
| Gemini returns empty / `block_reason="SAFETY"` for many samples | Default safety thresholds blocking adversarial inputs | Already mitigated by the default `BLOCK_NONE` overrides; pass `--no-relax-safety` only when measuring block rate |
| `Vendored SafeGuider not found` | Repo cloned without `vendors/` | Re-clone or check `.gitignore` did not strip the folder |
| BiLSTM / BERT eval crashes with "weights not found" | Forgot to train first | Run `scripts/train_task1_supervised.sh <baseline>` |

---

## 11. Per-baseline references

For prompt design, taxonomy mapping, hyperparameters, and quirks, see
the respective README:

* `src/utils/README.md` — *(implicit; everything is in code docstrings)*
* `src/BiLSTM/README.md`
* `src/BERT/README.md`
* `src/SafeGuider/README.md`
* `src/LlamaGuard/README.md`
* `src/Qwen/README.md`
* `src/Llama/README.md`
* `src/Gemini/README.md`
* `scripts/README.md` — bash entry points

---

## 12. Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{guardchat2026,
  title  = {GuardChat: Benchmarking Multi-Turn Jailbreak Attacks in T2I Systems},
  author = {Tran, Ngoc-Dai and Huynh, Thanh-Tuong and Le, Trung-Nghia},
  booktitle = {NeurIPS 2026 Datasets and Benchmarks Track},
  year   = {2026}
}
```

---

## 13. License

See `LICENSE`.

The vendored upstream code (`vendors/`) carries its own licence.
GuardChat itself is released for **research use only**. Do not use the
released adversarial prompts to attack production T2I systems.
