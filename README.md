# GuardChat — Benchmark Code

Reference evaluation code for the paper
**"GuardChat: Benchmarking Multi-Turn Jailbreak Attacks in T2I Systems"**.

GuardChat ships 10,000 prompt–conversation pairs across six NSFW
categories and defines two safety tasks:

| | Task 1 — Recognition | Task 2 — Rewriting |
|---|---|---|
| Goal | Multi-label classify whether a single prompt or a multi-turn conversation is unsafe, across `{sexual, illegal, shocking, violence, self-harm, harassment}`. | Rewrite an unsafe T2I prompt to remove NSFW concepts while preserving benign visual intent. |
| Metrics | **Macro-F1**, **Recall**, **ASR** = `1 – Recall` (single-turn vs multi-turn) | **CLIP cosine similarity** (this repo) + **Safe Generation Rate (SGR)** via external T2I models (out of scope here) |

This repository implements **7 baselines for Task 1** and **3 baselines
for Task 2**, all sharing a single output schema so a downstream
aggregator can compose Tables 1 and 2 of the paper.

---

## 1. Baselines

### Task 1 — Recognition

| Baseline | Type | Folder | Notes |
|----------|------|--------|-------|
| BiLSTM | supervised | `src/BiLSTM/` | PyTorch port of a stacked Bidirectional-LSTM, 6-way sigmoid head |
| BERT | supervised | `src/BERT/` | `bert-base-uncased` + multi-label head (`problem_type="multi_label_classification"`) |
| SafeGuider | supervised | `src/SafeGuider/` | CLIP-EOS embedding + 3-layer MLP, multi-label sigmoid |
| Llama-Guard 3 | zero-shot | `src/LlamaGuard/` | `meta-llama/Llama-Guard-3-8B`. 14 S-codes vs GuardChat's 6 — see [docs](docs/task1-llamaguard.md) |
| ShieldGemma-2B | zero-shot | `src/ShieldGemma/` | `google/shieldgemma-2b`, per-policy `P(Yes)` judge. 4 policies vs 6 — see [docs](docs/task1-shieldhgemma.md) |
| Qwen3Guard-Gen-8B | zero-shot | `src/Qwen3Guard/` | `Qwen/Qwen3Guard-Gen-8B`, 3-level severity + 9 categories — see [docs](docs/task1-qwen3guard.md) |
| Qwen2.5-7B | zero-shot | `src/Qwen/` | `Qwen/Qwen2.5-7B-Instruct`, custom prompt enforcing 6-key JSON output |

The three dedicated guard models each ship their own taxonomy that does
**not** match GuardChat's six categories. All three expose the same two
modes: `--mode guardchat` (the default — GuardChat's categories are
injected into the prompt, so the mapping is 1-to-1) and `--mode native`
(the model's own taxonomy plus a lossy mapping). Either way the raw model
output is stored in the results file, so any mapping can be re-derived
offline without re-running the model.

### Task 2 — Rewriting

| Baseline | Type | Folder | Notes |
|----------|------|--------|-------|
| SafeGuider | inference | `src/SafeGuider/` (`rewrite.py`) | Safety-aware beam-search over CLIP-EOS scores |
| Llama-3.1-8B | zero-shot | `src/Llama/` | `meta-llama/Llama-3.1-8B-Instruct` with the shared rewrite prompt |
| Gemini Flash | zero-shot (API) | `src/Gemini/` | `google-genai` SDK, `gemini-3.5-flash` (the paper's `gemini-2.5-flash` is retired for new keys) |

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
│   ├── ShieldGemma/           ← Task 1 zero-shot (ShieldGemma-2B)
│   ├── Qwen3Guard/            ← Task 1 zero-shot (Qwen3Guard-Gen-8B)
│   ├── Qwen/                  ← Task 1 zero-shot (Qwen2.5-7B-Instruct)
│   ├── Llama/                 ← Task 2 zero-shot (Llama-3.1-8B-Instruct)
│   └── Gemini/                ← Task 2 API (Gemini 2.5 Flash)
└── vendors/
    └── SafeGuider/             ← upstream SafeGuider modules used at runtime
        ├── classifier.py       ← ThreeLayerClassifier (binary head, Task 2)
        ├── encoder.py          ← CLIPEncoder (CLIP text encoder + EOS embedding)
        ├── beam_search.py      ← SafetyAwareBeamSearch
        └── weights/            ← SD1.4_safeguider.pt + cached CLIP snapshot
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

GuardChat is hosted on HuggingFace at
[`multimedia-synergy-lab/GuardChat`](https://huggingface.co/datasets/multimedia-synergy-lab/GuardChat).
The loader pulls from there by default — **no manual download required**.
Three splits are available:

| Split | Size | Purpose |
|-------|------|---------|
| `train` | 9,000 | Task-1 supervised training (BiLSTM, BERT, SafeGuider) |
| `test` | 1,000 | Task-1 / Task-2 evaluation (oracle-verified) |
| `full` | 10,000 | Combined; useful for analysis / stats |

Each record carries:

```jsonc
{
  "id": 1,
  "category": "sexual",                  // single string (one of 6 canonical names)
  "prompt": "...",                        // enhanced toxic prompt
  "raw_prompt": "...",                    // original prompt before enhancement
  "source": "I2P",
  "conversation_generator": "Gemma-4-31B",
  "conversation": [
    {"turn_id": 1, "role": "user", "content": "..."},
    ...
  ],
  "conversation_text": "user: ...\nuser: ..."
}
```

`src.utils.load_guardchat` is the single entry point for both backends:

```python
from src.utils import load_guardchat

# Default: pull `test` split from HuggingFace.
samples = load_guardchat()

# Explicit split:
samples = load_guardchat("multimedia-synergy-lab/GuardChat", split="train")

# Local file (JSON / JSONL / {"data": [...]}):
samples = load_guardchat("./data/guardchat/test.jsonl")
```

The loader normalises `category` (single string) into a one-hot 6-dim
`label_vector`. Older or local files using `labels` (list) /
`label_vector` (vector) / `{"sexual": 1, ...}` (dict) all work too.

**DiffusionDB safe prompts** (5,000 benign prompts mixed in for Task-1
training) are not on HF in our release — supply a local JSON list at
`./data/diffusiondb_safe.json` (override with `DIFFUSIONDB_SAFE=...`):

```jsonc
[ {"prompt": "a serene mountain landscape"},
  {"prompt": "a cute corgi puppy"},
  ... ]
```

Training proceeds without these if the file is missing — just expect
weaker safe-side recall.

### 3.4. Weights

Run the downloader for whichever baselines you plan to evaluate:

```bash
# all gated + open-access weights
bash scripts/download_weights.sh

# subset
bash scripts/download_weights.sh llamaguard qwen
```

This step does **not** download the GuardChat dataset — that happens
lazily inside the eval / train CLIs via
`datasets.load_dataset("multimedia-synergy-lab/GuardChat", split=...)`.

Disk footprint:

| Baseline | Path | Size |
|----------|------|------|
| Llama-Guard-3-8B | `src/LlamaGuard/weights/Llama-Guard-3-8B/` | ~16 GB |
| ShieldGemma-2B | `src/ShieldGemma/weights/shieldgemma-2b/` | ~5 GB |
| Qwen3Guard-Gen-8B | `src/Qwen3Guard/weights/Qwen3Guard-Gen-8B/` | ~16 GB |
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

The Python CLIs are still callable directly. By default `--train` is
the HuggingFace repo id; pass a local path to override:

```bash
# HF default
python -m src.SafeGuider.train_recognition \
    --safe  data/diffusiondb_safe.json \
    --output src/SafeGuider/weights/recognition_multilabel.pt \
    --text-kind conversation \
    --epochs 10 --batch-size 32 --lr 2e-5 --weight-decay 1e-2

# Local file
python -m src.SafeGuider.train_recognition \
    --train  data/guardchat/train.jsonl \
    --safe   data/diffusiondb_safe.json \
    --output src/SafeGuider/weights/recognition_multilabel.pt
```

(See `src/<baseline>/README.md` for the full flag list.)

---

## 5. Benchmark Task 1 — Recognition

```bash
bash scripts/benchmark_task1.sh                   # all 7 baselines
bash scripts/benchmark_task1.sh llamaguard qwen   # zero-shot subset
```

The three guard models write **three** files each — one per input
representation (`prompt`, `raw_prompt`, `conversation`) — under
`experiment_results/task1/<model>/`, via their own scripts
(`scripts/benchmark_task1_{llamaguard,shieldgemma,qwen3guard}.sh`). The
remaining baselines write one JSON each to `${RESULTS_DIR}/`:

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

### Guard-model taxonomy mode

Llama-Guard and ShieldGemma both ship a taxonomy that does not line up with
GuardChat's six categories, so each exposes two modes:

```bash
# default: GuardChat's 6 categories passed to the model directly (class counts match)
LLAMAGUARD_MODE=guardchat   bash scripts/benchmark_task1.sh llamaguard
SHIELDGEMMA_MODE=guardchat  bash scripts/benchmark_task1.sh shieldgemma

# the models exactly as released; `shocking` is unreachable in both
LLAMAGUARD_MODE=native      bash scripts/benchmark_task1.sh llamaguard
SHIELDGEMMA_MODE=native     bash scripts/benchmark_task1.sh shieldgemma
```

These two rows write three files each (`prompt` / `raw_prompt` /
`conversation`) under `experiment_results/task1/<model>/`. See
`docs/task1-llamaguard.md` and `docs/task1-shieldhgemma.md` for the
trade-offs, weight download, and HF token setup.

---

## 6. Benchmark Task 2 — Rewriting

```bash
# all 3 rewriters
bash scripts/benchmark_task2.sh

# one
bash scripts/benchmark_task2.sh gemini
bash scripts/benchmark_task2.sh safeguider
```

Each baseline rewrites **two** input representations and writes one file
per representation:

```
experiment_results/task2/<baseline>/<baseline>_task2_prompt.json
experiment_results/task2/<baseline>/<baseline>_task2_conversation.json
```

`prompt` is the enhanced adversarial prompt; `conversation` is the 6-9
turn dialogue, which must come back with its turn count intact.
`raw_prompt` is out of scope — it is the unedited seed prompt, not
something GuardChat attacks with.

All three baselines serialise the **same record schema**
(`src/utils/task2_eval.py`), so one aggregator composes Table 2:

```jsonc
{
  "prompt": {
    "summary": {
      "num_samples": 1000,
      "num_usable": 940,          // status == "ok" and non-empty
      "status_counts": {"ok": 940, "blocked": 60},
      "error_kind_counts": {"provider_block": 60},
      "fraction_modified": 0.97,
      "mean_length_ratio": 0.80   // stand-in until SBERT lands
    },
    "rewrites": [
      {
        "sample_id": "0001",
        "text_kind": "prompt",
        "status": "ok",           // ok | refusal | blocked | empty | parse_failed | error
        "error_kind": null,       // why it failed, when it did
        "original_text": "...",
        "rewritten_text": "...",  // EMPTY on failure - never a fallback string
        "was_modified": true,
        "model": "gemini-3.5-flash",
        "gold_category": "violence",
        "extra": { }              // baseline-specific diagnostics
      }
    ]
  },
  "meta": { }
}
```

`status` says *that* a row failed; `error_kind` says *why*, separating
"this sample is unrewritable" (`provider_block`, `model_refusal`) from
"the run hit a limit" (`quota`, `auth`, `oom`, ...). The CLIs print an
`ACTION NEEDED` line when the second kind appears — an exhausted quota
must not be read as a bad rewriter.

Failed rows keep an **empty** `rewritten_text`. No fallback text is ever
substituted: `"a serene landscape"` would score a free SGR point.

Runs are checkpointed to `<output>.partial.jsonl` after every sample;
re-run with `RESUME=1` to pick up where a kill left off.

### Safe Generation Rate (SGR)

SGR is **not computed in this repo**. It needs the `rewritten_text`
field pushed through external T2I systems (FLUX.1, Gemini Flash Image,
DALL-E 3) and the resulting images judged by the safety gate. A rewrite
counts as a success only if an image is produced **and** it passes the
gate — blocked by the T2I filter and generated-but-NSFW are both
failures.

Semantic similarity (SBERT) is likewise computed downstream from the
same field.

### Gemini-specific knobs

```bash
export GEMINI_API_KEY=AIzaSy...          # or put it in the repo-root .env

GEMINI_MODEL=gemini-3.5-flash \
GEMINI_WORKERS=4 bash scripts/benchmark_task2.sh gemini
```

The client relaxes Gemini's safety thresholds to `BLOCK_NONE` so the
model is allowed to *read* adversarial GuardChat input and emit a
sanitised rewrite. Rows the provider still blocks after 3 attempts are
recorded as `blocked` / `provider_block` with empty text — a real
result, not an error.

### SafeGuider-specific knobs

SafeGuider does not generate: it **deletes words** until the recognizer's
`P[safe]` clears 0.80. Needs `vendors/SafeGuider/weights/SD1.4_safeguider.pt`
plus the CLIP text encoder (open access, fetched on first use).

```bash
SAFEGUIDER_GATE=recognizer bash scripts/benchmark_task2.sh safeguider
```

`--gate recognizer` (default) reproduces the published pipeline: prompts
the recognizer judges safe pass through **unmodified**, which is a real
SafeGuider failure mode. `--gate always` rewrites every row instead.
See `src/SafeGuider/README.md` §4 for the CLIP 77-token limitation,
which bites on GuardChat's 99-word average prompt.

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
datasets           >= 2.18         (load multimedia-synergy-lab/GuardChat)
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
└── task2/                        ← Task 2 rewrites (see §6)
    ├── gemini/gemini_task2_{prompt,conversation}.json
    ├── llama/llama_task2_{prompt,conversation}.json
    └── safeguider/safeguider_task2_{prompt,conversation}.json
```

All Task-1 JSONs share the same `{"single": {...}, "conversation": {...}}`
structure. All Task-2 JSONs share the same
`{"<kind>": {"summary": {...}, "rewrites": [...]}, "meta": {...}}`
structure, with baseline-specific diagnostics confined to each record's
`extra` object.

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
* `src/Qwen/README.md`
* `src/Llama/README.md`
* `src/Gemini/README.md`
* `src/ShieldGemma/` — *(see `docs/task1-shieldhgemma.md`)*
* `src/LlamaGuard/` — *(see `docs/task1-llamaguard.md`)*
* `scripts/README.md` — bash entry points