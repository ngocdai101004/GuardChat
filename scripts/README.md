# scripts/ — bash entry points

Thin wrappers around the per-baseline Python CLIs. Every script sources
`scripts/env.sh` first to resolve paths and runtime defaults, so a
single `export` overrides settings for the whole run, e.g.:

```bash
DATA_DIR=/mnt/guardchat RESULTS_DIR=/mnt/results DTYPE=nf4 \
  bash scripts/benchmark_task1.sh
```

## File index

| Script | Purpose |
|--------|---------|
| `env.sh` | Sourced by every other script. Defines path / runtime defaults and helper functions. Override anything by `export`-ing before invoking. |
| `download_weights.sh` | Snapshot-download Llama-Guard-3-8B / Llama-3.1-8B-Instruct (gated, needs HF token) and Qwen2.5-7B-Instruct + the CLIP encoder used by SafeGuider (open access). |
| `train_task1_supervised.sh` | Train the three supervised Task-1 baselines (BiLSTM, BERT, SafeGuider). Recipe matches paper Section 6.1. |
| `benchmark_task1.sh` | Evaluate Task 1 across the five baselines (BiLSTM, BERT, SafeGuider, Llama-Guard, Qwen). |
| `benchmark_task2.sh` | Evaluate Task 2 across the three baselines (SafeGuider beam-search, Llama-3.1-8B, Gemini 2.5 Flash API). |
| `benchmark_all.sh` | End-to-end: train + Task 1 + Task 2. `SKIP_TRAIN=1` to skip the training step. |

## Common usage

All scripts accept positional baseline names. Omit them to run every
baseline of the relevant kind:

```bash
# Run everything end-to-end (download -> train -> evaluate both tasks)
bash scripts/download_weights.sh
bash scripts/benchmark_all.sh

# Subset: only the zero-shot Task-1 baselines
bash scripts/benchmark_task1.sh llamaguard qwen

# Subset: only Gemini for Task 2
bash scripts/benchmark_task2.sh gemini

# Quick iteration: skip the long training step
SKIP_TRAIN=1 bash scripts/benchmark_all.sh
```

## Outputs

* `${RESULTS_DIR}/{baseline}_task1.json` — Task 1 metrics + predictions.
* `${RESULTS_DIR}/{baseline}_task2.json` — Task 2 rewrites + summary.
* `${RESULTS_DIR}/{baseline}_train_history.json` — per-epoch training metrics
  for the supervised models.

`${RESULTS_DIR}` defaults to `./results/`.

## Required env vars

* `HF_TOKEN` (or a prior `huggingface-cli login`) — only for the gated
  Meta repos (`download_weights.sh llamaguard llama`).
* `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) — only for `benchmark_task2.sh
  gemini`.

Everything else has a sensible default in `env.sh`.

## Override knobs

| Env var | Default | What it controls |
|---------|---------|------------------|
| `DATA_DIR` | `${REPO_ROOT}/data` | Root data folder |
| `GUARDCHAT_TRAIN` | `${DATA_DIR}/guardchat/train.jsonl` | Train split |
| `GUARDCHAT_TEST` | `${DATA_DIR}/guardchat/test.jsonl` | Test split |
| `DIFFUSIONDB_SAFE` | `${DATA_DIR}/diffusiondb_safe.json` | Safe prompts (label = 0) |
| `RESULTS_DIR` | `${REPO_ROOT}/results` | Where eval JSONs land |
| `PYTHON` | `python` | Python interpreter (point at venv) |
| `DTYPE` | `bfloat16` | LLM weight dtype: `bfloat16 / float16 / int8 / nf4` |
| `TEXT_KIND` | `both` | Task 1 representation: `single / conversation / both` |
| `EPOCHS` | `10` | Training epochs |
| `BATCH_SIZE` | `32` | Training batch size |
| `LR` | `2e-5` | Training learning rate |
| `WD` | `1e-2` | Training weight decay |
| `DEVICE` | unset | Force `cuda` / `cpu` (else `device_map='auto'`) |
| `LLAMAGUARD_MODE` | `native` | Llama-Guard taxonomy: `native / custom` |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model id |
