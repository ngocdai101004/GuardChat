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
| `download_weights.sh` | Snapshot-download Llama-Guard-3-8B / Llama-3.1-8B-Instruct (gated, needs HF token) and Qwen2.5-7B-Instruct + the CLIP encoder used by SafeGuider + the SBERT similarity encoder (open access). |
| `train_task1_supervised.sh` | Train the three supervised Task-1 baselines (BiLSTM, BERT, SafeGuider). Recipe matches paper Section 6.1. |
| `benchmark_task1.sh` | Evaluate Task 1 across all seven baselines (BiLSTM, BERT, SafeGuider, Llama-Guard, Qwen, ShieldGemma, Qwen3Guard). |
| `benchmark_task1_llamaguard.sh` | Llama-Guard-3-8B alone, over the three input representations. Weights auto-download into `src/LlamaGuard/weights/`. See `docs/task1-llamaguard.md`. |
| `benchmark_task1_shieldgemma.sh` | ShieldGemma-2B alone, same three representations. See `docs/task1-shieldhgemma.md`. |
| `benchmark_task1_qwen3guard.sh` | Qwen3Guard-Gen-8B alone, same three representations. Ungated repo, so no HF token needed. See `docs/task1-qwen3guard.md`. |
| `benchmark_task2.sh` | Evaluate Task 2 across the three baselines (SafeGuider beam-search, Llama-3.1-8B, Gemini Flash API). |
| `benchmark_task2_gemini.sh` | Gemini Flash alone, over both Task-2 representations (enhanced prompt + conversation). Resumable; API key from env or `.env`. See `src/Gemini/README.md`. |
| `benchmark_task2_llama.sh` | Llama-3.1-8B-Instruct alone, same two representations. Gated repo — needs `HF_TOKEN`. See `src/Llama/README.md`. |
| `benchmark_task2_safeguider.sh` | SafeGuider beam-search rewriter alone, same two representations. No API key and no generation — needs `SD1.4_safeguider.pt`. See `src/SafeGuider/README.md`. |
| `eval_task2_similarity.sh` | Score finished Task-2 rewrites for semantic preservation with SBERT (`all-mpnet-base-v2`). Offline metric pass over `rewritten_text` — no rewriter runs, no API budget. Replaces the CLIP similarity of the first revision. See `src/SBERT/README.md`. |
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

# Subset: only Gemini for Task 2 (both representations)
bash scripts/benchmark_task2_gemini.sh

# Smoke test it on 10 samples first
LIMIT=10 bash scripts/benchmark_task2_gemini.sh

# Quick iteration: skip the long training step
SKIP_TRAIN=1 bash scripts/benchmark_all.sh
```

## Outputs

* `experiment_results/task1/{baseline}/{baseline}_task1_{kind}.json` — Task 1
  metrics + predictions, one file per input representation.
* `experiment_results/task2/{gemini,llama,safeguider}/{baseline}_task2_{prompt,conversation}.json`
  — Task 2 rewrites + summary, one file per representation.
* `experiment_results/task2/similarity/{baseline}_task2_{prompt,conversation}_sbert.json`
  — SBERT semantic similarity per sample, plus `sbert_similarity_summary.json`
  with the cross-baseline table.
* `${RESULTS_DIR}/{baseline}_train_history.json` — per-epoch training metrics
  for the supervised models.

`${RESULTS_DIR}` defaults to `./results/`.

## Required env vars

* `HF_TOKEN` (or a prior `huggingface-cli login`) — only for the gated
  Meta repos (`download_weights.sh llamaguard llama`).
* `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) — only for the Gemini Task-2
  rewriter. Can live in the repo-root `.env` instead of the environment.

Everything else has a sensible default in `env.sh`.

## Override knobs

| Env var | Default | What it controls |
|---------|---------|------------------|
| `GUARDCHAT_DATASET` | `multimedia-synergy-lab/GuardChat` | HF repo id used by both `GUARDCHAT_TRAIN` and `GUARDCHAT_TEST` defaults |
| `GUARDCHAT_TRAIN` | `${GUARDCHAT_DATASET}` | Train data source: HF repo id or local JSON/JSONL path |
| `GUARDCHAT_TEST` | `${GUARDCHAT_DATASET}` | Test data source (same format as train) |
| `GUARDCHAT_TRAIN_SPLIT` | `train` | HF split when `GUARDCHAT_TRAIN` is a repo id |
| `GUARDCHAT_TEST_SPLIT` | `test` | HF split when `GUARDCHAT_TEST` is a repo id |
| `DIFFUSIONDB_SAFE` | `${DATA_DIR}/diffusiondb_safe.json` | Safe prompts (label = 0) — local file only |
| `DATA_DIR` | `${REPO_ROOT}/data` | Root data folder for local files (DiffusionDB safe) |
| `RESULTS_DIR` | `${REPO_ROOT}/results` | Where eval JSONs land |
| `PYTHON` | `python` | Python interpreter (point at venv) |
| `DTYPE` | `bfloat16` | LLM weight dtype: `bfloat16 / float16 / int8 / nf4` |
| `TEXT_KIND` | `both` | Task 1 representation: `single / conversation / both` |
| `EPOCHS` | `10` | Training epochs |
| `BATCH_SIZE` | `32` | Training batch size |
| `LR` | `2e-5` | Training learning rate |
| `WD` | `1e-2` | Training weight decay |
| `DEVICE` | unset | Force `cuda` / `cpu` (else `device_map='auto'`) |
| `LLAMAGUARD_MODE` | `guardchat` | Llama-Guard taxonomy: `guardchat / native` |
| `LLAMAGUARD_CONV_FORMAT` | `concat` | Multi-turn encoding: `concat / turns` |
| `SHIELDGEMMA_MODE` | `guardchat` | ShieldGemma policy set: `guardchat / native` |
| `QWEN3GUARD_MODE` | `guardchat` | Qwen3Guard taxonomy: `guardchat / native` |
| `QWEN3GUARD_CONV_FORMAT` | `concat` | Multi-turn encoding: `concat / turns` |
| `QWEN3GUARD_CONTROVERSIAL` | `unsafe` | How Qwen3Guard's middle severity level is read: `unsafe / safe` |
| `GEMINI_MODEL` | `gemini-3.5-flash` | Gemini model id. The paper's `gemini-2.5-flash` is retired for new API keys — see `src/Gemini/README.md` §2 |
| `GEMINI_TEST` | `build_dataset/dataset/final_df_test.json` | Test split for `benchmark_task2_gemini.sh` |
| `GEMINI_OUT` | `experiment_results/task2/gemini` | Where the Task-2 Gemini rewrites land |
| `GEMINI_WORKERS` | `4` | Concurrent Gemini API requests; lower on a rate-limited key |
| `GEMINI_TEMPERATURE` | `0.0` | Gemini sampling temperature |
| `LLAMA_TEST` | `build_dataset/dataset/final_df_test.json` | Test split for `benchmark_task2_llama.sh` |
| `LLAMA_OUT` | `experiment_results/task2/llama` | Where the Task-2 Llama rewrites land |
| `LLAMA_BATCH_SIZE` | `8` | Sequences generated together; lower this first on OOM |
| `LLAMA_RETRIES` | `3` | Attempts per sample; attempts after the first are sampled |
| `DTYPE_LLAMA` | `auto` | Llama Task-2 weight dtype |
| `SAFEGUIDER_TEST` | `build_dataset/dataset/final_df_test.json` | Test split for `benchmark_task2_safeguider.sh` |
| `SAFEGUIDER_OUT` | `experiment_results/task2/safeguider` | Where the Task-2 SafeGuider rewrites land |
| `SAFEGUIDER_ENCODER` | `openai/clip-vit-large-patch14` | Text encoder; must match the recognizer checkpoint |
| `SAFEGUIDER_GATE` | `recognizer` | `recognizer` = published pipeline (safe-judged prompts pass through untouched); `always` = rewrite every row |
| `SAFEGUIDER_BATCH_SIZE` | `64` | Beam candidates per encoder pass; throughput only |
| `BEAM_WIDTH` / `MAX_DEPTH` | `6` / `25` | Beam-search size, upstream defaults |
| `SAFETY_THRESHOLD` / `SIMILARITY_FLOOR` | `0.70` / `0.10` | Beam-search accept thresholds. `0.70` **deviates** from upstream's `0.80` |
| `PATIENCE` | `10` | Abandon a beam search after N depths with no gain. **No upstream equivalent**; set `0` for published behaviour |
| `TASK2_RESULTS` | `experiment_results/task2` | Root scanned by `eval_task2_similarity.sh` for finished rewrites |
| `SIMILARITY_OUT` | `${TASK2_RESULTS}/similarity` | Where SBERT similarity sidecars land |
| `SBERT_WEIGHTS` | `src/SBERT/weights/all-mpnet-base-v2` | Similarity encoder snapshot; auto-populated on first run |
| `SBERT_BATCH_SIZE` | `64` | Texts per encoder pass; throughput only, scores do not depend on it |
| `MAX_SEQ_LENGTH` | unset | Override the encoder's 384-token window (mpnet caps at 512) |
| `PER_TURN` | `1` | Set `0` to skip the per-turn conversation similarity |
| `LIMIT` | unset | Cap the sample count (smoke tests) |
| `RESUME` | `0` | Set to `1` to reuse a `.partial.jsonl` checkpoint |

To pin a local file instead of the HF default, just override:

```bash
GUARDCHAT_TEST=/mnt/guardchat/test.jsonl bash scripts/benchmark_task1.sh
```
