# Llama-3.1-8B-Instruct — GuardChat Task 2 Baseline (rewriting)

Zero-shot prompt rewriter for the GuardChat **NSFW concept removal**
task. Given an unsafe input $P_{unsafe}$, it produces a sanitised
$P_{safe} = g(P_{unsafe})$ that neutralises the targeted NSFW concepts
while preserving as much benign visual intent as possible.

This is the open-source counterpart to `src/Gemini/`. Both see the
**same** system prompts and the same `[Tn]` turn contract
(`src/utils/rewrite_prompt.py`) and serialise to the **same** record
schema (`src/utils/task2_eval.py`), so Table 2 compares the two models
rather than two different task framings.

Inference only — there is no `train_rewrite.py`.

---

## 1. Layout

```
src/Llama/
├── __init__.py
├── README.md
├── model.py            # LlamaModel: load + batched chat generation
├── rewrite.py          # RewritePipeline: GuardChatSample -> RewriteRecord
├── eval_rewrite.py     # CLI: rewrite both representations, checkpoint, summarise
├── download_weights.py # snapshot_download into weights/
├── requirements.txt
├── configs/
│   └── rewrite.yaml
└── weights/
    └── Llama-3.1-8B-Instruct/   # populated on first run (~16 GB)
```

---

## 2. Install, licence, weights

```bash
pip install -r src/Llama/requirements.txt
```

`transformers >= 4.43` is the floor — it is the first release with the
Llama 3.1 architecture (`rope_scaling` type `llama3`); older versions
fail to load the config.

`meta-llama/Llama-3.1-8B-Instruct` is **gated**:

1. Accept Meta's licence at
   https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct (approval is
   usually instant).
2. Put the token in the repo-root `.env` (git-ignored):
   ```
   HF_TOKEN=hf_...
   ```
   `huggingface-cli login` or `export HF_TOKEN=...` work too.
3. Download:
   ```bash
   python -m src.Llama.download_weights
   ```

~16 GB, 4 safetensors shards, into
`src/Llama/weights/Llama-3.1-8B-Instruct/`. The `original/*` folder
(Meta's duplicated raw `.pth` weights) is skipped — nothing here reads
it, and it doubles the footprint. Pass `--include-original` if you want
it anyway.

The download is optional: `eval_rewrite` populates the same folder on
first use. Doing it separately surfaces a licence problem before the GPU
is booked.

---

## 3. Run

```bash
python -m src.Llama.eval_rewrite \
    --test build_dataset/dataset/final_df_test.json \
    --text-kind all \
    --output-dir experiment_results/task2/llama
```

or via the script (both representations, model loaded once):

```bash
bash scripts/benchmark_task2_llama.sh          # all
LIMIT=10 bash scripts/benchmark_task2_llama.sh # smoke test
```

`--text-kind all` writes one file per input representation:

| File | Input |
|------|-------|
| `llama_task2_prompt.json` | the enhanced adversarial prompt |
| `llama_task2_conversation.json` | the 6–9 turn dialogue |

Checkpointed to `<output>.partial.jsonl` after every **batch**; re-run
with `--resume` to pick up where a killed process stopped.

---

## 4. Batching and the token budget

2,000 generations, each re-emitting up to ~1,500 tokens. One at a time
leaves the GPU nearly idle, so requests are batched with **left**
padding — the only correct padding side for decoder-only generation
(right padding would put pad tokens between the prompt and the first
generated token).

Llama 3.1 ships no pad token; the loader uses its reserved
`<|finetune_right_pad_id|>`, falling back to EOS.

`--batch-size` (default 8) is the main lever on both throughput and
memory. Lower it before lowering precision. `conversation` needs far
more KV cache than `prompt` — if only the conversation pass OOMs, run
the two kinds separately with different batch sizes:

```bash
bash scripts/benchmark_task2_llama.sh prompt                        # batch 8
LLAMA_BATCH_SIZE=4 bash scripts/benchmark_task2_llama.sh conversation
```

**The generation window is adaptive.** A rewrite is roughly as long as
its input, so `max_new_tokens` is derived per batch from the longest
source text (`1.4 ×` its token count `+ 96`), capped at 1024 for
`prompt` and 2560 for `conversation`. A fixed worst-case window would
make every short batch pay for the longest sample in the split. Pin it
with `--max-new-tokens` if you want the old behaviour.

Memory at a glance:

| `--dtype` | Weights | Notes |
|-----------|---------|-------|
| `bfloat16` (auto on GPU) | ~16 GB | default |
| `float16` | ~16 GB | if bf16 is unsupported |
| `float32` (auto on CPU) | ~32 GB | rarely useful |
| `int8` | ~9 GB | needs bitsandbytes (CUDA) |
| `nf4` | ~5 GB | needs bitsandbytes (CUDA) |

Add roughly `batch_size × sequence_length` of KV cache on top.

---

## 5. Retries — and why they sample

Llama-3.1-8B is a general instruct model, not a safety model. It refuses
some GuardChat inputs outright, and at 8B it sometimes breaks the `[Tn]`
turn contract. Those rows are retried up to `--retries` times (default
3, **total** attempts).

Attempt 1 is greedy. **Attempts 2+ sample** (temperature 0.7, top_p 0.9)
— a greedy re-run of the same prompt is bit-for-bit identical, so
retrying it would burn GPU time to reproduce the same failure. A run in
which nothing fails is therefore exactly reproducible.

Only `refusal`, `empty` and `parse_failed` are retried. `attempts` on
each record shows how many passes it took; `summary.num_retried` counts
the rows that needed more than one.

There is no provider here to block a request, so `status: "blocked"`
never appears — the local analogue is `refusal`.

---

## 6. Output schema

Identical to the Gemini baseline — see `src/Gemini/README.md` §5–6 for
the `[Tn]` conversation contract, the full `status` / `error_kind`
tables, and the record layout. Llama-specific notes:

* `error_kind: "oom"` means the batch died on device memory. It is an
  **infrastructure** failure, not model behaviour: lower `--batch-size`
  and re-run with `--resume`. The CLI prints an `ACTION NEEDED` line
  when any appear.
* `elapsed_sec` is the batch wall-clock divided across its rows —
  per-sample timings are meaningless under batching.
* `model` is always `meta-llama/Llama-3.1-8B-Instruct`; `meta.weights`
  records the local path actually loaded.

**No fallback text is ever substituted.** Failed rows keep an empty
`rewritten_text`. Filling them with `"a serene landscape"` would hand
the rewriter a free Safe Generation Rate point for a sample it failed to
handle.

---

## 7. Metrics are computed downstream

This step produces $P_{safe}$ only:

* **Safe Generation Rate** — feed $P_{safe}$ to FLUX.1 / Gemini Flash
  Image / DALL·E 3, then score the images with the safety gate. A sample
  succeeds **only** if an image was generated *and* it passes the gate.
* **Semantic similarity** — SBERT cosine between `original_text` and
  `rewritten_text`.

Until SBERT lands, `summary.mean_length_ratio` is the cheap stand-in; the
CLI warns when it drops below 0.5.

---

## 8. Python API

```python
from src.utils import load_guardchat
from src.Llama.rewrite import RewritePipeline

pipe = RewritePipeline.from_pretrained(batch_size=8)   # token from .env
samples = load_guardchat("build_dataset/dataset/final_df_test.json")

rec = pipe.rewrite_sample(samples[0], kind="conversation")
print(rec.status, rec.num_turns_in, "->", rec.num_turns_out)
print(rec.rewritten_text)
```
