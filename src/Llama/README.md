# Llama-3.1-8B-Instruct — GuardChat Task 2 Baseline (rewriting)

Zero-shot evaluation of `meta-llama/Llama-3.1-8B-Instruct` on the
GuardChat **NSFW concept removal via prompt rewriting** task. Output
JSON schema, CLI surface, and CLIP-similarity computation are aligned
with `src/SafeGuider/eval_rewrite.py` so a single benchmark aggregator
can compose Table 2 across baselines.

There is **no training** for this baseline. The rewriter is configured
entirely by the shared system prompt in
`src/utils/rewrite_prompt.py`, which is also reused by
`src/Gemini/`.

---

## 1. Layout

```
src/Llama/
├── __init__.py
├── README.md
├── model.py                # LlamaModel: load + apply chat template + generate
├── rewrite.py              # RewritePipeline (no trainer)
├── download_weights.py     # CLI: snapshot_download into weights/
├── eval_rewrite.py         # CLI: run rewriting + CLIP similarity
├── configs/
│   └── rewrite.yaml
└── weights/
    └── README.md           # how to populate weights/Llama-3.1-8B-Instruct/
```

The system prompt + user template + response cleanup live in
`src/utils/rewrite_prompt.py`. Both `src/Llama/` and `src/Gemini/`
import them so the two baselines see exactly the same task framing.

---

## 2. Library versions (tested)

| Package | Minimum | Why |
|---------|---------|-----|
| `torch` | `>= 2.1` | required by `transformers >= 4.43` |
| `transformers` | `>= 4.43` | Llama 3.1 architecture (same as LlamaGuard) |
| `accelerate` | `>= 0.26` | `device_map='auto'` |
| `huggingface_hub` | `>= 0.20` | gated `snapshot_download` |
| `bitsandbytes` | `>= 0.43` (optional) | `--dtype int8` / `--dtype nf4` |

---

## 3. Download weights

Llama-3.1-8B-Instruct is **gated**. Steps:

1. Accept the licence at
   https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
2. Authenticate: `huggingface-cli login` or `export HF_TOKEN=hf_...`
3. Download:

```bash
python -m src.Llama.download_weights \
    --local-dir src/Llama/weights/Llama-3.1-8B-Instruct
```

Disk: ~16 GB (`.safetensors`, fp16). Subsequent loads are fully offline.

---

## 4. Run rewriting

```bash
python -m src.Llama.eval_rewrite \
    --test data/guardchat/test.jsonl \
    --weights src/Llama/weights/Llama-3.1-8B-Instruct \
    --dtype bfloat16 \
    --output results/llama_task2.json
```

The CLI:

1. Loads each `GuardChatSample` from the test split.
2. Runs the rewrite chat template (system + user) through the model.
3. Cleans the model output (`cleanup_rewrite_response`) and falls back
   to a generic `"a serene landscape"` if cleanup yields an empty
   string.
4. Computes CLIP cosine similarity between the original and the
   rewritten prompt using the **same vendored encoder as SafeGuider's
   Task 2** (`openai/clip-vit-large-patch14` by default), so the
   similarity columns are directly comparable across baselines.
5. Emits a JSON file matching the SafeGuider Task 2 schema.

Safe Generation Rate (SGR) against FLUX.1 / Gemini Image / DALL-E 3 is
**not** computed here. Feed the `rewritten_prompt` field to those T2I
systems in a separate pipeline.

### Memory-constrained GPUs

```bash
# 8-bit weights  (~9 GB GPU)
--dtype int8

# 4-bit NF4      (~5 GB GPU)
--dtype nf4
```

Both require `bitsandbytes >= 0.43`.

---

## 5. Prompt design

`src/utils/rewrite_prompt.py::SYSTEM_PROMPT` defines:

1. The task framing - rewrite an unsafe T2I prompt to remove NSFW
   content while preserving benign visual intent.
2. The six GuardChat NSFW categories with definitions paraphrased from
   Appendix A of the paper.
3. Four explicit rules:
   - Remove implications, not just exact words.
   - Preserve neutral visual elements when present.
   - Do not add new content.
   - Fall back to a generic safe alternative if necessary.
4. A hard format constraint: output ONLY the rewritten prompt, no
   prose / quotes / markdown.

The user template wraps the prompt in `<<< ... >>>` delimiters so the
model does not accidentally fold the input into its own output.

`cleanup_rewrite_response` then strips common leftover decorations
(markdown fences, "Here is the rewritten prompt:" preambles, outer
quote pairs).

### Custom prompts

Pass `--system-prompt-file path/to/your.txt` to swap in an alternative
system prompt for prompt-engineering ablations.

---

## 6. Python API

```python
from src.utils import load_guardchat
from src.Llama.rewrite import RewritePipeline

pipe = RewritePipeline.from_pretrained(
    weights="src/Llama/weights/Llama-3.1-8B-Instruct",
    dtype="bfloat16",
)
samples = load_guardchat("data/guardchat/test.jsonl")
results = pipe.rewrite_samples(samples)
for r in results[:3]:
    print(r.to_dict())
```

Each result carries the cleaned rewrite alongside the model's raw
output for transparency:

```jsonc
{
  "sample_id": "0001",
  "original_prompt": "...",
  "rewritten_prompt": "...",
  "was_modified": true,
  "raw_response": "Here is the rewritten prompt:\n\"...\"",
  "elapsed_sec": 1.83,
  "label_names": ["violence"],
  "source": "I2P"
}
```

After the eval CLI runs, each record additionally carries a
`clip_similarity` field.

---

## 7. Notes

* The rewrite pipeline always produces a non-empty rewrite. If the
  model returns nothing or the cleanup yields an empty string, we fall
  back to `"a serene landscape"` so downstream T2I evaluation never
  receives an empty prompt.
* Greedy decoding (`do_sample=False`) is the default - we want a
  deterministic rewrite, not creative variance. Pass
  `--max-new-tokens` higher only if you observe truncation; 200 tokens
  (~150 words) handles GuardChat's enhanced prompts comfortably.
* `RewritePipeline` does **not** expose a trainer. To fine-tune
  Llama-3.1 on GuardChat rewriting, use a separate LoRA pipeline (out
  of scope for this evaluation code).
