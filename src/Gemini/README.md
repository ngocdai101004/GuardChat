# Gemini 2.5 Flash — GuardChat Task 2 Baseline (rewriting via API)

Proprietary T2I-prompt rewriter for the GuardChat **NSFW concept
removal** task, powered by the Gemini 2.5 Flash API. Output JSON
schema, CLI surface, and CLIP-similarity computation are aligned with
`src/Llama/eval_rewrite.py` and `src/SafeGuider/eval_rewrite.py`, so a
single benchmark aggregator can compose Table 2 across baselines.

There is **no training and no weight download** for this baseline -
the only setup is a Gemini API key.

---

## 1. Layout

```
src/Gemini/
├── __init__.py
├── README.md
├── client.py           # GeminiClient: lazy SDK import + safety relaxation + retry
├── rewrite.py          # RewritePipeline: GuardChatSample -> RewriteResult
├── eval_rewrite.py     # CLI: rewrite + CLIP similarity + summary
└── configs/
    └── rewrite.yaml
```

The system prompt and user template live in
`src/utils/rewrite_prompt.py` - the same file used by `src/Llama/`,
so the two Task-2 baselines see exactly the same task framing.

---

## 2. Library versions (tested)

| Package | Minimum | Why |
|---------|---------|-----|
| `google-genai` | `>= 0.3` | the new Gemini SDK (`from google import genai`); ships first-class Gemini 2.5 support |
| `torch` | `>= 2.1` | only for the shared CLIP encoder used in similarity scoring |
| `transformers` | `>= 4.43` | same CLIP encoder dependency |
| `numpy` | `>= 1.23` | summary statistics |

> The older `google-generativeai` package is not used here. If you have
> it installed, you can keep it - the two coexist on different import
> paths (`google.generativeai.*` vs `google.genai.*`). The runtime
> picks `google.genai`.

Install with:

```bash
pip install -U "google-genai>=0.3"
```

---

## 3. Authentication

1. Get a free API key from https://aistudio.google.com/.
2. Export it:

```bash
export GEMINI_API_KEY=AIzaSy...
# or:
export GOOGLE_API_KEY=AIzaSy...
```

Either env var works. You can also pass `--api-key` on the command
line, but env vars are recommended for batch runs so the key never
ends up in shell history.

---

## 4. Run rewriting

```bash
python -m src.Gemini.eval_rewrite \
    --test data/guardchat/test.jsonl \
    --model gemini-2.5-flash \
    --output results/gemini_task2.json
```

The CLI:

1. Loads each `GuardChatSample` from the test split.
2. Sends a single `generate_content` request per sample with:
   - **System instruction** = `src.utils.REWRITE_SYSTEM_PROMPT`.
   - **User content** = the GuardChat-wrapped enhanced prompt.
   - **Safety overrides** (default): `BLOCK_NONE` on the four
     overridable harm categories so Gemini is allowed to read
     adversarial GuardChat inputs and produce a sanitised rewrite.
   - **Greedy decoding**: `temperature=0`, `max_output_tokens=256`.
3. Cleans the rewrite via `cleanup_rewrite_response`.
4. Falls back to `"a serene landscape"` whenever Gemini blocks the
   request or returns an empty string, so downstream T2I evaluation
   never sees an empty rewrite.
5. Computes CLIP cosine similarity using the SafeGuider-vendored
   encoder (`openai/clip-vit-large-patch14`) for cross-baseline
   comparability.
6. Emits a JSON file matching the SafeGuider / Llama Task 2 schema,
   plus two Gemini-specific diagnostics (`blocked`, `block_reason`).

Safe Generation Rate (SGR) against FLUX.1 / Gemini Image / DALL-E 3 is
**not** computed here - feed the `rewritten_prompt` field to those
T2I systems separately.

---

## 5. Why `BLOCK_NONE`?

GuardChat's test set is **adversarial by construction**. With Gemini's
default safety thresholds, the API often refuses adversarial inputs
and returns either an empty response or a `prompt_feedback.block_reason`
of `SAFETY` / `PROHIBITED_CONTENT`. That makes the rewriter useless
for the very inputs it is meant to handle.

We therefore relax the four overridable harm categories
(`HARM_CATEGORY_HARASSMENT`, `HARM_CATEGORY_HATE_SPEECH`,
`HARM_CATEGORY_SEXUALLY_EXPLICIT`, `HARM_CATEGORY_DANGEROUS_CONTENT`)
to `BLOCK_NONE`. The system instruction still pins the model to emit
*safe* rewrites, so the *output* is constrained even though the
*input* is allowed through.

Pass `--no-relax-safety` if you instead want to measure how often
Gemini's default filter refuses GuardChat inputs at the API layer.

---

## 6. Python API

```python
from src.utils import load_guardchat
from src.Gemini.rewrite import RewritePipeline

pipe = RewritePipeline.from_api_key(
    # Reads GEMINI_API_KEY / GOOGLE_API_KEY if api_key is None.
    model_name="gemini-2.5-flash",
)
samples = load_guardchat("data/guardchat/test.jsonl")
results = pipe.rewrite_samples(samples)
for r in results[:3]:
    print(r.to_dict())
```

Each result carries the cleaned rewrite plus block-detection
diagnostics:

```jsonc
{
  "sample_id": "0001",
  "original_prompt": "...",
  "rewritten_prompt": "...",
  "was_modified": true,
  "raw_response": "...",
  "elapsed_sec": 0.42,
  "label_names": ["violence"],
  "source": "I2P",
  "blocked": false,
  "block_reason": null,
  "finish_reason": "STOP",
  "model_name": "gemini-2.5-flash"
}
```

After the eval CLI runs, each record additionally carries a
`clip_similarity` field.

---

## 7. Retry & rate-limiting

The client retries transient failures (HTTP 429 / 5xx, timeouts,
network exceptions) with exponential backoff:

```
attempt 1 -> sleep 2s -> attempt 2 -> sleep 4s -> attempt 3 -> fail
```

Tune via `--retries N` and `--backoff-seconds S`. Safety blocks are
*not* retried - re-running with the same input would not change the
verdict; instead the block is recorded on the result and the cleanup
step inserts the safe-fallback rewrite.

---

## 8. Notes

* Each request is one round-trip. With Gemini Flash the median
  latency is ~0.3-1.0 s; expect ~5-15 minutes for the full 1k-sample
  test set depending on your network.
* The CLIP-similarity step is **local** and uses the same encoder
  SafeGuider's Task 2 evaluation uses, so the `clip_similarity` field
  is directly comparable across all Task 2 baselines.
* `RewritePipeline` does **not** expose a trainer - this is an API
  baseline by definition.
