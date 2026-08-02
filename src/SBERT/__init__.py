"""SBERT semantic similarity for GuardChat Task 2.

Not a baseline - a metric. Task 2 asks a rewriter to strip NSFW concepts
while keeping the request intact, and this module measures the second
half: how close is $P_{safe}$ to $P_{unsafe}$ in meaning?

The metric was CLIP cosine similarity in the first revision. A reviewer
objected that CLIP's text tower is trained to align text with images
rather than text with text, and is therefore not a text-to-text
similarity space. The objection also lands on a second, more mechanical
point: CLIP reads 77 tokens, and GuardChat's enhanced prompts average 99
words. ``sentence-transformers/all-mpnet-base-v2`` is trained
contrastively on 1B+ text pairs and reads 384, so it addresses both.

Scoring runs offline over finished Task-2 result files - see
:mod:`src.SBERT.eval_similarity` - so it costs nothing to re-run and the
rewriter outputs are never modified.
"""

from .clip_baseline import (
    DEFAULT_CLIP_MODEL,
    CLIPSimilarityEncoder,
)
from .model import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LOCAL_DIR,
    DEFAULT_MODEL_NAME,
    FALLBACK_MAX_SEQ_LENGTH,
    IGNORE_PATTERNS,
    SBERTEncoder,
    load_encoder,
    mean_pool,
    read_max_seq_length,
)
from .similarity import (
    load_result_file,
    score_records,
    summarise_scores,
)

__all__ = [
    "CLIPSimilarityEncoder",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CLIP_MODEL",
    "DEFAULT_LOCAL_DIR",
    "DEFAULT_MODEL_NAME",
    "FALLBACK_MAX_SEQ_LENGTH",
    "IGNORE_PATTERNS",
    "SBERTEncoder",
    "load_encoder",
    "load_result_file",
    "mean_pool",
    "read_max_seq_length",
    "score_records",
    "summarise_scores",
]
