#!/usr/bin/env bash
# Score finished Task-2 rewrites for semantic preservation with SBERT.
#
# This is an OFFLINE metric pass. It reads the JSON files the rewriters
# already wrote and never calls a rewriter, so it costs no API budget and
# can be re-run whenever the metric definition changes. Source files are
# read-only; results land in a separate folder.
#
# Metric: cosine similarity between mean-pooled, L2-normalised
# `sentence-transformers/all-mpnet-base-v2` embeddings of the original
# and the rewritten text. This REPLACES the CLIP cosine similarity of the
# first revision - CLIP's text tower is trained to align text with
# images, not text with text, and its 77-token window truncates ~80% of
# GuardChat's enhanced prompts. SBERT is trained on text pairs and reads
# 384 tokens.
#
# Usage:
#   bash scripts/eval_task2_similarity.sh                    # llama + gemini
#   bash scripts/eval_task2_similarity.sh llama              # one baseline
#   bash scripts/eval_task2_similarity.sh llama gemini safeguider
#   bash scripts/eval_task2_similarity.sh experiment_results/task2/llama/debug10
#
# A bare name is resolved to ${TASK2_RESULTS}/<name>; anything containing
# a "/" or ending in .json is used as a path verbatim, so a debug folder
# or a single file works too.
#
# Inputs (override via env):
#   TASK2_RESULTS   default: experiment_results/task2
#   SIMILARITY_OUT  default: experiment_results/task2/similarity
#   SBERT_WEIGHTS   default: src/SBERT/weights/all-mpnet-base-v2
#                   populated on first run, or ahead of time with
#                   `bash scripts/download_weights.sh sbert`
#   DEVICE          default: auto  (cuda when available)
#   SBERT_BATCH_SIZE default: 64   throughput only; scores do not depend on it
#   MAX_SEQ_LENGTH  default: unset (384, read from the checkpoint)
#   PER_TURN        default: 1     also score conversations turn by turn
#
# Outputs:
#   ${SIMILARITY_OUT}/<slug>_task2_<kind>_sbert.json   per-sample scores
#   ${SIMILARITY_OUT}/sbert_similarity_summary.json    cross-baseline table
#
# Reading the table:
#   mean       - scored rows only; rewriter failures are excluded
#   penalised  - failures counted as 0.0. Compare rewriters on THIS, or a
#                model that refuses half the dataset looks like the best
#                one at preserving meaning.
#   per-turn   - conversations only, immune to the 384-token window

set -euo pipefail

source "$(dirname "$0")/env.sh"

TASK2_RESULTS="${TASK2_RESULTS:-${REPO_ROOT}/experiment_results/task2}"
SIMILARITY_OUT="${SIMILARITY_OUT:-${TASK2_RESULTS}/similarity}"
SBERT_BATCH_SIZE="${SBERT_BATCH_SIZE:-64}"
PER_TURN="${PER_TURN:-1}"

TARGETS=("$@")
if [[ ${#TARGETS[@]} -eq 0 ]]; then
    TARGETS=(llama gemini)
fi

# Resolve each target to a path, then check it exists up front so a typo
# fails before the encoder is loaded.
INPUTS=()
for tgt in "${TARGETS[@]}"; do
    if [[ "${tgt}" == */* || "${tgt}" == *.json ]]; then
        path="${tgt}"
    else
        path="${TASK2_RESULTS}/${tgt}"
    fi
    if [[ ! -e "${path}" ]]; then
        echo "ERROR: no Task-2 results at ${path}" >&2
        echo "Hint: run the baseline first, e.g. bash scripts/benchmark_task2_${tgt}.sh" >&2
        exit 1
    fi
    INPUTS+=("${path}")
done

mkdir -p "${SIMILARITY_OUT}"

EXTRA=()
[[ -n "${DEVICE:-}" ]] && EXTRA+=(--device "${DEVICE}")
[[ -n "${MAX_SEQ_LENGTH:-}" ]] && EXTRA+=(--max-seq-length "${MAX_SEQ_LENGTH}")
[[ "${PER_TURN}" == "0" ]] && EXTRA+=(--no-per-turn)

section "SBERT semantic similarity (Task 2)"
# ${EXTRA[@]+...} guards the expansion: under `set -u`, bash 3.2 (the
# system bash on macOS) treats an empty array as unset.
run_module src.SBERT.eval_similarity \
    --results "${INPUTS[@]}" \
    --weights "${SBERT_WEIGHTS}" \
    --batch-size "${SBERT_BATCH_SIZE}" \
    --output-dir "${SIMILARITY_OUT}" \
    ${EXTRA[@]+"${EXTRA[@]}"}

section "Done"
echo "Similarity scores saved under ${SIMILARITY_OUT}/"
ls -1 "${SIMILARITY_OUT}"/*_sbert.json 2>/dev/null || true
