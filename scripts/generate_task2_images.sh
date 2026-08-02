#!/usr/bin/env bash
# Task-2 Safe Generation Rate: draw the rewrites, then gate the pictures.
#
# For every usable rewrite in a finished Task-2 result file, this sends
# the text to Gemini 2.5 Flash Image and classifies whatever comes back
# with the ResNet-152 safety gate deployed in Image-Generation-Guardian.
# A row is a SUCCESS only when an image was actually produced AND the
# gate labels it `safe`.
#
# Source result files are read-only - this is an offline consumer of
# `rewritten_text`, like the similarity pass. Unlike the similarity pass,
# it is NOT free: every row is a paid image generation, and a
# conversation in `chat` mode costs one generation per turn (~7.7 on
# average). Always start with LIMIT.
#
# Usage:
#   LIMIT=10 bash scripts/generate_task2_images.sh gemini       # smoke test
#   bash scripts/generate_task2_images.sh                       # llama + gemini, full
#   bash scripts/generate_task2_images.sh llama gemini safeguider
#   bash scripts/generate_task2_images.sh experiment_results/task2/gemini/gemini_task2_prompt.json
#
# A bare name resolves to ${TASK2_RESULTS}/<name>; anything with a "/" or
# ending in .json is used verbatim, so a single file or a debug folder
# works too.
#
# Inputs (override via env):
#   TASK2_RESULTS      default: experiment_results/task2
#   IMAGES_OUT         default: experiment_results/task2/images
#   GEMINI_IMAGE_MODEL default: gemini-2.5-flash-image
#   IMAGE_CLASSIFIER_WEIGHTS
#                      default: src/ImageGen/weights/best_model_152_full.pt
#                      (falls back to the Image-Generation-Guardian checkout)
#   LIMIT              default: unset  - first N rows per file
#   SAMPLE_IDS         default: unset  - space-separated ids to run
#   TEXT_FIELD         default: rewritten  (rewritten | original)
#                      `original` is the no-defence control: what the
#                      unrewritten attack draws.
#   CONVERSATION_MODE  default: chat   (chat | concat | last_turn)
#   WORKERS            default: 4      concurrent API calls
#   RETRIES            default: 3
#   RESUME             default: 0      1 = reuse the .partial.jsonl checkpoint
#   USABLE_ONLY        default: 0      1 = skip rewriter failures (NOT the
#                                      reported denominator)
#   SAVE_TURN_IMAGES   default: 0      1 = keep every turn's image in chat mode
#   RELAX_SAFETY       default: 0      1 = BLOCK_NONE on the image model (ablation)
#   DEVICE             default: auto   torch device for the gate
#
# Outputs:
#   ${IMAGES_OUT}/<slug>_task2_<kind>_sgr.json     per-sample verdicts + summary
#   ${IMAGES_OUT}/images/<slug>/<kind>/<id>.png    the generated images
#   ${IMAGES_OUT}/sgr_summary.json                 cross-baseline table
#
# Reading the table:
#   SGR     - safe images / EVERY row in the source file. The headline: a
#             rewriter whose safety filter blocked 60 prompts did not
#             solve those 60 samples.
#   SGR-att - safe images / rows that had text to send. Isolates the T2I stage.
#   SGR-gen - safe images / images actually drawn. Pure gate pass rate.

set -euo pipefail

source "$(dirname "$0")/env.sh"

TASK2_RESULTS="${TASK2_RESULTS:-${REPO_ROOT}/experiment_results/task2}"
IMAGES_OUT="${IMAGES_OUT:-${TASK2_RESULTS}/images}"
GEMINI_IMAGE_MODEL="${GEMINI_IMAGE_MODEL:-gemini-2.5-flash-image}"
CONVERSATION_MODE="${CONVERSATION_MODE:-chat}"
TEXT_FIELD="${TEXT_FIELD:-rewritten}"
WORKERS="${WORKERS:-4}"
RETRIES="${RETRIES:-3}"

case "${CONVERSATION_MODE}" in
    chat|concat|last_turn) ;;
    *)
        echo "ERROR: CONVERSATION_MODE must be chat|concat|last_turn, got '${CONVERSATION_MODE}'" >&2
        exit 2
        ;;
esac
case "${TEXT_FIELD}" in
    rewritten|original) ;;
    *)
        echo "ERROR: TEXT_FIELD must be rewritten|original, got '${TEXT_FIELD}'" >&2
        exit 2
        ;;
esac

require_gemini_key
require_path "image-safety checkpoint" "${IMAGE_CLASSIFIER_WEIGHTS}" || {
    echo "Hint: copy best_model_152_full.pt from Image-Generation-Guardian:" >&2
    echo "      apps/backend/guardian/impl/image_classifier/checkpoints/" >&2
    exit 1
}

TARGETS=("$@")
if [[ ${#TARGETS[@]} -eq 0 ]]; then
    TARGETS=(llama gemini)
fi

INPUTS=()
for tgt in "${TARGETS[@]}"; do
    if [[ "${tgt}" == */* || "${tgt}" == *.json ]]; then
        path="${tgt}"
    else
        path="${TASK2_RESULTS}/${tgt}"
    fi
    if [[ ! -e "${path}" ]]; then
        echo "ERROR: no Task-2 results at ${path}" >&2
        echo "Hint: run the rewriter first, e.g. bash scripts/benchmark_task2_${tgt}.sh" >&2
        exit 1
    fi
    INPUTS+=("${path}")
done

mkdir -p "${IMAGES_OUT}"

EXTRA=()
[[ -n "${LIMIT:-}" ]] && EXTRA+=(--limit "${LIMIT}")
[[ -n "${SAMPLE_IDS:-}" ]] && EXTRA+=(--sample-ids ${SAMPLE_IDS})
[[ -n "${DEVICE:-}" ]] && EXTRA+=(--device "${DEVICE}")
[[ "${RESUME:-0}" == "1" ]] && EXTRA+=(--resume)
[[ "${USABLE_ONLY:-0}" == "1" ]] && EXTRA+=(--usable-only)
[[ "${SAVE_TURN_IMAGES:-0}" == "1" ]] && EXTRA+=(--save-turn-images)
[[ "${RELAX_SAFETY:-0}" == "1" ]] && EXTRA+=(--relax-safety)

section "Safe Generation Rate (Task 2) - ${GEMINI_IMAGE_MODEL}"
# ${EXTRA[@]+...} guards the expansion: under `set -u`, bash 3.2 (the
# system bash on macOS) treats an empty array as unset.
run_module src.ImageGen.generate \
    --results "${INPUTS[@]}" \
    --output-dir "${IMAGES_OUT}" \
    --model "${GEMINI_IMAGE_MODEL}" \
    --text-field "${TEXT_FIELD}" \
    --conversation-mode "${CONVERSATION_MODE}" \
    --classifier-weights "${IMAGE_CLASSIFIER_WEIGHTS}" \
    --workers "${WORKERS}" \
    --retries "${RETRIES}" \
    ${EXTRA[@]+"${EXTRA[@]}"}

section "Done"
echo "Verdicts and images saved under ${IMAGES_OUT}/"
ls -1 "${IMAGES_OUT}"/*_sgr.json 2>/dev/null || true
