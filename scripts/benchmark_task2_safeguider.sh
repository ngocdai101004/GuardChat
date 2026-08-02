#!/usr/bin/env bash
# Run Task 2 (NSFW Concept Removal via Prompt Rewriting) with SafeGuider.
#
# Not a language model: SafeGuider scores the CLIP EOS embedding with the
# pre-trained binary recognizer and beam-searches over WORD DELETIONS
# until the score crosses the safe threshold. Nothing is generated, so
# there is no API key and no refusal - but also no substitution: the
# output is always a subsequence of the input.
#
# Needs vendors/SafeGuider/weights/SD1.4_safeguider.pt from the upstream
# release. The CLIP text encoder is fetched into
# vendors/SafeGuider/weights/clip-vit-large-patch14/ on first run.
#
# Usage:
#   bash scripts/benchmark_task2_safeguider.sh               # both cases
#   bash scripts/benchmark_task2_safeguider.sh prompt        # one case
#   bash scripts/benchmark_task2_safeguider.sh prompt conversation
#
# Cases:
#   prompt        enhanced adversarial prompt  -> safeguider_task2_prompt.json
#   conversation  6-9 turn dialogue, rewritten turn by turn
#                                              -> safeguider_task2_conversation.json
#   all           shorthand for both
#
# Inputs (override via env):
#   SAFEGUIDER_TEST        default: build_dataset/dataset/final_df_test.json
#   SAFEGUIDER_BINARY_WEIGHTS
#                          default: vendors/SafeGuider/weights/SD1.4_safeguider.pt
#   SAFEGUIDER_ENCODER     default: openai/clip-vit-large-patch14
#                          must match the checkpoint's training encoder
#   SAFEGUIDER_OUT         default: experiment_results/task2/safeguider
#   SAFEGUIDER_GATE        default: recognizer  (recognizer | always)
#                          'recognizer' reproduces the published pipeline:
#                          prompts the recognizer calls safe are passed
#                          through unmodified. 'always' rewrites every row.
#   SAFEGUIDER_BATCH_SIZE  default: 64   candidates per encoder pass;
#                          throughput only, results do not depend on it
#   BEAM_WIDTH             default: 6
#   MAX_DEPTH              default: 25
#   PATIENCE               default: 0 (upstream: always run to MAX_DEPTH)
#                          set N to abandon a search after N depths with no
#                          gain. Cuts the hopeless cases, which are also the
#                          slowest. NOT upstream - report it with results.
#   SAFETY_THRESHOLD       default: 0.80
#   SIMILARITY_FLOOR       default: 0.10
#   DEVICE                 default: unset (cuda when available)
#   LIMIT                  optional cap on samples, for smoke tests
#   RESUME                 set to 1 to reuse a .partial.jsonl checkpoint
#
# This step produces P_safe only. Safe Generation Rate and semantic
# similarity are computed downstream from the `rewritten_text` field -
# see src/SafeGuider/README.md.

set -euo pipefail

source "$(dirname "$0")/env.sh"

SAFEGUIDER_TEST="${SAFEGUIDER_TEST:-${REPO_ROOT}/build_dataset/dataset/final_df_test.json}"
SAFEGUIDER_OUT="${SAFEGUIDER_OUT:-${REPO_ROOT}/experiment_results/task2/safeguider}"
SAFEGUIDER_ENCODER="${SAFEGUIDER_ENCODER:-openai/clip-vit-large-patch14}"
SAFEGUIDER_GATE="${SAFEGUIDER_GATE:-recognizer}"
SAFEGUIDER_BATCH_SIZE="${SAFEGUIDER_BATCH_SIZE:-64}"
BEAM_WIDTH="${BEAM_WIDTH:-6}"
MAX_DEPTH="${MAX_DEPTH:-25}"
PATIENCE="${PATIENCE:-0}"
SAFETY_THRESHOLD="${SAFETY_THRESHOLD:-0.80}"
SIMILARITY_FLOOR="${SIMILARITY_FLOOR:-0.10}"

TARGETS=("$@")
if [[ ${#TARGETS[@]} -eq 0 ]]; then
    TARGETS=(all)
fi

# Expand the shorthand and validate every requested case up front, so a
# typo fails before the encoder is loaded.
KINDS=()
for tgt in "${TARGETS[@]}"; do
    case "${tgt}" in
        all)                  KINDS+=(prompt conversation) ;;
        prompt|conversation)  KINDS+=("${tgt}") ;;
        *)
            echo "Unknown case: ${tgt}" >&2
            echo "Choose from: prompt | conversation | all" >&2
            exit 2
            ;;
    esac
done

require_data "SAFEGUIDER_TEST" "${SAFEGUIDER_TEST}"
require_path "SAFEGUIDER_BINARY_WEIGHTS" "${SAFEGUIDER_BINARY_WEIGHTS}"
mkdir -p "${SAFEGUIDER_OUT}"

EXTRA=()
[[ -n "${DEVICE:-}" ]] && EXTRA+=(--device "${DEVICE}")
[[ -n "${LIMIT:-}" ]] && EXTRA+=(--limit "${LIMIT}")
[[ "${RESUME:-0}" == "1" ]] && EXTRA+=(--resume)

# Both cases go through a single process so CLIP is loaded once.
if [[ ${#KINDS[@]} -eq 2 ]]; then
    KINDS=(all)
fi

for kind in "${KINDS[@]}"; do
    section "Rewrite with SafeGuider beam search (Task 2) - ${kind}"
    # ${EXTRA[@]+...} guards the expansion: under `set -u`, bash 3.2
    # (the system bash on macOS) treats an empty array as unset.
    run_module src.SafeGuider.eval_rewrite \
        --test "${SAFEGUIDER_TEST}" \
        --weights "${SAFEGUIDER_BINARY_WEIGHTS}" \
        --encoder-model "${SAFEGUIDER_ENCODER}" \
        --text-kind "${kind}" \
        --gate "${SAFEGUIDER_GATE}" \
        --beam-width "${BEAM_WIDTH}" \
        --max-depth "${MAX_DEPTH}" \
        --safety-threshold "${SAFETY_THRESHOLD}" \
        --similarity-floor "${SIMILARITY_FLOOR}" \
        --batch-size "${SAFEGUIDER_BATCH_SIZE}" \
        --patience "${PATIENCE}" \
        --output-dir "${SAFEGUIDER_OUT}" \
        ${EXTRA[@]+"${EXTRA[@]}"}
done

section "Done"
echo "Rewrites saved under ${SAFEGUIDER_OUT}/"
ls -1 "${SAFEGUIDER_OUT}"/safeguider_task2_*.json 2>/dev/null || true
