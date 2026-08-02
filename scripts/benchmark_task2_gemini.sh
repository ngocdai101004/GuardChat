#!/usr/bin/env bash
# Run Task 2 (NSFW Concept Removal via Prompt Rewriting) with Gemini Flash.
#
# API-only and zero-shot: no training step and no weight download. The
# only requirement is GEMINI_API_KEY (or GOOGLE_API_KEY) in the
# environment or in the repo-root .env file.
#
# Usage:
#   bash scripts/benchmark_task2_gemini.sh                # both cases
#   bash scripts/benchmark_task2_gemini.sh prompt         # one case
#   bash scripts/benchmark_task2_gemini.sh prompt conversation
#
# Cases:
#   prompt        enhanced adversarial prompt  -> gemini_task2_prompt.json
#   conversation  6-9 turn dialogue            -> gemini_task2_conversation.json
#   all           shorthand for both
#
# Inputs (override via env):
#   GEMINI_API_KEY      required (GOOGLE_API_KEY also accepted; .env works)
#   GEMINI_TEST         default: build_dataset/dataset/final_df_test.json
#   GEMINI_MODEL        default: the CLI default (gemini-3.5-flash - the id
#                       named in the paper, gemini-2.5-flash, is retired for
#                       new API keys; see src/Gemini/README.md section 2)
#   GEMINI_OUT          default: experiment_results/task2/gemini
#   GEMINI_WORKERS      default: 4   concurrent requests; lower on a
#                       rate-limited key
#   GEMINI_TEMPERATURE  default: 0.0
#   LIMIT               optional cap on samples, for smoke tests
#   RESUME              set to 1 to reuse a .partial.jsonl checkpoint
#
# This step produces P_safe only. Safe Generation Rate and semantic
# similarity are computed downstream from the `rewritten_text` field -
# see src/Gemini/README.md section 7.

set -euo pipefail

source "$(dirname "$0")/env.sh"

GEMINI_TEST="${GEMINI_TEST:-${REPO_ROOT}/build_dataset/dataset/final_df_test.json}"
GEMINI_OUT="${GEMINI_OUT:-${REPO_ROOT}/experiment_results/task2/gemini}"
GEMINI_WORKERS="${GEMINI_WORKERS:-4}"
GEMINI_TEMPERATURE="${GEMINI_TEMPERATURE:-0.0}"

TARGETS=("$@")
if [[ ${#TARGETS[@]} -eq 0 ]]; then
    TARGETS=(all)
fi

# Expand the shorthand and validate every requested case up front, so a
# typo fails before any API call is billed.
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

require_data "GEMINI_TEST" "${GEMINI_TEST}"
require_gemini_key
mkdir -p "${GEMINI_OUT}"

EXTRA=()
[[ -n "${LIMIT:-}" ]] && EXTRA+=(--limit "${LIMIT}")
[[ "${RESUME:-0}" == "1" ]] && EXTRA+=(--resume)
[[ -n "${GEMINI_MODEL:-}" ]] && EXTRA+=(--model "${GEMINI_MODEL}")

# Both cases in one process: the client, the key check and the preflight
# call are set up once instead of twice.
if [[ ${#KINDS[@]} -eq 2 ]]; then
    KINDS=(all)
fi

for kind in "${KINDS[@]}"; do
    section "Rewrite with Gemini Flash (Task 2, zero-shot) - ${kind}"
    # ${EXTRA[@]+...} guards the expansion: under `set -u`, bash 3.2
    # (the system bash on macOS) treats an empty array as unset.
    run_module src.Gemini.eval_rewrite \
        --test "${GEMINI_TEST}" \
        --text-kind "${kind}" \
        --workers "${GEMINI_WORKERS}" \
        --temperature "${GEMINI_TEMPERATURE}" \
        --output-dir "${GEMINI_OUT}" \
        ${EXTRA[@]+"${EXTRA[@]}"}
done

section "Done"
echo "Rewrites saved under ${GEMINI_OUT}/"
ls -1 "${GEMINI_OUT}"/gemini_task2_*.json 2>/dev/null || true
