#!/usr/bin/env bash
# Run Task 2 (NSFW Concept Removal via Prompt Rewriting) benchmark.
#
# Usage:
#   bash scripts/benchmark_task2.sh                  # all 3 baselines
#   bash scripts/benchmark_task2.sh safeguider llama # subset
#   bash scripts/benchmark_task2.sh gemini           # API only
#
# Targets:
#   safeguider  (beam-search rewriter, needs SD1.4_safeguider.pt locally)
#   llama       (Llama-3.1-8B-Instruct, needs local snapshot)
#   gemini      (Gemini Flash API, needs GEMINI_API_KEY - env or .env)
#   all         shorthand for safeguider llama gemini
#
# Inputs (override via env):
#   GUARDCHAT_TEST         default: multimedia-synergy-lab/GuardChat (HF)
#   GUARDCHAT_TEST_SPLIT   default: test
#
# Outputs (one file per input representation, resumable):
#   gemini      experiment_results/task2/gemini/gemini_task2_{prompt,conversation}.json
#   llama       experiment_results/task2/llama/llama_task2_{prompt,conversation}.json
#   safeguider  experiment_results/task2/safeguider/safeguider_task2_{prompt,conversation}.json
#
# All three baselines share the Task-2 record schema
# (src/utils/task2_eval.py), so one aggregator composes Table 2. Use
# scripts/benchmark_task2_{gemini,llama,safeguider}.sh directly to tune
# the per-baseline knobs.
#
# Safe Generation Rate (SGR) is NOT computed here - feed the rewritten
# text to FLUX.1 / Gemini Flash Image / DALL-E 3 in a separate pipeline
# and score the images with the safety gate.

set -euo pipefail

source "$(dirname "$0")/env.sh"

TARGETS=("$@")
if [[ ${#TARGETS[@]} -eq 0 ]]; then
    TARGETS=(safeguider llama gemini)
fi

require_data "GUARDCHAT_TEST" "${GUARDCHAT_TEST}"

# Each baseline has its own script, carrying knobs the others have no
# analogue for (beam width here, batch size there, worker count for the
# API). Delegate rather than duplicate the flag lists.
eval_safeguider() {
    SAFEGUIDER_TEST="${GUARDCHAT_TEST}" \
    bash "${SCRIPT_DIR}/benchmark_task2_safeguider.sh" all
}

eval_llama() {
    LLAMA_TEST="${GUARDCHAT_TEST}" \
    bash "${SCRIPT_DIR}/benchmark_task2_llama.sh" all
}

eval_gemini() {
    GEMINI_TEST="${GUARDCHAT_TEST}" \
    bash "${SCRIPT_DIR}/benchmark_task2_gemini.sh" all
}

for tgt in "${TARGETS[@]}"; do
    case "${tgt}" in
        safeguider) eval_safeguider ;;
        llama)      eval_llama ;;
        gemini)     eval_gemini ;;
        all)
            eval_safeguider
            eval_llama
            eval_gemini
            ;;
        *)
            echo "Unknown target: ${tgt}" >&2
            echo "Choose from: safeguider | llama | gemini | all" >&2
            exit 2
            ;;
    esac
done

section "Done"
echo "Rewrites saved under ${REPO_ROOT}/experiment_results/task2/"
ls -1 "${REPO_ROOT}"/experiment_results/task2/*/*_task2_*.json 2>/dev/null || true
