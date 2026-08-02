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
# Outputs:
#   gemini      experiment_results/task2/gemini/gemini_task2_{prompt,conversation}.json
#   safeguider  ${RESULTS_DIR}/safeguider_task2.json      (old single-file shape)
#   llama       ${RESULTS_DIR}/llama_task2.json           (old single-file shape)
#
# NOTE: only the Gemini branch has been ported to the two-representation
# schema (enhanced prompt + conversation, one file per representation,
# resumable). SafeGuider and Llama still write the older single-file
# `{summary, rewrites}` shape over the enhanced prompt only, and still
# compute CLIP similarity inline. Use scripts/benchmark_task2_gemini.sh
# directly for the current pipeline.
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

LLM_DEVICE_FLAG=()
[[ -n "${DEVICE:-}" ]] && LLM_DEVICE_FLAG=(--device "${DEVICE}")

eval_safeguider() {
    section "Eval SafeGuider beam-search rewriter (Task 2)"
    require_path "SAFEGUIDER_BINARY_WEIGHTS" "${SAFEGUIDER_BINARY_WEIGHTS}"
    run_module src.SafeGuider.eval_rewrite \
        --test "${GUARDCHAT_TEST}" \
        --split "${GUARDCHAT_TEST_SPLIT}" \
        --weights "${SAFEGUIDER_BINARY_WEIGHTS}" \
        "${LLM_DEVICE_FLAG[@]}" \
        --output "${RESULTS_DIR}/safeguider_task2.json"
}

eval_llama() {
    section "Eval Llama-3.1-8B-Instruct rewriter (Task 2)"
    require_path "LLAMA_WEIGHTS" "${LLAMA_WEIGHTS}"
    run_module src.Llama.eval_rewrite \
        --test "${GUARDCHAT_TEST}" \
        --split "${GUARDCHAT_TEST_SPLIT}" \
        --weights "${LLAMA_WEIGHTS}" \
        --dtype "${DTYPE}" \
        "${LLM_DEVICE_FLAG[@]}" \
        --output "${RESULTS_DIR}/llama_task2.json"
}

# Gemini has its own script - it rewrites both input representations
# and writes one file per representation under
# experiment_results/task2/gemini/, so it does not fit the single
# --output shape the other two still use. Delegate rather than
# duplicate the flag list.
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
echo "Results saved under ${RESULTS_DIR}/"
ls -1 "${RESULTS_DIR}"/*_task2.json 2>/dev/null || true
