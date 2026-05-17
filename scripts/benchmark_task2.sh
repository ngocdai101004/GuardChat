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
#   gemini      (Gemini 2.5 Flash API, needs GEMINI_API_KEY env var)
#   all         shorthand for safeguider llama gemini
#
# Inputs (override via env):
#   GUARDCHAT_TEST         default: multimedia-synergy-lab/GuardChat (HF)
#   GUARDCHAT_TEST_SPLIT   default: test
#
# Outputs:
#   ${RESULTS_DIR}/{safeguider,llama,gemini}_task2.json
#
# Each output JSON has:
#   { "summary": {...},
#     "rewrites": [ {original_prompt, rewritten_prompt, clip_similarity, ...}, ... ] }
#
# Safe Generation Rate (SGR) is NOT computed here - feed the
# `rewritten_prompt` field from these JSONs to FLUX.1 / Gemini Image /
# DALL-E 3 in a separate pipeline and compute SGR externally.

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

eval_gemini() {
    section "Eval Gemini 2.5 Flash rewriter (Task 2)"
    if [[ -z "${GEMINI_API_KEY:-}" && -z "${GOOGLE_API_KEY:-}" ]]; then
        echo "ERROR: Gemini requires GEMINI_API_KEY or GOOGLE_API_KEY in the env." >&2
        echo "       Get a key at https://aistudio.google.com/." >&2
        return 1
    fi
    run_module src.Gemini.eval_rewrite \
        --test "${GUARDCHAT_TEST}" \
        --split "${GUARDCHAT_TEST_SPLIT}" \
        --model "${GEMINI_MODEL:-gemini-2.5-flash}" \
        --output "${RESULTS_DIR}/gemini_task2.json"
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
