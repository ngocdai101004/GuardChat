#!/usr/bin/env bash
# Run Task 1 (Multi-Label Unsafe Text Recognition) benchmark across baselines.
#
# Usage:
#   bash scripts/benchmark_task1.sh                          # all 6 baselines
#   bash scripts/benchmark_task1.sh bilstm bert              # subset
#   bash scripts/benchmark_task1.sh llamaguard qwen          # zero-shot only
#
# Targets:
#   bilstm       (supervised, needs trained weights)
#   bert         (supervised, needs trained weights)
#   safeguider   (supervised, needs trained weights)
#   llamaguard   (zero-shot, meta-llama/Llama-Guard-3-8B; needs HF_TOKEN)
#   qwen         (zero-shot, needs Qwen2.5-7B-Instruct local snapshot)
#   shieldgemma  (zero-shot, google/shieldgemma-2b; needs HF_TOKEN)
#   all          shorthand for every target above
#
# Inputs (override via env):
#   GUARDCHAT_TEST         default: multimedia-synergy-lab/GuardChat (HF)
#   GUARDCHAT_TEST_SPLIT   default: test
#   TEXT_KIND              default: both (single + conversation)
#
# Outputs:
#   ${RESULTS_DIR}/{bilstm,bert,safeguider,qwen}_task1.json
#   experiment_results/task1/{llamaguard,shieldgemma}/<model>_task1_{prompt,
#       raw_prompt,conversation}.json   (see the per-model scripts)
#
# Each output JSON has:
#   { "single":       { "metrics": {...}, "predictions": [...] },
#     "conversation": { "metrics": {...}, "predictions": [...] } }

set -euo pipefail

source "$(dirname "$0")/env.sh"

TARGETS=("$@")
if [[ ${#TARGETS[@]} -eq 0 ]]; then
    TARGETS=(bilstm bert safeguider llamaguard qwen shieldgemma)
fi

require_data "GUARDCHAT_TEST" "${GUARDCHAT_TEST}"

# Optional - some baselines accept extra device / dtype flags via env.
LLM_DEVICE_FLAG=()
[[ -n "${DEVICE:-}" ]] && LLM_DEVICE_FLAG=(--device "${DEVICE}")

eval_bilstm() {
    section "Eval BiLSTM (Task 1)"
    require_path "BILSTM_WEIGHTS" "${BILSTM_WEIGHTS}"
    run_module src.BiLSTM.eval_recognition \
        --test "${GUARDCHAT_TEST}" \
        --split "${GUARDCHAT_TEST_SPLIT}" \
        --weights "${BILSTM_WEIGHTS}" \
        --text-kind "${TEXT_KIND}" \
        --output "${RESULTS_DIR}/bilstm_task1.json"
}

eval_bert() {
    section "Eval BERT (Task 1)"
    require_path "BERT_WEIGHTS" "${BERT_WEIGHTS}"
    run_module src.BERT.eval_recognition \
        --test "${GUARDCHAT_TEST}" \
        --split "${GUARDCHAT_TEST_SPLIT}" \
        --weights "${BERT_WEIGHTS}" \
        --text-kind "${TEXT_KIND}" \
        --output "${RESULTS_DIR}/bert_task1.json"
}

eval_safeguider() {
    section "Eval SafeGuider (Task 1)"
    require_path "SAFEGUIDER_RECOG_WEIGHTS" "${SAFEGUIDER_RECOG_WEIGHTS}"
    run_module src.SafeGuider.eval_recognition \
        --test "${GUARDCHAT_TEST}" \
        --split "${GUARDCHAT_TEST_SPLIT}" \
        --weights "${SAFEGUIDER_RECOG_WEIGHTS}" \
        --text-kind "${TEXT_KIND}" \
        --output "${RESULTS_DIR}/safeguider_task1.json"
}

eval_llamaguard() {
    section "Eval Llama-Guard-3-8B (Task 1)"
    # Delegated: LlamaGuard writes three files (prompt / raw_prompt /
    # conversation) under experiment_results/task1/llamaguard/ rather
    # than the single ${RESULTS_DIR}/*_task1.json of the supervised rows.
    bash "${SCRIPT_DIR}/benchmark_task1_llamaguard.sh" all
}

eval_qwen() {
    section "Eval Qwen2.5-7B-Instruct (Task 1, zero-shot)"
    require_path "QWEN_WEIGHTS" "${QWEN_WEIGHTS}"
    run_module src.Qwen.eval_recognition \
        --test "${GUARDCHAT_TEST}" \
        --split "${GUARDCHAT_TEST_SPLIT}" \
        --weights "${QWEN_WEIGHTS}" \
        --dtype "${DTYPE}" \
        --text-kind "${TEXT_KIND}" \
        "${LLM_DEVICE_FLAG[@]}" \
        --output "${RESULTS_DIR}/qwen_task1.json"
}

eval_shieldgemma() {
    section "Eval ShieldGemma-2B (Task 1, zero-shot)"
    # Delegated: ShieldGemma writes three files (prompt / raw_prompt /
    # conversation) under experiment_results/task1/shieldgemma/ rather
    # than the single ${RESULTS_DIR}/*_task1.json of the other baselines.
    bash "${SCRIPT_DIR}/benchmark_task1_shieldgemma.sh" all
}

for tgt in "${TARGETS[@]}"; do
    case "${tgt}" in
        bilstm)      eval_bilstm ;;
        bert)        eval_bert ;;
        safeguider)  eval_safeguider ;;
        llamaguard)  eval_llamaguard ;;
        qwen)        eval_qwen ;;
        shieldgemma) eval_shieldgemma ;;
        all)
            eval_bilstm
            eval_bert
            eval_safeguider
            eval_llamaguard
            eval_qwen
            eval_shieldgemma
            ;;
        *)
            echo "Unknown target: ${tgt}" >&2
            echo "Choose from: bilstm | bert | safeguider | llamaguard | qwen | shieldgemma | all" >&2
            exit 2
            ;;
    esac
done

section "Done"
echo "Results saved under ${RESULTS_DIR}/"
ls -1 "${RESULTS_DIR}"/*_task1.json 2>/dev/null || true
