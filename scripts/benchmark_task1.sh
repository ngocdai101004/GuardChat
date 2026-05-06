#!/usr/bin/env bash
# Run Task 1 (Multi-Label Unsafe Text Recognition) benchmark across baselines.
#
# Usage:
#   bash scripts/benchmark_task1.sh                          # all 5 baselines
#   bash scripts/benchmark_task1.sh bilstm bert              # subset
#   bash scripts/benchmark_task1.sh llamaguard qwen          # zero-shot only
#
# Targets:
#   bilstm      (supervised, needs trained weights)
#   bert        (supervised, needs trained weights)
#   safeguider  (supervised, needs trained weights)
#   llamaguard  (zero-shot, needs Llama-Guard-3-8B local snapshot)
#   qwen        (zero-shot, needs Qwen2.5-7B-Instruct local snapshot)
#   all         shorthand for bilstm bert safeguider llamaguard qwen
#
# Outputs:
#   ${RESULTS_DIR}/{bilstm,bert,safeguider,llamaguard,qwen}_task1.json
#
# Each output JSON has:
#   { "single":       { "metrics": {...}, "predictions": [...] },
#     "conversation": { "metrics": {...}, "predictions": [...] } }

set -euo pipefail

source "$(dirname "$0")/env.sh"

TARGETS=("$@")
if [[ ${#TARGETS[@]} -eq 0 ]]; then
    TARGETS=(bilstm bert safeguider llamaguard qwen)
fi

require_path "GUARDCHAT_TEST" "${GUARDCHAT_TEST}"

# Optional - some baselines accept extra device / dtype flags via env.
LLM_DEVICE_FLAG=()
[[ -n "${DEVICE:-}" ]] && LLM_DEVICE_FLAG=(--device "${DEVICE}")

eval_bilstm() {
    section "Eval BiLSTM (Task 1)"
    require_path "BILSTM_WEIGHTS" "${BILSTM_WEIGHTS}"
    run_module src.BiLSTM.eval_recognition \
        --test "${GUARDCHAT_TEST}" \
        --weights "${BILSTM_WEIGHTS}" \
        --text-kind "${TEXT_KIND}" \
        --output "${RESULTS_DIR}/bilstm_task1.json"
}

eval_bert() {
    section "Eval BERT (Task 1)"
    require_path "BERT_WEIGHTS" "${BERT_WEIGHTS}"
    run_module src.BERT.eval_recognition \
        --test "${GUARDCHAT_TEST}" \
        --weights "${BERT_WEIGHTS}" \
        --text-kind "${TEXT_KIND}" \
        --output "${RESULTS_DIR}/bert_task1.json"
}

eval_safeguider() {
    section "Eval SafeGuider (Task 1)"
    require_path "SAFEGUIDER_RECOG_WEIGHTS" "${SAFEGUIDER_RECOG_WEIGHTS}"
    run_module src.SafeGuider.eval_recognition \
        --test "${GUARDCHAT_TEST}" \
        --weights "${SAFEGUIDER_RECOG_WEIGHTS}" \
        --text-kind "${TEXT_KIND}" \
        --output "${RESULTS_DIR}/safeguider_task1.json"
}

eval_llamaguard() {
    section "Eval Llama-Guard-3-8B (Task 1, zero-shot)"
    require_path "LLAMAGUARD_WEIGHTS" "${LLAMAGUARD_WEIGHTS}"
    run_module src.LlamaGuard.eval_recognition \
        --test "${GUARDCHAT_TEST}" \
        --weights "${LLAMAGUARD_WEIGHTS}" \
        --mode "${LLAMAGUARD_MODE:-native}" \
        --dtype "${DTYPE}" \
        --text-kind "${TEXT_KIND}" \
        "${LLM_DEVICE_FLAG[@]}" \
        --output "${RESULTS_DIR}/llamaguard_task1.json"
}

eval_qwen() {
    section "Eval Qwen2.5-7B-Instruct (Task 1, zero-shot)"
    require_path "QWEN_WEIGHTS" "${QWEN_WEIGHTS}"
    run_module src.Qwen.eval_recognition \
        --test "${GUARDCHAT_TEST}" \
        --weights "${QWEN_WEIGHTS}" \
        --dtype "${DTYPE}" \
        --text-kind "${TEXT_KIND}" \
        "${LLM_DEVICE_FLAG[@]}" \
        --output "${RESULTS_DIR}/qwen_task1.json"
}

for tgt in "${TARGETS[@]}"; do
    case "${tgt}" in
        bilstm)     eval_bilstm ;;
        bert)       eval_bert ;;
        safeguider) eval_safeguider ;;
        llamaguard) eval_llamaguard ;;
        qwen)       eval_qwen ;;
        all)
            eval_bilstm
            eval_bert
            eval_safeguider
            eval_llamaguard
            eval_qwen
            ;;
        *)
            echo "Unknown target: ${tgt}" >&2
            echo "Choose from: bilstm | bert | safeguider | llamaguard | qwen | all" >&2
            exit 2
            ;;
    esac
done

section "Done"
echo "Results saved under ${RESULTS_DIR}/"
ls -1 "${RESULTS_DIR}"/*_task1.json 2>/dev/null || true
