#!/usr/bin/env bash
# Run Task 1 (Multi-Label Unsafe Text Recognition) with Llama-Guard-3-8B.
#
# Llama-Guard is zero-shot: no training step. The weights are fetched
# once into src/LlamaGuard/weights/Llama-Guard-3-8B/ (~16 GB) on the
# first run, provided HF_TOKEN can read the gated repo.
#
# Usage:
#   bash scripts/benchmark_task1_llamaguard.sh                 # all 3 cases
#   bash scripts/benchmark_task1_llamaguard.sh prompt          # one case
#   bash scripts/benchmark_task1_llamaguard.sh prompt raw_prompt
#
# Cases:
#   prompt        enhanced adversarial prompt      -> llamaguard_task1_prompt.json
#   raw_prompt    original seed prompt             -> llamaguard_task1_raw_prompt.json
#   conversation  multi-turn dialogue              -> llamaguard_task1_conversation.json
#   all           shorthand for the three above
#
# Inputs (override via env):
#   HF_TOKEN               HuggingFace token for the gated Meta licence
#                          (or put HF_TOKEN=... in the repo-root .env)
#   LLAMAGUARD_TEST        default: build_dataset/dataset/final_df_test.json
#   LLAMAGUARD_WEIGHTS     default: src/LlamaGuard/weights/Llama-Guard-3-8B
#   LLAMAGUARD_MODE        default: guardchat  ('guardchat' | 'native')
#   LLAMAGUARD_CONV_FORMAT default: concat     ('concat' | 'turns')
#   LLAMAGUARD_OUT         default: experiment_results/task1/llamaguard
#   DEVICE                 default: auto  (auto | cuda | mps | cpu)
#   DTYPE_LG               default: auto  (auto | bfloat16 | float16 | float32 | int8 | nf4)
#   LIMIT                  optional cap on samples, for smoke tests
#   RESUME                 set to 1 to reuse a .partial.jsonl checkpoint
#
# Metrics are recomputed downstream: every prediction stores the raw
# verdict and S-codes, so the taxonomy mapping can change without
# re-running the model.

set -euo pipefail

source "$(dirname "$0")/env.sh"

LLAMAGUARD_TEST="${LLAMAGUARD_TEST:-${REPO_ROOT}/build_dataset/dataset/final_df_test.json}"
LLAMAGUARD_MODE="${LLAMAGUARD_MODE:-guardchat}"
LLAMAGUARD_CONV_FORMAT="${LLAMAGUARD_CONV_FORMAT:-concat}"
LLAMAGUARD_OUT="${LLAMAGUARD_OUT:-${REPO_ROOT}/experiment_results/task1/llamaguard}"
DEVICE="${DEVICE:-auto}"
DTYPE_LG="${DTYPE_LG:-auto}"

TARGETS=("$@")
if [[ ${#TARGETS[@]} -eq 0 ]]; then
    TARGETS=(all)
fi

# Expand the shorthand and validate every requested case up front, so a
# typo fails before the 8B model is loaded.
KINDS=()
for tgt in "${TARGETS[@]}"; do
    case "${tgt}" in
        all)          KINDS+=(prompt raw_prompt conversation) ;;
        prompt|raw_prompt|conversation) KINDS+=("${tgt}") ;;
        *)
            echo "Unknown case: ${tgt}" >&2
            echo "Choose from: prompt | raw_prompt | conversation | all" >&2
            exit 2
            ;;
    esac
done

require_data "LLAMAGUARD_TEST" "${LLAMAGUARD_TEST}"
mkdir -p "${LLAMAGUARD_OUT}"

EXTRA=()
[[ -n "${LIMIT:-}" ]] && EXTRA+=(--limit "${LIMIT}")
[[ "${RESUME:-0}" == "1" ]] && EXTRA+=(--resume)

# All three cases go through a single process so the 8B backbone is
# loaded once instead of three times.
if [[ ${#KINDS[@]} -eq 3 ]]; then
    KINDS=(all)
fi

for kind in "${KINDS[@]}"; do
    section "Eval Llama-Guard-3-8B (Task 1, zero-shot) - ${kind}"
    run_module src.LlamaGuard.eval_recognition \
        --test "${LLAMAGUARD_TEST}" \
        --weights "${LLAMAGUARD_WEIGHTS}" \
        --mode "${LLAMAGUARD_MODE}" \
        --dtype "${DTYPE_LG}" \
        --device "${DEVICE}" \
        --text-kind "${kind}" \
        --conv-format "${LLAMAGUARD_CONV_FORMAT}" \
        --output-dir "${LLAMAGUARD_OUT}" \
        "${EXTRA[@]}"
done

section "Done"
echo "Results saved under ${LLAMAGUARD_OUT}/"
ls -1 "${LLAMAGUARD_OUT}"/llamaguard_task1_*.json 2>/dev/null || true
