#!/usr/bin/env bash
# Run Task 1 (Multi-Label Unsafe Text Recognition) with Qwen3Guard-Gen-8B.
#
# Qwen3Guard is zero-shot: no training step. The weights are fetched once
# into src/Qwen3Guard/weights/Qwen3Guard-Gen-8B/ (~16 GB) on the first
# run. The repo is Apache-2.0 and ungated, so HF_TOKEN is optional.
#
# Usage:
#   bash scripts/benchmark_task1_qwen3guard.sh                 # all 3 cases
#   bash scripts/benchmark_task1_qwen3guard.sh prompt          # one case
#   bash scripts/benchmark_task1_qwen3guard.sh prompt raw_prompt
#
# Cases:
#   prompt        enhanced adversarial prompt      -> qwen3guard_task1_prompt.json
#   raw_prompt    original seed prompt             -> qwen3guard_task1_raw_prompt.json
#   conversation  multi-turn dialogue              -> qwen3guard_task1_conversation.json
#   all           shorthand for the three above
#
# Inputs (override via env):
#   HF_TOKEN                 optional; only raises the anonymous rate limit
#   QWEN3GUARD_TEST          default: build_dataset/dataset/final_df_test.json
#   QWEN3GUARD_WEIGHTS       default: src/Qwen3Guard/weights/Qwen3Guard-Gen-8B
#   QWEN3GUARD_MODE          default: guardchat  ('guardchat' | 'native')
#   QWEN3GUARD_CONV_FORMAT   default: concat     ('concat' | 'turns')
#   QWEN3GUARD_CONTROVERSIAL default: unsafe     ('unsafe' | 'safe')
#   QWEN3GUARD_OUT           default: experiment_results/task1/qwen3guard
#   DEVICE                   default: auto  (auto | cuda | mps | cpu)
#   DTYPE_Q3G                default: auto  (auto | bfloat16 | float16 | float32 | int8 | nf4)
#   LIMIT                    optional cap on samples, for smoke tests
#   RESUME                   set to 1 to reuse a .partial.jsonl checkpoint
#
# Metrics are recomputed downstream: every prediction stores the raw
# severity and category names, so both the taxonomy mapping and the
# Controversial reading can change without re-running the model.

set -euo pipefail

source "$(dirname "$0")/env.sh"

QWEN3GUARD_TEST="${QWEN3GUARD_TEST:-${REPO_ROOT}/build_dataset/dataset/final_df_test.json}"
QWEN3GUARD_MODE="${QWEN3GUARD_MODE:-guardchat}"
QWEN3GUARD_CONV_FORMAT="${QWEN3GUARD_CONV_FORMAT:-concat}"
QWEN3GUARD_CONTROVERSIAL="${QWEN3GUARD_CONTROVERSIAL:-unsafe}"
QWEN3GUARD_OUT="${QWEN3GUARD_OUT:-${REPO_ROOT}/experiment_results/task1/qwen3guard}"
DEVICE="${DEVICE:-auto}"
DTYPE_Q3G="${DTYPE_Q3G:-auto}"

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

require_data "QWEN3GUARD_TEST" "${QWEN3GUARD_TEST}"
mkdir -p "${QWEN3GUARD_OUT}"

EXTRA=()
[[ -n "${LIMIT:-}" ]] && EXTRA+=(--limit "${LIMIT}")
[[ "${RESUME:-0}" == "1" ]] && EXTRA+=(--resume)

# All three cases go through a single process so the 8B backbone is
# loaded once instead of three times.
if [[ ${#KINDS[@]} -eq 3 ]]; then
    KINDS=(all)
fi

for kind in "${KINDS[@]}"; do
    section "Eval Qwen3Guard-Gen-8B (Task 1, zero-shot) - ${kind}"
    run_module src.Qwen3Guard.eval_recognition \
        --test "${QWEN3GUARD_TEST}" \
        --weights "${QWEN3GUARD_WEIGHTS}" \
        --mode "${QWEN3GUARD_MODE}" \
        --controversial "${QWEN3GUARD_CONTROVERSIAL}" \
        --dtype "${DTYPE_Q3G}" \
        --device "${DEVICE}" \
        --text-kind "${kind}" \
        --conv-format "${QWEN3GUARD_CONV_FORMAT}" \
        --output-dir "${QWEN3GUARD_OUT}" \
        "${EXTRA[@]}"
done

section "Done"
echo "Results saved under ${QWEN3GUARD_OUT}/"
ls -1 "${QWEN3GUARD_OUT}"/qwen3guard_task1_*.json 2>/dev/null || true
