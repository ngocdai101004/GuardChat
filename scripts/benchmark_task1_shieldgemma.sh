#!/usr/bin/env bash
# Run Task 1 (Multi-Label Unsafe Text Recognition) with ShieldGemma-2B.
#
# ShieldGemma is zero-shot: no training step. The weights are fetched
# once into src/ShieldGemma/weights/shieldgemma-2b/ (~5 GB) on the first
# run, provided HF_TOKEN can read the gated repo.
#
# Usage:
#   bash scripts/benchmark_task1_shieldgemma.sh                 # all 3 cases
#   bash scripts/benchmark_task1_shieldgemma.sh prompt          # one case
#   bash scripts/benchmark_task1_shieldgemma.sh prompt raw_prompt
#
# Cases:
#   prompt        enhanced adversarial prompt      -> shieldgemma_task1_prompt.json
#   raw_prompt    original seed prompt             -> shieldgemma_task1_raw_prompt.json
#   conversation  concatenated multi-turn dialogue -> shieldgemma_task1_conversation.json
#   all           shorthand for the three above
#
# Inputs (override via env):
#   HF_TOKEN               HuggingFace token for the gated Gemma licence
#                          (or put HF_TOKEN=... in the repo-root .env)
#   SHIELDGEMMA_TEST       default: build_dataset/dataset/final_df_test.json
#   SHIELDGEMMA_WEIGHTS    default: src/ShieldGemma/weights/shieldgemma-2b
#   SHIELDGEMMA_MODE       default: guardchat  ('guardchat' | 'native')
#   SHIELDGEMMA_OUT        default: experiment_results/task1/shieldgemma
#   SHIELDGEMMA_THRESHOLD  default: 0.5
#   SHIELDGEMMA_BATCH      default: 4
#   DEVICE                 default: auto  (auto | cuda | mps | cpu)
#   DTYPE_SG               default: auto  (auto | bfloat16 | float16 | float32 | int8 | nf4)
#   LIMIT                  optional cap on samples, for smoke tests
#   RESUME                 set to 1 to reuse a .partial.jsonl checkpoint
#
# Metrics are recomputed downstream: every prediction stores the raw
# per-policy P(Yes), so thresholds can change without re-running.

set -euo pipefail

source "$(dirname "$0")/env.sh"

SHIELDGEMMA_TEST="${SHIELDGEMMA_TEST:-${REPO_ROOT}/build_dataset/dataset/final_df_test.json}"
SHIELDGEMMA_WEIGHTS="${SHIELDGEMMA_WEIGHTS:-${REPO_ROOT}/src/ShieldGemma/weights/shieldgemma-2b}"
SHIELDGEMMA_MODE="${SHIELDGEMMA_MODE:-guardchat}"
SHIELDGEMMA_OUT="${SHIELDGEMMA_OUT:-${REPO_ROOT}/experiment_results/task1/shieldgemma}"
SHIELDGEMMA_THRESHOLD="${SHIELDGEMMA_THRESHOLD:-0.5}"
SHIELDGEMMA_BATCH="${SHIELDGEMMA_BATCH:-4}"
DEVICE="${DEVICE:-auto}"
DTYPE_SG="${DTYPE_SG:-auto}"

TARGETS=("$@")
if [[ ${#TARGETS[@]} -eq 0 ]]; then
    TARGETS=(all)
fi

# Expand the shorthand and validate every requested case up front, so a
# typo fails before the 2.6B model is loaded.
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

require_data "SHIELDGEMMA_TEST" "${SHIELDGEMMA_TEST}"
mkdir -p "${SHIELDGEMMA_OUT}"

EXTRA=()
[[ -n "${LIMIT:-}" ]] && EXTRA+=(--limit "${LIMIT}")
[[ "${RESUME:-0}" == "1" ]] && EXTRA+=(--resume)

# All three cases go through a single process so the 2.6B backbone is
# loaded once instead of three times.
if [[ ${#KINDS[@]} -eq 3 ]]; then
    KINDS=(all)
fi

for kind in "${KINDS[@]}"; do
    section "Eval ShieldGemma-2B (Task 1, zero-shot) - ${kind}"
    run_module src.ShieldGemma.eval_recognition \
        --test "${SHIELDGEMMA_TEST}" \
        --weights "${SHIELDGEMMA_WEIGHTS}" \
        --mode "${SHIELDGEMMA_MODE}" \
        --dtype "${DTYPE_SG}" \
        --device "${DEVICE}" \
        --text-kind "${kind}" \
        --threshold "${SHIELDGEMMA_THRESHOLD}" \
        --batch-size "${SHIELDGEMMA_BATCH}" \
        --output-dir "${SHIELDGEMMA_OUT}" \
        "${EXTRA[@]}"
done

section "Done"
echo "Results saved under ${SHIELDGEMMA_OUT}/"
ls -1 "${SHIELDGEMMA_OUT}"/shieldgemma_task1_*.json 2>/dev/null || true
