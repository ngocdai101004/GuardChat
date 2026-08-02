#!/usr/bin/env bash
# Run Task 2 (NSFW Concept Removal via Prompt Rewriting) with Llama-3.1-8B.
#
# Zero-shot: no training step. The weights are fetched once into
# src/Llama/weights/Llama-3.1-8B-Instruct/ (~16 GB) on the first run.
# meta-llama/Llama-3.1-8B-Instruct is GATED - accept the licence on the
# model page and provide HF_TOKEN (env or repo-root .env) first.
#
# Usage:
#   bash scripts/benchmark_task2_llama.sh                # both cases
#   bash scripts/benchmark_task2_llama.sh prompt         # one case
#   bash scripts/benchmark_task2_llama.sh prompt conversation
#
# Cases:
#   prompt        enhanced adversarial prompt  -> llama_task2_prompt.json
#   conversation  6-9 turn dialogue            -> llama_task2_conversation.json
#   all           shorthand for both
#
# Inputs (override via env):
#   HF_TOKEN            required (gated repo); env or repo-root .env
#   LLAMA_TEST          default: build_dataset/dataset/final_df_test.json
#   LLAMA_WEIGHTS       default: src/Llama/weights/Llama-3.1-8B-Instruct
#   LLAMA_OUT           default: experiment_results/task2/llama
#   LLAMA_BATCH_SIZE    default: 8    lower this first if you hit OOM
#   LLAMA_RETRIES       default: 3    attempts per sample; retries are sampled
#   DEVICE              default: auto  (auto | cuda | mps | cpu)
#   DTYPE_LLAMA         default: auto  (auto | bfloat16 | float16 | float32 | int8 | nf4)
#   LIMIT               optional cap on samples, for smoke tests
#   RESUME              set to 1 to reuse a .partial.jsonl checkpoint
#
# This step produces P_safe only. Safe Generation Rate and semantic
# similarity are computed downstream from the `rewritten_text` field -
# see src/Llama/README.md.

set -euo pipefail

source "$(dirname "$0")/env.sh"

LLAMA_TEST="${LLAMA_TEST:-${REPO_ROOT}/build_dataset/dataset/final_df_test.json}"
LLAMA_OUT="${LLAMA_OUT:-${REPO_ROOT}/experiment_results/task2/llama}"
LLAMA_BATCH_SIZE="${LLAMA_BATCH_SIZE:-8}"
LLAMA_RETRIES="${LLAMA_RETRIES:-3}"
DEVICE="${DEVICE:-auto}"
DTYPE_LLAMA="${DTYPE_LLAMA:-auto}"

TARGETS=("$@")
if [[ ${#TARGETS[@]} -eq 0 ]]; then
    TARGETS=(all)
fi

# Expand the shorthand and validate every requested case up front, so a
# typo fails before the 8B model is loaded.
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

require_data "LLAMA_TEST" "${LLAMA_TEST}"
mkdir -p "${LLAMA_OUT}"

EXTRA=()
[[ -n "${LIMIT:-}" ]] && EXTRA+=(--limit "${LIMIT}")
[[ "${RESUME:-0}" == "1" ]] && EXTRA+=(--resume)

# Both cases go through a single process so the 8B backbone is loaded
# once instead of twice.
if [[ ${#KINDS[@]} -eq 2 ]]; then
    KINDS=(all)
fi

for kind in "${KINDS[@]}"; do
    section "Rewrite with Llama-3.1-8B-Instruct (Task 2, zero-shot) - ${kind}"
    # ${EXTRA[@]+...} guards the expansion: under `set -u`, bash 3.2
    # (the system bash on macOS) treats an empty array as unset.
    run_module src.Llama.eval_rewrite \
        --test "${LLAMA_TEST}" \
        --weights "${LLAMA_WEIGHTS}" \
        --text-kind "${kind}" \
        --dtype "${DTYPE_LLAMA}" \
        --device "${DEVICE}" \
        --batch-size "${LLAMA_BATCH_SIZE}" \
        --retries "${LLAMA_RETRIES}" \
        --output-dir "${LLAMA_OUT}" \
        ${EXTRA[@]+"${EXTRA[@]}"}
done

section "Done"
echo "Rewrites saved under ${LLAMA_OUT}/"
ls -1 "${LLAMA_OUT}"/llama_task2_*.json 2>/dev/null || true
