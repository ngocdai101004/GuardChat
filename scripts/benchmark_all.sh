#!/usr/bin/env bash
# End-to-end pipeline: train supervised Task-1 baselines, then run both
# Task-1 and Task-2 benchmarks.
#
# Usage:
#   bash scripts/benchmark_all.sh
#   SKIP_TRAIN=1 bash scripts/benchmark_all.sh    # eval only (use existing weights)
#
# Note: this script assumes weights for the LLM baselines (Llama-Guard-3,
# Llama-3.1, Qwen2.5) and the SafeGuider binary classifier are already
# downloaded / placed locally. Run scripts/download_weights.sh first.

set -euo pipefail

source "$(dirname "$0")/env.sh"

if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
    bash "${SCRIPT_DIR}/train_task1_supervised.sh" all
else
    section "SKIP_TRAIN=1 -> skipping training; expecting existing weights"
fi

bash "${SCRIPT_DIR}/benchmark_task1.sh" all
bash "${SCRIPT_DIR}/benchmark_task2.sh" all

section "All done"
echo "Train histories + benchmark results under ${RESULTS_DIR}/"
ls -1 "${RESULTS_DIR}"/ 2>/dev/null || true
