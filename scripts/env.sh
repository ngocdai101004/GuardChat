#!/usr/bin/env bash
# Common environment variables for every script in scripts/.
# Source this from other scripts: `source "$(dirname "$0")/env.sh"`.
#
# Override anything from the shell:
#   DATA_DIR=/mnt/data RESULTS_DIR=/mnt/results bash scripts/benchmark_task1.sh

set -euo pipefail

# Resolve repo root regardless of where the caller invoked the script from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------- User-tunable paths ----------------

# Where GuardChat splits + DiffusionDB safe prompts live. Expected layout:
#   ${DATA_DIR}/guardchat/train.jsonl
#   ${DATA_DIR}/guardchat/test.jsonl
#   ${DATA_DIR}/diffusiondb_safe.json
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"

GUARDCHAT_TRAIN="${GUARDCHAT_TRAIN:-${DATA_DIR}/guardchat/train.jsonl}"
GUARDCHAT_TEST="${GUARDCHAT_TEST:-${DATA_DIR}/guardchat/test.jsonl}"
DIFFUSIONDB_SAFE="${DIFFUSIONDB_SAFE:-${DATA_DIR}/diffusiondb_safe.json}"

# Where benchmark JSON outputs land.
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results}"

# Per-baseline weight paths. Override individually if you keep weights
# on a different volume.
SAFEGUIDER_RECOG_WEIGHTS="${SAFEGUIDER_RECOG_WEIGHTS:-${REPO_ROOT}/src/SafeGuider/weights/recognition_multilabel.pt}"
SAFEGUIDER_BINARY_WEIGHTS="${SAFEGUIDER_BINARY_WEIGHTS:-${REPO_ROOT}/vendors/SafeGuider/weights/SD1.4_safeguider.pt}"
BILSTM_WEIGHTS="${BILSTM_WEIGHTS:-${REPO_ROOT}/src/BiLSTM/weights/bilstm_multilabel.pt}"
BERT_WEIGHTS="${BERT_WEIGHTS:-${REPO_ROOT}/src/BERT/weights/bert_multilabel}"

LLAMAGUARD_WEIGHTS="${LLAMAGUARD_WEIGHTS:-${REPO_ROOT}/src/LlamaGuard/weights/Llama-Guard-3-8B}"
QWEN_WEIGHTS="${QWEN_WEIGHTS:-${REPO_ROOT}/src/Qwen/weights/Qwen2.5-7B-Instruct}"
LLAMA_WEIGHTS="${LLAMA_WEIGHTS:-${REPO_ROOT}/src/Llama/weights/Llama-3.1-8B-Instruct}"

# ---------------- Runtime defaults ----------------

# Python interpreter. Use a venv-local python by exporting PYTHON before
# sourcing, e.g. `PYTHON=$HOME/venv/bin/python bash scripts/...`.
PYTHON="${PYTHON:-python}"

# Forward to the LLM-based baselines. bf16 is the recommended default;
# pass `DTYPE=nf4` for memory-constrained GPUs (needs bitsandbytes).
DTYPE="${DTYPE:-bfloat16}"

# Convenience: which Task 1 representation to use during eval. The CLIs
# also accept "both" - we default to "both" so a single run produces
# both single-turn and multi-turn ASR rows.
TEXT_KIND="${TEXT_KIND:-both}"

mkdir -p "${RESULTS_DIR}"

# ---------------- Helpers ----------------

# Pretty section header.
section() {
    local title="$1"
    echo
    echo "================================================================"
    echo "  ${title}"
    echo "================================================================"
}

# Run a python module from the repo root so `from src.X import ...` works
# regardless of the user's current working directory.
run_module() {
    (cd "${REPO_ROOT}" && "${PYTHON}" -m "$@")
}

# Confirm a path exists; exit non-zero with a helpful message otherwise.
require_path() {
    local kind="$1"
    local path="$2"
    if [[ ! -e "${path}" ]]; then
        echo "ERROR: ${kind} not found at ${path}" >&2
        echo "Hint: edit scripts/env.sh or export ${kind^^} before running." >&2
        return 1
    fi
}

export REPO_ROOT SCRIPT_DIR DATA_DIR RESULTS_DIR
export GUARDCHAT_TRAIN GUARDCHAT_TEST DIFFUSIONDB_SAFE
export SAFEGUIDER_RECOG_WEIGHTS SAFEGUIDER_BINARY_WEIGHTS
export BILSTM_WEIGHTS BERT_WEIGHTS
export LLAMAGUARD_WEIGHTS QWEN_WEIGHTS LLAMA_WEIGHTS
export PYTHON DTYPE TEXT_KIND
