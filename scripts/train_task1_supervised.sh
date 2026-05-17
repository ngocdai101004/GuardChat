#!/usr/bin/env bash
# Train every supervised Task-1 baseline (BiLSTM, BERT, SafeGuider).
#
# Usage:
#   bash scripts/train_task1_supervised.sh                 # train all 3
#   bash scripts/train_task1_supervised.sh bilstm          # train one
#   bash scripts/train_task1_supervised.sh bilstm bert     # train two
#
# Recipe (paper, Section 6.1):
#   14,000 samples (9,000 GuardChat conversational + 5,000 DiffusionDB safe)
#   AdamW lr=2e-5, weight_decay=0.01, batch 32, 10 epochs.
#
# Inputs (override via env):
#   GUARDCHAT_TRAIN         default: multimedia-synergy-lab/GuardChat (HF)
#   GUARDCHAT_TRAIN_SPLIT   default: train
#   DIFFUSIONDB_SAFE        default: data/diffusiondb_safe.json
#   TEXT_KIND               default: both (training itself uses "conversation")
# Outputs:
#   src/BiLSTM/weights/bilstm_multilabel.pt
#   src/BERT/weights/bert_multilabel/        (HuggingFace save_pretrained dir)
#   src/SafeGuider/weights/recognition_multilabel.pt

set -euo pipefail

source "$(dirname "$0")/env.sh"

TARGETS=("$@")
if [[ ${#TARGETS[@]} -eq 0 ]]; then
    TARGETS=(bilstm bert safeguider)
fi

# Common training hyperparameters - paper recipe.
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-2e-5}"
WD="${WD:-1e-2}"
SEED="${SEED:-111}"

require_data "GUARDCHAT_TRAIN" "${GUARDCHAT_TRAIN}"
if [[ ! -e "${DIFFUSIONDB_SAFE}" ]]; then
    echo "WARNING: ${DIFFUSIONDB_SAFE} not found - training without safe prompts." >&2
    SAFE_FLAG=()
else
    SAFE_FLAG=(--safe "${DIFFUSIONDB_SAFE}")
fi

train_bilstm() {
    section "Training BiLSTM (Task 1)"
    mkdir -p "$(dirname "${BILSTM_WEIGHTS}")"
    run_module src.BiLSTM.train_recognition \
        --train "${GUARDCHAT_TRAIN}" \
        --train-split "${GUARDCHAT_TRAIN_SPLIT}" \
        "${SAFE_FLAG[@]}" \
        --text-kind conversation \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --lr "${LR}" \
        --weight-decay "${WD}" \
        --seed "${SEED}" \
        --output "${BILSTM_WEIGHTS}" \
        --history-out "${RESULTS_DIR}/bilstm_train_history.json"
}

train_bert() {
    section "Training BERT (Task 1)"
    mkdir -p "${BERT_WEIGHTS}"
    run_module src.BERT.train_recognition \
        --train "${GUARDCHAT_TRAIN}" \
        --train-split "${GUARDCHAT_TRAIN_SPLIT}" \
        "${SAFE_FLAG[@]}" \
        --text-kind conversation \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --lr "${LR}" \
        --weight-decay "${WD}" \
        --seed "${SEED}" \
        --output "${BERT_WEIGHTS}" \
        --history-out "${RESULTS_DIR}/bert_train_history.json"
}

train_safeguider() {
    section "Training SafeGuider multi-label head (Task 1)"
    mkdir -p "$(dirname "${SAFEGUIDER_RECOG_WEIGHTS}")"
    run_module src.SafeGuider.train_recognition \
        --train "${GUARDCHAT_TRAIN}" \
        --train-split "${GUARDCHAT_TRAIN_SPLIT}" \
        "${SAFE_FLAG[@]}" \
        --text-kind conversation \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --lr "${LR}" \
        --weight-decay "${WD}" \
        --seed "${SEED}" \
        --output "${SAFEGUIDER_RECOG_WEIGHTS}" \
        --history-out "${RESULTS_DIR}/safeguider_train_history.json"
}

for tgt in "${TARGETS[@]}"; do
    case "${tgt}" in
        bilstm)     train_bilstm ;;
        bert)       train_bert ;;
        safeguider) train_safeguider ;;
        all)
            train_bilstm
            train_bert
            train_safeguider
            ;;
        *)
            echo "Unknown target: ${tgt}" >&2
            echo "Choose from: bilstm | bert | safeguider | all" >&2
            exit 2
            ;;
    esac
done

section "Done"
echo "Histories saved under ${RESULTS_DIR}/"
