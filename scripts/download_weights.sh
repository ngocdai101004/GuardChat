#!/usr/bin/env bash
# Download model weights for every baseline that needs a local snapshot.
#
# Usage:
#   bash scripts/download_weights.sh                  # all baselines
#   bash scripts/download_weights.sh llamaguard qwen  # selected baselines
#   bash scripts/download_weights.sh --help
#
# Targets:
#   llamaguard  -> meta-llama/Llama-Guard-3-8B            (gated; needs HF_TOKEN)
#   llama       -> meta-llama/Llama-3.1-8B-Instruct        (gated; needs HF_TOKEN)
#   qwen        -> Qwen/Qwen2.5-7B-Instruct               (open access)
#   safeguider  -> openai/clip-vit-large-patch14          (open access; for the
#                                                          CLIP encoder used by
#                                                          src/SafeGuider/)
#
# Notes:
#   * BiLSTM / BERT / SafeGuider Task-1 head are TRAINED from scratch on
#     GuardChat (no pre-trained weights to download). Use
#     scripts/train_task1_supervised.sh.
#   * The SafeGuider Task-2 binary classifier (`SD1.4_safeguider.pt`) is NOT
#     on HuggingFace; obtain it from the upstream SafeGuider release and
#     place it at vendors/SafeGuider/weights/SD1.4_safeguider.pt.
#   * Gemini is API-only - nothing to download.

set -euo pipefail

source "$(dirname "$0")/env.sh"

print_help() {
    sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
    exit 0
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    print_help
fi

TARGETS=("$@")
if [[ ${#TARGETS[@]} -eq 0 ]]; then
    TARGETS=(llamaguard llama qwen safeguider)
fi

require_hf_token() {
    if [[ -z "${HF_TOKEN:-}" ]] && ! "${PYTHON}" -c "import huggingface_hub; tok=huggingface_hub.get_token(); raise SystemExit(0 if tok else 1)" 2>/dev/null; then
        cat >&2 <<EOF
ERROR: $1 is gated on HuggingFace and no token is available.
Please either:
  - run \`huggingface-cli login\` once (writes ~/.cache/huggingface/token), or
  - export HF_TOKEN=hf_xxxxxxxx in this shell.
And accept the licence at https://huggingface.co/$2
EOF
        return 1
    fi
}

download_llamaguard() {
    section "Downloading Llama-Guard-3-8B (gated)"
    require_hf_token "Llama-Guard-3-8B" "meta-llama/Llama-Guard-3-8B"
    run_module src.LlamaGuard.download_weights --local-dir "${LLAMAGUARD_WEIGHTS}"
}

download_llama() {
    section "Downloading Llama-3.1-8B-Instruct (gated)"
    require_hf_token "Llama-3.1-8B-Instruct" "meta-llama/Llama-3.1-8B-Instruct"
    run_module src.Llama.download_weights --local-dir "${LLAMA_WEIGHTS}"
}

download_qwen() {
    section "Downloading Qwen2.5-7B-Instruct"
    run_module src.Qwen.download_weights --local-dir "${QWEN_WEIGHTS}"
}

download_safeguider_clip() {
    section "Downloading openai/clip-vit-large-patch14 for SafeGuider"
    # The vendored SafeGuider auto-downloads the CLIP encoder on first use,
    # but we trigger it explicitly here so subsequent eval runs are offline.
    "${PYTHON}" - <<'PY'
import os, sys
sys.path.insert(0, os.path.join(os.environ["REPO_ROOT"], "vendors", "SafeGuider"))
from encoder import resolve_encoder_path  # noqa: E402
target = resolve_encoder_path("openai/clip-vit-large-patch14")
print(f"[safeguider] CLIP encoder ready at {target}")
PY

    if [[ ! -f "${SAFEGUIDER_BINARY_WEIGHTS}" ]]; then
        echo
        echo "WARNING: ${SAFEGUIDER_BINARY_WEIGHTS} not found." >&2
        echo "         The SafeGuider Task-2 rewriter requires this binary safety" >&2
        echo "         classifier checkpoint. Obtain SD1.4_safeguider.pt from the" >&2
        echo "         upstream SafeGuider release and place it at the path above." >&2
    fi
}

for tgt in "${TARGETS[@]}"; do
    case "${tgt}" in
        llamaguard) download_llamaguard ;;
        llama)      download_llama ;;
        qwen)       download_qwen ;;
        safeguider) download_safeguider_clip ;;
        all)
            download_llamaguard
            download_llama
            download_qwen
            download_safeguider_clip
            ;;
        *)
            echo "Unknown target: ${tgt}" >&2
            exit 2
            ;;
    esac
done

section "Done"
