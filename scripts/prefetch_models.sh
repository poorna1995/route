#!/usr/bin/env bash
# Pre-download Phase A pool weights with classic HF hub (avoids flaky XET on RunPod).
set -euo pipefail
cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
source scripts/setup_env.sh

export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

if [[ -z "${HF_TOKEN:-}" ]] && [[ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo "Set HF_TOKEN before prefetch (gated Llama models)." >&2
  exit 1
fi

TOKEN_ARGS=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  TOKEN_ARGS=(--token "$HF_TOKEN")
fi

MODELS=(
  meta-llama/Llama-3.2-3B-Instruct
  meta-llama/Llama-3.1-8B-Instruct
)

for model in "${MODELS[@]}"; do
  echo "[prefetch] $model"
  if command -v hf >/dev/null 2>&1; then
    hf download "$model" "${TOKEN_ARGS[@]}"
  else
    huggingface-cli download "$model" "${TOKEN_ARGS[@]}" --resume-download
  fi
done

echo "[prefetch] done — cache at $HF_HOME"
