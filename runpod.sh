#!/usr/bin/env bash
# RunPod one-liner: bash runpod.sh [smoke|pilot|resume RUN_DIR]
#
# First run provisions /workspace/.venv-llm-routing + HF cache on the network volume.
# Subsequent runs reuse the env when pyproject.toml deps are unchanged (~seconds).
set -euo pipefail
cd "$(dirname "$0")"

# shellcheck disable=SC1091
source scripts/setup_env.sh

if [[ -z "${HF_TOKEN:-}" ]] && [[ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  echo "runpod.sh: set HF_TOKEN (Hugging Face read token) before oracle runs." >&2
  echo "  Accept Llama licenses on huggingface.co, then: export HF_TOKEN=hf_..." >&2
  exit 1
fi

export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

MODE="${1:-smoke}"
SETTING="experiments/candidates/arc.yaml"

case "$MODE" in
  prefetch)
    bash scripts/prefetch_models.sh
    ;;
  smoke)
    python run.py all --setting "$SETTING" --name arc-smoke --smoke
    ;;
  pilot)
    python run.py all --setting "$SETTING" --name arc-pilot
    ;;
  resume)
    python run.py resume --run "${2:?usage: runpod.sh resume experiments/runs/<id>}"
    ;;
  *)
    echo "usage: runpod.sh [smoke|pilot|prefetch|resume RUN_DIR]" >&2
    exit 1
    ;;
esac
