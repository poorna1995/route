#!/usr/bin/env bash
# RunPod one-liner: bash runpod.sh [smoke|pilot|resume RUN_DIR]
#
# First run provisions /workspace/.venv-llm-routing + HF cache on the network volume.
# Subsequent runs reuse the env when pyproject.toml deps are unchanged (~seconds).
set -euo pipefail
cd "$(dirname "$0")"

# shellcheck disable=SC1091
source scripts/setup_env.sh

MODE="${1:-smoke}"
SETTING="experiments/candidates/arc.yaml"

case "$MODE" in
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
    echo "usage: runpod.sh [smoke|pilot|resume RUN_DIR]" >&2
    exit 1
    ;;
esac
