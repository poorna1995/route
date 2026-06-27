#!/usr/bin/env bash
# C3 layerwise probes — RunPod GPU runner (Llama 3.2 1B/3B).
#
# Workflow: Parity → Smoke → CALIB → (decision gate) → TEST
#
# One-time on pod:
#   cd /workspace/llm_routing
#   uv sync && uv pip install torch --index-url https://download.pytorch.org/whl/cu124
#   huggingface-cli login
#   chmod +x scripts/run_c3_runpod.sh scripts/run_c3_postprocess.sh
#
# Usage:
#   ./scripts/run_c3_runpod.sh parity              # Parity — 1B + 3B (n=10 each)
#   ./scripts/run_c3_runpod.sh smoke               # Smoke  — 10 weak + 10 strong
#   ./scripts/run_c3_runpod.sh extract calib all   # CALIB  — weak + strong
#   ./scripts/run_c3_runpod.sh extract test all    # TEST   — after decision gate
#
# Env: DEVICE, DTYPE, MARGIN_TOL, STAB_EPS, STAB_K, SMOKE_N, BATCH_SIZE, REPR_ONLY, CAMPAIGN, PY
#   REPR_ONLY=1  skip intermediate LM-head (Route B drift + terminal margin only)
#
# CPU postprocess: ./scripts/run_c3_postprocess.sh calib|test|all
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/scripts:${PYTHONPATH:-}"

PY="${PY:-.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY=python
fi

WEAK="${WEAK:-meta-llama/Llama-3.2-1B-Instruct}"
STRONG="${STRONG:-meta-llama/Llama-3.2-3B-Instruct}"
SPLITS="${SPLITS:-analysis/splits.json}"
CAMPAIGN="${CAMPAIGN:-experiments/campaigns/C3_llama_confidence_formation/M5}"

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
MARGIN_TOL="${MARGIN_TOL:-0.001}"
STAB_EPS="${STAB_EPS:-0.02}"
STAB_K="${STAB_K:-2}"
SMOKE_N="${SMOKE_N:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
REPR_ONLY="${REPR_ONLY:-0}"

SPLIT_ARGS=(--splits-json "$SPLITS")
GPU_ARGS=(--device "$DEVICE" --dtype "$DTYPE")

mkdir -p "$CAMPAIGN/layer_traces"

usage() {
  sed -n '2,18p' "$0" | sed 's/^# \?//'
  exit 1
}

_parity_one() {
  local model="$1"
  local label="$2"
  local n="$3"
  echo ""
  echo "=== Parity (${label}) n=${n} ==="
  "$PY" scripts/run.py layerwise-parity \
    --model "$model" \
    --dataset arc_challenge \
    "${SPLIT_ARGS[@]}" --split-role test \
    --limit "$n" --seed 42 \
    "${GPU_ARGS[@]}" \
    --margin-tol "$MARGIN_TOL"
}

run_parity() {
  local n="${1:-$SMOKE_N}"
  echo "=== C3 Parity (weak 1B → strong 3B) ==="
  _parity_one "$WEAK" "weak 1B" "$n"
  _parity_one "$STRONG" "strong 3B" "$n"
  echo ""
  echo "PASS: terminal lm_head(last_hidden_state) matches out.logits (both models)"
}

run_smoke() {
  local n="${1:-$SMOKE_N}"
  echo "=== C3 Smoke (weak + strong, n=${n} each) ==="
  run_extract test weak "$n"
  run_extract test strong "$n"
  echo ""
  echo "Next: inspect layer traces manually"
  echo "  ${CAMPAIGN}/layer_traces/test_weak.jsonl"
  echo "  ${CAMPAIGN}/layer_traces/test_strong.jsonl"
  echo "  - margins should evolve across depth_fraction (not flat / wild)"
  echo "  - compare terminal margin to C0 probe CSV if available"
  echo "Then: ./scripts/run_c3_runpod.sh extract calib all"
}

run_extract() {
  local role="${1:?role: calib|test}"
  local pool="${2:?pool: weak|strong|all}"
  local limit="${3:-}"

  local limit_args=()
  if [[ -n "$limit" ]]; then
    limit_args=(--limit "$limit")
  elif [[ -n "${LIMIT:-}" ]]; then
    limit_args=(--limit "$LIMIT")
  else
    # Full CALIB/TEST — manifest caps via splits.json (not smoke default 10)
    limit_args=(--limit 99999)
  fi

  local stage="CALIB"
  [[ "$role" == test ]] && stage="TEST"
  echo "=== C3 ${stage} extract ==="

  _one_pool() {
    local p="$1"
    local model="$WEAK"
    [[ "$p" == strong ]] && model="$STRONG"
    local out_csv="${CAMPAIGN}/arc_${role}_${p}_layerwise.csv"
    local trace="${CAMPAIGN}/layer_traces/${role}_${p}.jsonl"
    local repr_args=()
    if [[ "$REPR_ONLY" == "1" ]]; then
      repr_args=(--repr-only)
    fi
    echo "--- ${stage} ${p} → ${out_csv} (batch_size=${BATCH_SIZE} repr_only=${REPR_ONLY}) ---"
    "$PY" scripts/run.py probes \
      --model "$model" \
      --dataset arc_challenge \
      "${SPLIT_ARGS[@]}" --split-role "$role" \
      --layerwise --overwrite --batch-size "$BATCH_SIZE" \
      "${repr_args[@]}" \
      --layer-trace "$trace" \
      --stab-eps "$STAB_EPS" --stab-k "$STAB_K" \
      --margin-tol "$MARGIN_TOL" \
      "${GPU_ARGS[@]}" \
      "${limit_args[@]}" \
      --output "$out_csv"
  }

  case "$pool" in
    weak)   _one_pool weak ;;
    strong) _one_pool strong ;;
    all)    _one_pool weak; _one_pool strong ;;
    *)      echo "unknown pool: $pool" >&2; exit 1 ;;
  esac
}

CMD="${1:-}"
shift || usage

case "$CMD" in
  parity)   run_parity "${1:-}" ;;
  smoke)    run_smoke "${1:-}" ;;
  extract)  run_extract "$@" ;;
  help|-h|--help) usage ;;
  *)
    echo "Unknown command: $CMD" >&2
    usage
    ;;
esac
