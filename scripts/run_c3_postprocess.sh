#!/usr/bin/env bash
# C3 layerwise — CPU postprocess (merge, F7, RH5 analysis).
#
# Usage:
#   ./scripts/run_c3_postprocess.sh calib
#   ./scripts/run_c3_postprocess.sh test
#   ./scripts/run_c3_postprocess.sh all
#   ./scripts/run_c3_postprocess.sh calib --allow-no-features   # dev only
#
# Produces F7 + RH5 JSON for weak and strong models separately.
#
# Env: PY, CAMPAIGN, ORACLE_CALIB, ORACLE_TEST, FEATURES, COMPLEXITY
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/scripts:${PYTHONPATH:-}"

PY="${PY:-.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY=python
fi

CAMPAIGN="${CAMPAIGN:-experiments/campaigns/C3_llama_confidence_formation/M5}"
ORACLE_CALIB="${ORACLE_CALIB:-experiments/M4/routing_opportunity/llama_arc_validation.json}"
ORACLE_TEST="${ORACLE_TEST:-experiments/M4/routing_opportunity/llama_arc_full.json}"
FEATURES="${FEATURES:-experiments/M5/llama_arc_validation_query_features.csv}"
COMPLEXITY="${COMPLEXITY:-analysis/selected_feature.json}"

usage() {
  sed -n '2,11p' "$0" | sed 's/^# \?//'
  exit 1
}

ALLOW_NO_FEATURES=0
CMD=""
for arg in "$@"; do
  case "$arg" in
    --allow-no-features) ALLOW_NO_FEATURES=1 ;;
    calib|test|all) CMD="$arg" ;;
    help|-h|--help) usage ;;
    *) echo "unknown arg: $arg" >&2; usage ;;
  esac
done
[[ -z "$CMD" ]] && usage

require_features() {
  if [[ "$ALLOW_NO_FEATURES" -eq 1 ]]; then
    echo "warning: --allow-no-features — merge without c(q)" >&2
    return
  fi
  local missing=0
  for f in "$FEATURES" "$COMPLEXITY"; do
    if [[ ! -f "$f" ]]; then
      echo "missing required merge input: $f" >&2
      missing=1
    fi
  done
  if [[ "$missing" -eq 1 ]]; then
    echo "pass --allow-no-features to merge without c(q) (dev only)" >&2
    exit 1
  fi
}

_plot_and_rh5() {
  local role="$1"
  local pool="$2"
  local merged="$3"
  local trace="${CAMPAIGN}/layer_traces/${role}_${pool}.jsonl"
  local f7="paper/figures/F7_confidence_evolution_${role}_${pool}.png"
  local rh5="analysis/c3_rh5_${role}_${pool}.json"

  if [[ ! -f "$trace" ]]; then
    echo "missing trace: $trace" >&2
    exit 1
  fi

  echo "=== F7 (${role} ${pool}) ==="
  "$PY" scripts/run.py plot formation \
    --layer-trace "$trace" \
    --merged-csv "$merged" \
    --output "$f7"

  echo "=== RH5 analyze-formation (${role} ${pool}) ==="
  "$PY" scripts/run.py analyze-formation \
    --layer-trace "$trace" \
    --merged-csv "$merged" \
    --output "$rh5"

  echo "  $f7"
  echo "  $rh5"
}

run_role() {
  local role="${1:?calib|test}"
  local oracle
  case "$role" in
    calib) oracle="$ORACLE_CALIB" ;;
    test)  oracle="$ORACLE_TEST" ;;
    *) echo "unknown role: $role" >&2; exit 1 ;;
  esac

  local weak="${CAMPAIGN}/arc_${role}_weak_layerwise.csv"
  local strong="${CAMPAIGN}/arc_${role}_strong_layerwise.csv"
  local merged="analysis/c3_arc_${role}_merged.csv"
  local relevance="analysis/c3_arc_${role}_routing_relevance.json"

  for f in "$weak" "$strong" "$oracle"; do
    if [[ ! -f "$f" ]]; then
      echo "missing: $f" >&2
      exit 1
    fi
  done

  require_features

  echo "=== C3 merge (${role}) ==="
  merge_args=(
    --weak-csv "$weak"
    --strong-csv "$strong"
    --oracle "$oracle"
    --output "$relevance"
    --merged-csv "$merged"
  )
  if [[ "$ALLOW_NO_FEATURES" -eq 0 ]]; then
    merge_args+=(--features-csv "$FEATURES" --complexity-selection "$COMPLEXITY")
  fi
  "$PY" scripts/run.py merge "${merge_args[@]}"

  _plot_and_rh5 "$role" weak "$merged"
  _plot_and_rh5 "$role" strong "$merged"

  echo ""
  echo "Wrote: $merged"
  if [[ "$role" == "calib" ]]; then
    echo ""
    echo "DECISION GATE — Inspect RH5 (weak + strong F7 above)."
    echo "  If interpretable → continue TEST: ./scripts/run_c3_runpod.sh extract test all"
    echo "  Otherwise → stop and move C3 to future work (no redesign)."
  fi
}

case "$CMD" in
  calib) run_role calib ;;
  test)  run_role test ;;
  all)   run_role calib; run_role test ;;
  *) usage ;;
esac
