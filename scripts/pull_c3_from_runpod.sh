#!/usr/bin/env bash
# Pull C3 RunPod artifacts to this Mac checkout.
#
# Run on Mac (not on the pod):
#   chmod +x scripts/pull_c3_from_runpod.sh
#   ./scripts/pull_c3_from_runpod.sh
#
# Env (from RunPod Connect tab):
#   RUNPOD_SSH=978k42wvqrgvr8-64411d7a@ssh.runpod.io   # proxy (recommended)
#   RUNPOD_SSH=root@213.173.107.79  RUNPOD_PORT=46342 # direct TCP
#   RUNPOD_KEY=~/.ssh/id_ed25519
#   REMOTE=/workspace/llm_routing
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RUNPOD_SSH="${RUNPOD_SSH:-978k42wvqrgvr8-64411d7a@ssh.runpod.io}"
RUNPOD_PORT="${RUNPOD_PORT:-}"
RUNPOD_KEY="${RUNPOD_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE="${REMOTE:-/workspace/llm_routing}"
CAMPAIGN="${CAMPAIGN:-experiments/campaigns/C3_llama_confidence_formation/M5}"

SSH_OPTS=(-i "$RUNPOD_KEY" -o IdentitiesOnly=yes)
[[ -n "$RUNPOD_PORT" ]] && SSH_OPTS+=(-p "$RUNPOD_PORT")

mkdir -p "$CAMPAIGN/layer_traces" paper/figures analysis

pull() {
  local rel="$1"
  local dest="$2"
  echo "  $rel"
  rsync -avz -e "ssh ${SSH_OPTS[*]}" \
    "${RUNPOD_SSH}:${REMOTE}/${rel}" "$dest" 2>/dev/null || \
  scp "${SSH_OPTS[@]}" -r \
    "${RUNPOD_SSH}:${REMOTE}/${rel}" "$dest" 2>/dev/null || \
  echo "    (skip — not on pod yet)"
}

echo "=== C3 campaign ==="
pull "${CAMPAIGN}/" "./${CAMPAIGN}/"

echo "=== F7 figures ==="
for role in calib test; do
  for pool in weak strong; do
    pull "paper/figures/F7_confidence_evolution_${role}_${pool}.png" paper/figures/
  done
done

echo "=== C3 analysis ==="
for f in \
  c3_rh5_calib_weak.json c3_rh5_calib_strong.json \
  c3_rh5_test_weak.json c3_rh5_test_strong.json \
  c3_arc_calib_merged.csv c3_arc_test_merged.csv \
  c3_arc_calib_routing_relevance.json c3_arc_test_routing_relevance.json; do
  pull "analysis/${f}" analysis/
done

echo ""
echo "=== Local summary ==="
ls -lh "$CAMPAIGN"/*.csv 2>/dev/null || echo "(no campaign CSVs)"
ls -lh "$CAMPAIGN/layer_traces/"*.jsonl 2>/dev/null || echo "(no JSONL traces)"
ls -lh paper/figures/F7_confidence_evolution_*.png 2>/dev/null || echo "(no F7 PNGs)"
ls -lh analysis/c3_* 2>/dev/null || echo "(no c3 analysis files)"
echo "Done."
