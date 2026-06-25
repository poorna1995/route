#!/usr/bin/env bash
# Resume-friendly CALIB/TEST oracle in small CPU batches.
# Each invocation appends to the same JSON (--max-pending). Re-run until "Wrote" + full summary.
#
# Usage:
#   ./scripts/batch_oracle_cpu.sh calib          # next batch, CALIB (299)
#   ./scripts/batch_oracle_cpu.sh test           # next batch, TEST (1172)
#   BATCH=25 ./scripts/batch_oracle_cpu.sh calib # smaller batch
#
# Tips on CPU/MPS:
#   - Use --weak-only then --strong-only in separate runs if RAM is tight (set PASS=weak|strong).
#   - Expect ~30–90s per query per model on CPU.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ROLE="${1:-calib}"
BATCH="${BATCH:-50}"
PASS="${PASS:-both}"   # both | weak | strong
DEVICE="${DEVICE:-cpu}"
DTYPE="${DTYPE:-}"

PY=(.venv/bin/python scripts/run.py oracle
  --weak meta-llama/Llama-3.2-1B-Instruct
  --strong meta-llama/Llama-3.2-3B-Instruct
  --dataset arc_challenge
  --splits-json analysis/splits.json
  --split-role "$ROLE"
  --seed 42
  --max-new-tokens 8
  --device "$DEVICE"
  --max-pending "$BATCH"
)

if [[ -n "$DTYPE" ]]; then
  PY+=(--dtype "$DTYPE")
fi

case "$ROLE" in
  calib) OUT=experiments/M4/routing_opportunity/arc_validation_oracle.json ;;
  test)  OUT=experiments/M4/routing_opportunity/arc_test_oracle.json ;;
  *) echo "role must be calib or test"; exit 1 ;;
esac

PY+=(--output "$OUT")

case "$PASS" in
  weak)   PY+=(--weak-only) ;;
  strong) PY+=(--strong-only) ;;
  both)   ;;
  *) echo "PASS must be both|weak|strong"; exit 1 ;;
esac

echo "=== oracle batch: role=$ROLE pass=$PASS batch=$BATCH device=$DEVICE ==="
echo "=== output: $OUT ==="
"${PY[@]}"
