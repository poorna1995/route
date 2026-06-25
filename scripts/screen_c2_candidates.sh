#!/usr/bin/env bash
# C2 screening for generalization candidates (Llama pool, n=50, seed=42).
# Requires GPU for reasonable runtime. Summaries → analysis/c2_*_summary.json
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
WEAK="meta-llama/Llama-3.2-1B-Instruct"
STRONG="meta-llama/Llama-3.2-3B-Instruct"
N=50
SEED=42
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"

run_c2() {
  local dataset="$1"
  local split="$2"
  local tag="$3"
  local question="$4"
  local out="experiments/M4/routing_opportunity/${tag}.json"
  echo "=== C2: ${dataset} (${tag}) ==="
  "$PY" scripts/run.py oracle \
    --weak "$WEAK" --strong "$STRONG" \
    --dataset "$dataset" --split "$split" \
    --limit "$N" --seed "$SEED" \
    --max-new-tokens 8 \
    --device "$DEVICE" --dtype "$DTYPE" \
    --output "$out"
  "$PY" scripts/run.py summarize-c2 \
    --oracle "$out" \
    --output "analysis/c2_${tag}_summary.json" \
    --scientific-question "$question"
}

run_c2 mmlu test llama_mmlu_n50 "Broad factual knowledge"
run_c2 boolq validation llama_boolq_n50 "Reading comprehension (yes/no)"

echo "Done. Review analysis/c2_llama_mmlu_n50_summary.json and analysis/c2_llama_boolq_n50_summary.json"
