#!/usr/bin/env bash
# Bootstrap Python env for llm_routing (local dev + RunPod).
#
# Usage:
#   source scripts/setup_env.sh          # activate env in current shell
#   bash scripts/setup_env.sh            # same, subshell-safe for runpod.sh
#   bash scripts/setup_env.sh --force    # reinstall even if marker matches
#
# Env overrides:
#   LLM_ROUTING_VENV          venv path (default: /workspace/.venv-llm-routing on RunPod, else .venv)
#   LLM_ROUTING_SKIP_ML=1     skip torch/transformers check (prepare-only stages)
#   HF_HOME                   HuggingFace cache root

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

_on_runpod() {
  [[ -n "${RUNPOD_POD_ID:-}" ]] || { [[ -d /workspace ]] && [[ -w /workspace ]]; }
}

if _on_runpod; then
  export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
  VENV="${LLM_ROUTING_VENV:-/workspace/.venv-llm-routing}"
  SYSTEM_SITE_PACKAGES=1
else
  export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
  VENV="${LLM_ROUTING_VENV:-$ROOT/.venv}"
  SYSTEM_SITE_PACKAGES=0
fi

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

MARKER="$VENV/.llm_routing_deps_hash"
DEPS_HASH="$(
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 pyproject.toml | awk '{print $1}'
  else
    md5sum pyproject.toml | awk '{print $1}'
  fi
)"

FORCE=0
for arg in "${@:-}"; do
  [[ "$arg" == "--force" ]] && FORCE=1
done

_venv_activate() {
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
}

_ensure_pip() {
  if ! python -m pip --version >/dev/null 2>&1; then
    python -m ensurepip --upgrade >/dev/null 2>&1 || true
  fi
  python -m pip --version >/dev/null 2>&1
}

_venv_broken() {
  [[ ! -d "$VENV/bin" ]] && return 0
  _venv_activate
  _has_core && return 1
  _ensure_pip || return 0
  return 1
}

_has_core() {
  python -c "import llm_routing, yaml, datasets" >/dev/null 2>&1
}

_has_ml() {
  python -c "import torch, transformers" >/dev/null 2>&1
}

_env_ready() {
  [[ -d "$VENV/bin" ]] || return 1
  [[ -f "$MARKER" ]] || return 1
  [[ "$(cat "$MARKER")" == "$DEPS_HASH" ]] || return 1
  _venv_activate
  _has_core || return 1
  if [[ "${LLM_ROUTING_SKIP_ML:-0}" != "1" ]]; then
    _has_ml || return 1
  fi
}

if [[ "$FORCE" != "1" ]] && _env_ready; then
  echo "[setup_env] Reusing $VENV"
  return 0 2>/dev/null || exit 0
fi

if [[ -d "$VENV" ]] && _venv_broken; then
  echo "[setup_env] Removing broken venv at $VENV"
  rm -rf "$VENV"
fi

echo "[setup_env] Provisioning $VENV"

if [[ ! -d "$VENV" ]]; then
  if [[ "$SYSTEM_SITE_PACKAGES" == "1" ]]; then
    echo "[setup_env] venv --system-site-packages (reuse image PyTorch)"
    python3 -m venv --system-site-packages "$VENV"
  else
    python3 -m venv "$VENV"
  fi
fi

_venv_activate
_ensure_pip || { echo "[setup_env] pip unavailable in $VENV" >&2; exit 1; }
if [[ "$SYSTEM_SITE_PACKAGES" == "1" ]] && _has_ml; then
  python -m pip install -q -U pip
else
  python -m pip install -q -U pip setuptools wheel
fi

# Core package (datasets, pyyaml, huggingface_hub) — always fast.
pip install -q -e .

if [[ "${LLM_ROUTING_SKIP_ML:-0}" == "1" ]]; then
  echo "[setup_env] ML skipped (LLM_ROUTING_SKIP_ML=1)"
else
  if _has_ml; then
    echo "[setup_env] torch present — installing ml extras only (no torch download)"
    pip install -q -e ".[ml]"
  else
    echo "[setup_env] torch missing — installing gpu + ml extras"
    pip install -q -e ".[gpu,ml]"
  fi
fi

echo "$DEPS_HASH" > "$MARKER"
echo "[setup_env] Ready: $VENV (HF_HOME=$HF_HOME)"
if [[ "${LLM_ROUTING_SKIP_ML:-0}" != "1" ]] && [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[setup_env] WARNING: HF_TOKEN not set — gated Llama models will fail at oracle" >&2
fi
