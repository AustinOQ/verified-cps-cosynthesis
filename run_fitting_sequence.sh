#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ARCH_ROOT="${ARCHITECTURE_FIT_ROOT:-$SCRIPT_DIR/bundle/architecture-fit}"
DEFAULT_PY="$HOME/git_stuff/AI_venv/bin/python"

if [[ -z "${PYTHON_BIN:-}" && -x "$DEFAULT_PY" ]]; then
  PYTHON_BIN="$DEFAULT_PY"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/outputs/latest}"

if [[ ! -d "$ARCH_ROOT" ]]; then
  echo "Missing bundled architecture-fit tree: $ARCH_ROOT" >&2
  exit 1
fi

cd "$SCRIPT_DIR"
PYTHONDONTWRITEBYTECODE=1 ARCHITECTURE_FIT_ROOT="$ARCH_ROOT" \
  "$PYTHON_BIN" "$SCRIPT_DIR/src/run_fitting_sequence.py" --out-dir "$OUT_DIR" "$@"
