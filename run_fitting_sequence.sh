#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ARCH_ROOT="$SCRIPT_DIR/bundle/architecture-fit"
PYTHON_BIN="${PYTHON_BIN:-python3}"

OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/outputs/latest}"

if [[ ! -d "$ARCH_ROOT" ]]; then
  echo "Missing bundled architecture-fit tree: $ARCH_ROOT" >&2
  exit 1
fi

cd "$SCRIPT_DIR"
PYTHONDONTWRITEBYTECODE=1 \
  "$PYTHON_BIN" "$SCRIPT_DIR/src/run_fitting_sequence.py" --out-dir "$OUT_DIR" "$@"
