#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEFAULT_OUTPUT="$(cd "$SCRIPT_DIR/.." && pwd)/fitter_artifact_overleaf.zip"
OUTPUT_PATH="${1:-$DEFAULT_OUTPUT}"

"$PYTHON_BIN" - "$SCRIPT_DIR" "$OUTPUT_PATH" <<'PY'
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile

source = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
excluded_directories = {".git", ".venv", "outputs", "__pycache__"}

output.parent.mkdir(parents=True, exist_ok=True)
with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(part in excluded_directories for part in relative.parts):
            continue
        if path.is_dir() or path.suffix == ".pyc" or path.resolve() == output:
            continue
        archive.write(path, Path(source.name) / relative)

print(output)
PY
