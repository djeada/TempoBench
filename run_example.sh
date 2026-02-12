#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

VENV_DIR="${VENV_DIR:-.venv}"
CONFIG_FILE="${CONFIG_FILE:-examples/unique_bench.yaml}"
OUT_DIR="${OUT_DIR:-artifacts_example}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 is required but was not found in PATH." >&2
  exit 1
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Error: config file not found: ${CONFIG_FILE}" >&2
  exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating virtual environment at ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

if ! python -c "import tembench" >/dev/null 2>&1; then
  echo "Installing TempoBench into ${VENV_DIR}"
  if ! python -m pip install --no-build-isolation -e .; then
    cat <<EOF >&2
Error: failed to install TempoBench dependencies.
If you are offline, run this once when online:
  python -m pip install -e .
EOF
    exit 1
  fi
fi

mkdir -p "${OUT_DIR}"

tembench run --config "${CONFIG_FILE}" --out-dir "${OUT_DIR}"
tembench summarize --runs "${OUT_DIR}/runs.jsonl" --out-csv "${OUT_DIR}/summary.csv"
tembench plot --summary "${OUT_DIR}/summary.csv" --out-html "${OUT_DIR}/runtime.html"
tembench report --summary "${OUT_DIR}/summary.csv" --output "${OUT_DIR}/report.html"

cat <<EOF

Example run complete.
Artifacts are in: ${OUT_DIR}
 - ${OUT_DIR}/runs.jsonl
 - ${OUT_DIR}/summary.csv
 - ${OUT_DIR}/runtime.html
 - ${OUT_DIR}/report.html
EOF
