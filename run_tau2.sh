#!/usr/bin/env bash
set -euo pipefail

TAU2_PYTHON="${TAU2_PYTHON:-/root/tau2-bench/.venv/bin/python}"

if [[ ! -x "$TAU2_PYTHON" ]]; then
  echo "TAU2_PYTHON is not executable: $TAU2_PYTHON" >&2
  exit 1
fi

exec "$TAU2_PYTHON" "$(dirname "$0")/run.py" "$@"
