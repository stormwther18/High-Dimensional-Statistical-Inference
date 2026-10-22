#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
PYTHON_BIN=${RIDGE_PYTHON:-python3}
"$PYTHON_BIN" Code/run_a2.py
"$PYTHON_BIN" Code/run_b_pilot.py
"$PYTHON_BIN" Code/analyze.py
