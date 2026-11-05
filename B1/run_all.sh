#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
python3 Code/test_core.py
python3 Code/run_experiment.py
python3 Code/validate_asymptotics.py
python3 Code/check_continuous_anchor.py
python3 Code/verify_results.py
python3 Code/analyze_results.py
python3 Code/build_report.py
