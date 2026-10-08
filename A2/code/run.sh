#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
RIDGE_PYTHON="${RIDGE_PYTHON:-python3}"
export MPLCONFIGDIR="$(pwd)/.mpl-cache"
"$RIDGE_PYTHON" code/experiment.py --workers "${RIDGE_WORKERS:-4}"
"$RIDGE_PYTHON" code/verify.py
"$RIDGE_PYTHON" code/analyze.py
"$RIDGE_PYTHON" code/build_report.py
mkdir -p .latex-build
xelatex -interaction=nonstopmode -halt-on-error -output-directory=.latex-build report.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=.latex-build report.tex
cp .latex-build/report.pdf report.pdf
