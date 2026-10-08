# High-Dimensional Statistical Inference

Course project for **High-Dimensional Statistical Inference**.

Author: Xinyi Hong

## Projects

### A1: Prediction risk in high-dimensional ridge regression

A1 studies the prediction risk of ridge regression and reproduces theoretical and numerical results under identity, BinaryTree, and Exponential covariance models. The experiments examine regularization, the dimension-to-sample-size ratio, bias and variance, and strong-signal behavior.

The folder includes Python implementations, simulation CSV files, experiment configurations, numerical verification records, a LaTeX report, and the compiled PDF.

### A2: Ridge model averaging and method comparisons

A2 studies model averaging across multiple ridge penalties. It compares GCV, leave-one-out, and five-fold cross-validation methods with single-penalty ridge regression, equal-weight averaging, regularized averaging, minimum-norm regression, and oracle benchmarks. Oracle methods use privileged information and serve only as reference benchmarks.

The experiments examine prediction risk, robustness, sample size, penalty grids, averaging weights, boundary behavior, and runtime. The folder includes Python code, raw and summarized results, figures, numerical verification records, and the submitted PDF report.

## Repository structure

- `A1/` contains the A1 submission materials.
- `A2/code/` contains the A2 experiment, verification, analysis, and report-building scripts.
- `A2/results/` contains the A2 experiment configurations and recorded results.
- `A2/figures/` contains the A2 figures in PDF and PNG formats.
- `A2/report/` contains the A2 submitted PDF report.

## Running the code

A1 requires Python 3.10 or later and NumPy. Run the following commands from `A1/`:

```bash
python3 A1_ridge_reproduction.py --check-only
python3 A1_ridge_reproduction.py
```

The full experiment command updates the generated results and the experiment sections of the LaTeX report. See `A1/A1_使用说明.md` for experiment modes and configuration options.

For A2, install the dependencies and run the numerical pipeline from `A2/`:

```bash
python3 -m pip install -r code/requirements.txt
python3 code/experiment.py --workers 4
python3 code/verify.py
python3 code/analyze.py
```

These commands update the numerical results and figures. The submitted A2 folder includes the compiled report but does not include `report_template.tex`, which is required by `code/build_report.py` and the report-compilation steps in `code/run.sh`.

## Update log

### 2026-10-08: Update 1

- Uploaded the A1 and A2 submission folders as the initial course project release.
- Added the A1 ridge prediction-risk implementation, covariance-model reproduction, simulation outputs, verification records, and LaTeX and PDF report.
- Added the A2 ridge model-averaging experiments and comparisons, raw and summarized results, verification records, figures, and PDF report.
- Added this README to describe the course project, repository structure, and code execution steps.
