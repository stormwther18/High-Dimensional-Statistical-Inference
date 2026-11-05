"""B2 formal experiment: training-only lower-bound selection for GCV-MA.

The data-generating parameters enter only the post-fit analytical risk evaluation.
All lower-bound choices use the observed training sample and shared outer folds.
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import a2_core


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Results"
FLOORS = (-4.0, -3.0, -2.0, -1.0, 0.0)
SEED = 20261105


def configurations():
    rows = []
    for n in (100, 200):
        for gamma in (0.95, 1.0, 1.05, 1.5):
            for cov in ("iid", "toeplitz"):
                for snr in (1.0, 5.0):
                    p = round(n * gamma)
                    rows.append({
                        "config_id": len(rows), "n": n, "p": p,
                        "gamma": p / n, "cov": cov, "snr": snr,
                    })
    return rows


def fit_gcv_ma(X, y, lambdas):
    """Exact GCV-MA fit used in A2, without computing unused inner-CV methods."""
    B, E, _, df, _, loo = a2_core.ridge_path(X, y, lambdas)
    a = 1.0 - df / len(y)
    if not np.all(a > 0):
        raise FloatingPointError("non-positive GCV denominator")
    w, diag = a2_core.simplex_qp(E / a)
    w = w / a
    w = w / w.sum()
    return B, E, df, loo, w, diag


def evaluate(B, E, df, w, beta, Sigma):
    bh = B @ w
    delta = bh - beta
    excess = float(delta @ Sigma @ delta)
    denom = 1.0 - float(df @ w) / len(E)
    gcv_score = float(np.mean((E @ w) ** 2) / denom**2)
    return excess, gcv_score


def task(item):
    cfg, rep = item
    threadpool_limits(1)
    n, p = cfg["n"], cfg["p"]
    Sigma, chol = a2_core.population(p, cfg["cov"])
    seed = np.random.SeedSequence([SEED, cfg["config_id"], rep])
    rb, rx, re, rs, rf = [np.random.default_rng(s) for s in seed.spawn(5)]

    beta = rb.normal(size=p)
    beta *= np.sqrt(cfg["snr"] / (beta @ Sigma @ beta))
    X = rx.normal(size=(n, p)) @ chol.T
    y = a2_core.response(X, beta, re, "gaussian")

    folds = np.array_split(rs.permutation(n), 5)
    fold_orders = [rf.permutation(n - len(val)) for val in folds]
    full = {}
    floor_rows = []
    for lo in FLOORS:
        lambdas = np.logspace(lo, 2.0, 21)
        fold_mse = []
        single_split_mse = None
        for fold_id, val_idx in enumerate(folds):
            mask = np.ones(n, dtype=bool)
            mask[val_idx] = False
            train_idx = np.flatnonzero(mask)
            B_in, _, _, _, w_in, _ = fit_gcv_ma(
                X[train_idx], y[train_idx], lambdas
            )
            squared_errors = (y[val_idx] - X[val_idx] @ (B_in @ w_in)) ** 2
            fold_mse.extend(squared_errors.tolist())
            if fold_id == 4:
                single_split_mse = float(np.mean(squared_errors))

        B, E, df, loo, w, diag = fit_gcv_ma(X, y, lambdas)
        excess, gcv_score = evaluate(B, E, df, w, beta, Sigma)
        oracle_w, oracle_diag = a2_core.simplex_qp(
            np.sqrt(p) * (chol.T @ (B - beta[:, None]))
        )
        oracle_excess, _ = evaluate(B, E, df, oracle_w, beta, Sigma)
        cv5_mse = float(np.mean(fold_mse))
        full[lo] = {
            "B": B, "E": E, "df": df, "loo": loo, "w": w,
            "excess": excess, "gcv_score": gcv_score,
            "oracle_excess": oracle_excess,
        }
        floor_rows.append({
            **cfg, "rep": rep, "log10_lower": lo,
            "cv5_mse": cv5_mse, "single_split_mse": single_split_mse,
            "gcv_ma_excess": excess, "gcv_ma_gcv_score": gcv_score,
            "oracle_ma_excess": oracle_excess,
            "gcv_qp_success": bool(diag["success"]),
            "gcv_qp_gap": float(diag["gap"]),
            "oracle_qp_success": bool(oracle_diag["success"]),
            "oracle_qp_gap": float(oracle_diag["gap"]),
        })

    chosen_cv5 = min(floor_rows, key=lambda r: (r["cv5_mse"], r["log10_lower"]))["log10_lower"]
    chosen_split = min(floor_rows, key=lambda r: (r["single_split_mse"], r["log10_lower"]))["log10_lower"]

    base = full[-4.0]
    a = 1.0 - base["df"] / n
    ridge_scores = np.mean((base["E"] / a) ** 2, axis=0)
    ridge_w = np.eye(len(a))[int(np.argmin(ridge_scores))]
    ridge_excess, ridge_gcv_score = evaluate(
        base["B"], base["E"], base["df"], ridge_w, beta, Sigma
    )
    loo_w, loo_diag = a2_core.simplex_qp(base["loo"])
    loo_excess, loo_gcv_score = evaluate(
        base["B"], base["E"], base["df"], loo_w, beta, Sigma
    )

    selected = {
        **cfg, "rep": rep,
        "chosen_log10_lower_cv5": chosen_cv5,
        "cv5_selected_excess": full[chosen_cv5]["excess"],
        "cv5_selected_gcv_score": full[chosen_cv5]["gcv_score"],
        "chosen_log10_lower_split": chosen_split,
        "split_selected_excess": full[chosen_split]["excess"],
        "split_selected_gcv_score": full[chosen_split]["gcv_score"],
        "full_grid_excess": full[-4.0]["excess"],
        "full_grid_gcv_score": full[-4.0]["gcv_score"],
        "fixed_one_excess": full[0.0]["excess"],
        "fixed_one_gcv_score": full[0.0]["gcv_score"],
        "ridge_gcv_excess": ridge_excess,
        "ridge_gcv_score": ridge_gcv_score,
        "loo_ma_excess": loo_excess,
        "loo_ma_gcv_score": loo_gcv_score,
        "loo_qp_success": bool(loo_diag["success"]),
        "loo_qp_gap": float(loo_diag["gap"]),
    }
    return floor_rows, selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reps", type=int, default=200)
    args = parser.parse_args()
    configs = configurations()
    if args.reps < 1:
        raise ValueError("--reps must be positive")
    OUT.mkdir(exist_ok=True)
    prefix = "" if args.reps == 200 else f"quick_{args.reps}_"
    raw_path = OUT / f"{prefix}B2_floor_raw.csv"
    selected_path = OUT / f"{prefix}B2_selected_raw.csv"
    for path in (raw_path, selected_path):
        if path.exists():
            path.unlink()

    tasks = [(cfg, rep) for cfg in configs for rep in range(args.reps)]
    started = time.perf_counter()
    raw_buffer, selected_buffer = [], []
    raw_header = selected_header = True
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (floor_rows, selected) in enumerate(pool.map(task, tasks, chunksize=2), 1):
            raw_buffer.extend(floor_rows)
            selected_buffer.append(selected)
            if i % args.reps == 0 or i == len(tasks):
                pd.DataFrame(raw_buffer).to_csv(
                    raw_path, mode="a", header=raw_header, index=False
                )
                pd.DataFrame(selected_buffer).to_csv(
                    selected_path, mode="a", header=selected_header, index=False
                )
                raw_buffer.clear()
                selected_buffer.clear()
                raw_header = selected_header = False
                print(
                    f"B2: {i}/{len(tasks)} repetitions; "
                    f"{time.perf_counter() - started:.1f}s",
                    flush=True,
                )

    metadata = {
        "seed": SEED,
        "scenarios": len(configs),
        "replications_per_scenario": args.reps,
        "total_independent_datasets": len(tasks),
        "log10_lower_candidates": list(FLOORS),
        "lambda_upper": 100.0,
        "lambda_count": 21,
        "selection": (
            "shared five-fold out-of-fold MSE; single 80/20 split control; "
            "selected grid refit on all training observations"
        ),
        "evaluation": "analytical conditional excess risk using held-out true beta and Sigma",
        "elapsed_sec": time.perf_counter() - started,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "core_sha256": hashlib.sha256((Path(__file__).parent / "a2_core.py").read_bytes()).hexdigest(),
        "configurations": configs,
    }
    (OUT / f"{prefix}B2_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
