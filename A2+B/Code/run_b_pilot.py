"""B pilot: choose the GCV-MA candidate lower bound using training-only validation."""
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
FLOORS = (-4., -3., -2., -1., 0.)
REPS = 100


def task(item):
    gamma, rep = item
    threadpool_limits(1)
    n, p = 100, round(100 * gamma)
    S, L = a2_core.population(p, "toeplitz")
    seed = np.random.SeedSequence([20261023, int(p), rep])
    rb, rx, re, rs, rf = [np.random.default_rng(s) for s in seed.spawn(5)]
    beta = rb.normal(size=p)
    beta /= np.sqrt(beta @ S @ beta)
    X = rx.normal(size=(n, p)) @ L.T
    y = a2_core.response(X, beta, re, "gaussian")
    order = rs.permutation(n)
    folds = np.array_split(order, 5)
    fit_order = rf.permutation(80)
    full_order = rs.permutation(n)
    records = []
    full = {}
    for lo in FLOORS:
        lambdas = np.logspace(lo, 2., 21)
        # Five shared folds provide both a five-fold rule and a single-split control.
        fold_losses = []
        for val_idx in folds:
            mask = np.ones(n, dtype=bool)
            mask[val_idx] = False
            train_idx = np.where(mask)[0]
            B_in, _, _, _, W_in, _, _, _, _ = a2_core.fit_feasible(
                X[train_idx], y[train_idx], lambdas, fit_order)
            fold_losses.extend((y[val_idx] - X[val_idx] @ (B_in @ W_in["GCV-MA"]))**2)
            if len(val_idx) and np.array_equal(val_idx, folds[-1]):
                val_loss = float(np.mean(fold_losses[-len(val_idx):]))
        cv5_loss = float(np.mean(fold_losses))
        B, E, df, _, W, _, _, _, _ = a2_core.fit_feasible(
            X, y, lambdas, full_order)
        bh = B @ W["GCV-MA"]
        excess = float((bh-beta) @ S @ (bh-beta))
        score = float(np.mean((E @ W["GCV-MA"])**2) / (1-df @ W["GCV-MA"]/n)**2)
        oracle_w, _ = a2_core.simplex_qp(np.sqrt(p)*(L.T@(B-beta[:, None])))
        oracle_b = B @ oracle_w
        oracle_excess = float((oracle_b-beta) @ S @ (oracle_b-beta))
        full[lo] = (excess, score, oracle_excess, B, E, df, W)
        records.append({"gamma": gamma, "rep": rep, "log10_lower": lo,
                        "validation_mse": val_loss, "cv5_mse": cv5_loss,
                        "gcv_ma_excess": excess,
                        "gcv_ma_score": score, "oracle_ma_excess": oracle_excess})
    chosen = min(records, key=lambda r: (r["validation_mse"], r["log10_lower"]))["log10_lower"]
    chosen_cv5 = min(records, key=lambda r: (r["cv5_mse"], r["log10_lower"]))["log10_lower"]
    B, E, df, W = full[-4.][3:]
    for method in ("Ridge-GCV", "LOO-MA"):
        bh = B @ W[method]
        ex = float((bh-beta) @ S @ (bh-beta))
        records.append({"gamma": gamma, "rep": rep, "log10_lower": -4.,
                        "method": method, "excess": ex})
    return records, {"gamma": gamma, "rep": rep, "chosen_log10_lower": chosen,
                     "selected_excess": full[chosen][0], "selected_gcv_score": full[chosen][1],
                     "selected_oracle_excess": full[chosen][2],
                     "validation_mse": next(r["validation_mse"] for r in records if r.get("log10_lower") == chosen and "validation_mse" in r),
                     "chosen_log10_lower_cv5": chosen_cv5,
                     "cv5_selected_excess": full[chosen_cv5][0],
                     "cv5_selected_gcv_score": full[chosen_cv5][1],
                     "cv5_selected_oracle_excess": full[chosen_cv5][2]}


def main():
    OUT.mkdir(exist_ok=True)
    tasks = [(gamma, r) for gamma in (1., 1.5) for r in range(REPS)]
    rows, selections = [], []
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i, (rr, ss) in enumerate(pool.map(task, tasks, chunksize=2), 1):
            rows.extend(rr)
            selections.append(ss)
            if i % 20 == 0:
                print(f"B pilot: {i}/{len(tasks)}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "B_pilot_raw.csv", index=False)
    pd.DataFrame(selections).to_csv(OUT / "B_pilot_selections.csv", index=False)
    metadata = {"seed": 20261023, "replications_per_gamma": REPS,
                "gammas": [1., 1.5], "log10_lower": list(FLOORS),
                "selection": "shared five-fold out-of-fold MSE (plus single 80/20 holdout control); refit selected grid on all 100 training observations",
                "python": sys.version, "platform": platform.platform(),
                "numpy": np.__version__, "pandas": pd.__version__,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "elapsed_sec": time.perf_counter()-start}
    (OUT / "B_pilot_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
