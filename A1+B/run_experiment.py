"""Run the pre-specified three-seed, two-covariance ridge stability experiment.

Requires Python >= 3.10 and NumPy. Run from any directory:
    python3 run_experiment.py
All generated files are confined to this script's directory. The prior A1
implementation is imported as a library; importing it never triggers its CLI.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SEEDS = (2026102201, 2026102202, 2026102203)
REPETITIONS, TEST_SIZE = 500, 100
ALPHA2, NOISE_VARIANCE = 1.0, 1.0


def source_path():
    for directory in (ROOT, *ROOT.parents):
        candidate = directory / "A1_ridge_reproduction.py"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Put A1_ridge_reproduction.py beside this script or in an ancestor directory")


def import_reference():
    path = source_path()
    spec = importlib.util.spec_from_file_location("a1_reference_library", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    rows = list(rows)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_group(master_seed, model_id, model, n, p, lambdas, theory, test, population, conditional):
    rows = []
    repetitions = len(test)
    for j, lam in enumerate(lambdas):
        row = dict(master_seed=master_seed, model_id=model_id, model=model,
                   n=n, p=p, gamma=p/n, alpha2=ALPHA2,
                   noise_variance=NOISE_VARIANCE, repetitions=repetitions,
                   test_size=TEST_SIZE, lambda_=float(lam),
                   sqrt_lambda=float(np.sqrt(lam)), theory=float(theory[j]))
        for key, values in (("test", test), ("population", population), ("conditional", conditional)):
            mean = float(values[:, j].mean())
            se = float(values[:, j].std(ddof=1)/np.sqrt(repetitions))
            row.update({key+"_mean": mean, key+"_se": se,
                        key+"_ci_low": mean-1.96*se, key+"_ci_high": mean+1.96*se})
        differences = test[:, j]-conditional[:, j]
        row["test_minus_conditional"] = float(differences.mean())
        row["test_minus_conditional_se"] = float(differences.std(ddof=1)/np.sqrt(repetitions))
        row["paired_difference_se"] = row["test_minus_conditional_se"]
        row["conditional_minus_theory"] = row["conditional_mean"]-row["theory"]
        row["test_minus_theory"] = row["test_mean"]-row["theory"]
        rows.append(row)
    return rows


def optima_row(rows):
    target = next(row for row in rows if row["lambda_"] == 1.0)
    observed = min(rows, key=lambda row: row["test_mean"])
    conditional_best = min(rows, key=lambda row: row["conditional_mean"])
    gap = target["test_minus_theory"]
    return dict(master_seed=target["master_seed"], model_id=target["model_id"],
                model=target["model"], n=target["n"], p=target["p"], gamma=target["gamma"],
                repetitions=target["repetitions"], test_size=target["test_size"],
                theory_optimal_lambda=1.0, theory_optimal_risk=target["theory"],
                test_mean_at_theory_optimum=target["test_mean"],
                test_se_at_theory_optimum=target["test_se"],
                test_ci_low_at_theory_optimum=target["test_ci_low"],
                test_ci_high_at_theory_optimum=target["test_ci_high"],
                test_minus_theory=gap, absolute_test_theory_gap=abs(gap),
                z_test_theory=gap/target["test_se"],
                conditional_mean_at_theory_optimum=target["conditional_mean"],
                conditional_se_at_theory_optimum=target["conditional_se"],
                population_mean_at_theory_optimum=target["population_mean"],
                population_se_at_theory_optimum=target["population_se"],
                test_minus_conditional=target["test_minus_conditional"],
                test_minus_conditional_se=target["test_minus_conditional_se"],
                paired_difference_se=target["test_minus_conditional_se"],
                conditional_minus_theory=target["conditional_minus_theory"],
                observed_optimal_lambda=observed["lambda_"],
                observed_optimal_test_mean=observed["test_mean"],
                observed_optimal_test_se=observed["test_se"],
                relative_deviation=abs(observed["lambda_"]-1.0),
                observed_minus_theory_optimum_test_mean=observed["test_mean"]-target["test_mean"],
                theory_risk_at_observed_optimum=observed["theory"],
                theory_regret_at_observed_optimum=observed["theory"]-target["theory"],
                conditional_optimal_lambda=conditional_best["lambda_"])


def main():
    start_clock = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    DATA.mkdir(parents=True, exist_ok=True)
    reference = import_reference()
    lambdas = np.unique(np.r_[np.linspace(0.1, np.sqrt(10), 61)**2, 1.0])
    assert len(lambdas) == 62
    models = ((1, "BinaryTree", 15, reference.binary_tree_spectrum(4)),
              (2, "Exponential", 20, reference.exponential_spectrum(20, rate=1.0)))
    write_csv(DATA/"lambda_grid.csv", [dict(lambda_index=i, lambda_=float(lam),
              sqrt_lambda=float(np.sqrt(lam)), is_theory_optimum=int(lam == 1.0))
              for i, lam in enumerate(lambdas)])
    write_csv(DATA/"population_spectra.csv", [dict(model_id=mid, model=name, n=n,
              p=len(t), gamma=len(t)/n, eigen_index=j, eigenvalue=float(value),
              mean_eigenvalue=float(t.mean()))
              for mid, name, n, t in models for j, value in enumerate(t)])
    raw_fields = ["master_seed", "model_id", "model", "n", "p", "gamma", "alpha2",
                  "noise_variance", "repetition", "test_size", "lambda_index",
                  "lambda_", "sqrt_lambda", "theory", "test_mse", "population_risk", "conditional_risk"]
    summaries, optima, all_values, stream_info = [], [], {}, []
    with (DATA/"raw_repetitions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=raw_fields)
        writer.writeheader()
        for master_seed in SEEDS:
            for model_id, model, n, t in models:
                group_clock = time.perf_counter()
                p = len(t)
                sequence = np.random.SeedSequence([master_seed, model_id])
                rng = np.random.default_rng(sequence)
                stream_info.append(dict(master_seed=master_seed, model_id=model_id,
                    entropy=[master_seed, model_id], bit_generator=type(rng.bit_generator).__name__,
                    spawn_key=list(sequence.spawn_key)))
                theory = reference.general_theoretical_risk(p/n, lambdas, t, ALPHA2)
                test = np.empty((REPETITIONS, len(lambdas)))
                population, conditional = np.empty_like(test), np.empty_like(test)
                for k in range(REPETITIONS):
                    x = rng.standard_normal((n, p))*np.sqrt(t)
                    s, basis, dual = reference.spectrum(x)
                    beta = rng.standard_normal(p)*np.sqrt(ALPHA2/p)
                    y = x@beta+rng.standard_normal(n)*np.sqrt(NOISE_VARIANCE)
                    fitted = reference.fit_from_spectrum(x, y, s, basis, dual, lambdas)
                    test_x = rng.standard_normal((TEST_SIZE, p))*np.sqrt(t)
                    test_y = test_x@beta+rng.standard_normal(TEST_SIZE)*np.sqrt(NOISE_VARIANCE)
                    test[k] = np.mean((test_y[:, None]-test_x@fitted)**2, axis=0)
                    population[k] = NOISE_VARIANCE+np.sum(t[:, None]*(fitted-beta[:, None])**2, axis=0)
                    weights, missing = reference.covariance_spectral_weights(x, s, basis, dual, t)
                    conditional[k] = reference.weighted_conditional_risk(s, weights, missing, n, p, lambdas, ALPHA2)
                    for j, lam in enumerate(lambdas):
                        writer.writerow(dict(master_seed=master_seed, model_id=model_id,
                            model=model, n=n, p=p, gamma=p/n, alpha2=ALPHA2,
                            noise_variance=NOISE_VARIANCE, repetition=k+1,
                            test_size=TEST_SIZE, lambda_index=j, lambda_=float(lam),
                            sqrt_lambda=float(np.sqrt(lam)), theory=float(theory[j]),
                            test_mse=float(test[k,j]), population_risk=float(population[k,j]),
                            conditional_risk=float(conditional[k,j])))
                rows = summarize_group(master_seed, model_id, model, n, p, lambdas, theory,
                                       test, population, conditional)
                summaries.extend(rows)
                optima.append(optima_row(rows))
                all_values.setdefault(model, []).append((test, population, conditional))
                print(json.dumps(dict(master_seed=master_seed, model=model,
                     elapsed_seconds=round(time.perf_counter()-group_clock, 3),
                     **{key: optima[-1][key] for key in ("test_mean_at_theory_optimum",
                     "test_se_at_theory_optimum", "observed_optimal_lambda",
                     "relative_deviation", "theory_regret_at_observed_optimum")})), flush=True)
    write_csv(DATA/"scan_summary.csv", summaries)
    write_csv(DATA/"optima_summary.csv", optima)
    pooled_summaries, pooled_optima = [], []
    for mid, model, n, t in models:
        test, population, conditional = [np.concatenate([group[j] for group in all_values[model]])
                                        for j in range(3)]
        theory = reference.general_theoretical_risk(len(t)/n, lambdas, t, ALPHA2)
        rows = summarize_group("pooled", mid, model, n, len(t), lambdas, theory, test, population, conditional)
        pooled_summaries.extend(rows)
        pooled_optima.append(optima_row(rows))
    write_csv(DATA/"pooled_scan_summary.csv", pooled_summaries)
    write_csv(DATA/"pooled_optima_summary.csv", pooled_optima)
    config = dict(started_at_utc=started, finished_at_utc=datetime.now(timezone.utc).isoformat(),
        execution_seconds=time.perf_counter()-start_clock, master_seeds=list(SEEDS),
        repetitions_per_seed_model=REPETITIONS, test_size=TEST_SIZE, alpha2=ALPHA2,
        noise_variance=NOISE_VARIANCE, gamma=1.0,
        models=[dict(model_id=mid, model=model, n=n, p=len(t),
                     depth=4 if model=="BinaryTree" else None,
                     exponential_rate=1.0 if model=="Exponential" else None,
                     mean_eigenvalue=float(t.mean())) for mid,model,n,t in models],
        lambda_grid_expression="np.unique(np.r_[np.linspace(.1,np.sqrt(10),61)**2,1.0])",
        lambda_count=len(lambdas), lambda_values=lambdas.tolist(), expected_raw_rows=186000,
        seed_sequence_expression="np.random.SeedSequence([master_seed, model_id])",
        random_streams=stream_info, common_random_numbers="Within each repetition all lambdas share X, beta, training noise, test X, and test noise.",
        training_test_independence="Training and testing rows and noises are independent; both use the same beta.",
        risk_definitions={"test_mse":"Mean squared error of the 100 independent test rows conditional on fitted beta.",
          "population_risk":"1 + (fitted_beta-beta)^T Sigma (fitted_beta-beta)",
          "conditional_risk":"Exact E[population_risk | X], integrating beta and training noise.",
          "theory":"General covariance deterministic equivalent using empirical population spectrum H_p."},
        interval_definition="Pointwise normal approximation mean +/- 1.96 SE; SE=sample sd/sqrt(R).",
        relative_deviation_definition="abs(observed_optimal_lambda-theory_optimal_lambda)/theory_optimal_lambda",
        theory_regret_definition="Deterministic R(observed_optimal_lambda)-R(1); descriptive, not a significance test.",
        z_definition="(test mean at lambda=1 - deterministic theory at lambda=1)/test SE; descriptive Monte Carlo scale, without model bias correction.",
        python_version=platform.python_version(), numpy_version=np.__version__,
        source_sha256={"run_experiment.py":sha256(__file__),
                       "verify_results.py":sha256(ROOT/"verify_results.py"),
                       "A1_ridge_reproduction.py":sha256(source_path())},
        output_sha256={path.name:sha256(path) for path in sorted(DATA.glob("*.csv"))})
    (ROOT/"config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"Completed {config['expected_raw_rows']} raw rows in {config['execution_seconds']:.3f} seconds", flush=True)


if __name__ == "__main__":
    main()
