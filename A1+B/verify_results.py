"""Independently recompute summaries from the saved raw data and verify fits.

The summary generation functions in run_experiment.py are intentionally never
imported or called. No pre-existing A1 file is written by this script.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT/"data"


def load_csv(filename):
    with (DATA/filename).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    start = time.perf_counter()
    config = json.loads((ROOT/"config.json").read_text(encoding="utf-8"))
    paths = [directory/"A1_ridge_reproduction.py" for directory in (ROOT,*ROOT.parents)]
    path = next(path for path in paths if path.is_file())
    spec = importlib.util.spec_from_file_location("verify_a1_reference", path)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    assert fingerprint(path) == config["source_sha256"]["A1_ridge_reproduction.py"]
    for name in ("run_experiment.py", "verify_results.py"):
        assert fingerprint(ROOT/name) == config["source_sha256"][name], name
    for name, expected in config["output_sha256"].items():
        assert fingerprint(DATA/name) == expected, name
    grid = np.unique(np.r_[np.linspace(.1,np.sqrt(10),61)**2,1.0])
    stored_grid = load_csv("lambda_grid.csv")
    np.testing.assert_array_equal([float(row["lambda_"]) for row in stored_grid], grid)
    assert len(grid) == 62 and sum(float(row["lambda_"]) == 1.0 for row in stored_grid) == 1
    groups, rows_read = {}, 0
    seen = set()
    for row in load_csv("raw_repetitions.csv"):
        seed, mid = int(row["master_seed"]), int(row["model_id"])
        rep, j = int(row["repetition"]), int(row["lambda_index"])
        assert (seed,mid,rep,j) not in seen
        seen.add((seed,mid,rep,j))
        assert seed in config["master_seeds"] and mid in (1,2)
        assert 1 <= rep <= 500 and 0 <= j < 62
        assert float(row["lambda_"]) == grid[j]
        assert int(row["n"]) == int(row["p"]) == (15 if mid == 1 else 20)
        assert row["model"] == ("BinaryTree" if mid == 1 else "Exponential")
        assert float(row["gamma"]) == float(row["alpha2"]) == float(row["noise_variance"]) == 1.0
        assert int(row["test_size"]) == 100
        key = (seed,mid)
        if key not in groups:
            groups[key] = np.full((500,62,4), np.nan)
        groups[key][rep-1,j] = [float(row[name]) for name in
                               ("test_mse","population_risk","conditional_risk","theory")]
        rows_read += 1
    assert rows_read == 186000 and len(seen) == 186000 and len(groups) == 6
    for values in groups.values():
        assert np.isfinite(values).all() and (values > 0).all()

    # Build a fresh lookup from raw arrays, using independent moment operations.
    computed = {}
    all_groups = dict(groups)
    for mid in (1,2):
        all_groups[("pooled",mid)] = np.concatenate([groups[(seed,mid)] for seed in config["master_seeds"]])
    max_error = 0.0
    compared_summary_rows = 0
    summary_keys = set()
    for filename in ("scan_summary.csv","pooled_scan_summary.csv"):
        for stored in load_csv(filename):
            seed = stored["master_seed"] if stored["master_seed"] == "pooled" else int(stored["master_seed"])
            mid = int(stored["model_id"])
            j = int(np.flatnonzero(grid == float(stored["lambda_"]))[0])
            assert (seed,mid,j) not in summary_keys
            summary_keys.add((seed,mid,j))
            assert int(stored["n"]) == int(stored["p"]) == (15 if mid == 1 else 20)
            assert stored["model"] == ("BinaryTree" if mid == 1 else "Exponential")
            assert float(stored["gamma"]) == float(stored["alpha2"]) == float(stored["noise_variance"]) == 1.
            assert int(stored["test_size"]) == 100
            np.testing.assert_allclose(float(stored["sqrt_lambda"]),np.sqrt(grid[j]),rtol=2e-13)
            values = all_groups[(seed,mid)][:,j]
            count = len(values)
            expected = {"theory": float(values[0,3]), "repetitions":count}
            for index,label in enumerate(("test","population","conditional")):
                vector = values[:,index]
                mean = math.fsum(vector)/count
                se = math.sqrt(math.fsum((float(v)-mean)**2 for v in vector)/(count-1)/count)
                expected.update({label+"_mean":mean, label+"_se":se,
                     label+"_ci_low":mean-1.96*se,label+"_ci_high":mean+1.96*se})
            difference = values[:,0]-values[:,2]
            mean = math.fsum(difference)/count
            expected.update(test_minus_conditional=mean,
                test_minus_conditional_se=math.sqrt(math.fsum((float(v)-mean)**2 for v in difference)/(count-1)/count),
                conditional_minus_theory=expected["conditional_mean"]-expected["theory"],
                test_minus_theory=expected["test_mean"]-expected["theory"])
            expected["paired_difference_se"] = expected["test_minus_conditional_se"]
            for name, value in expected.items():
                max_error = max(max_error,abs(float(stored[name])-value))
                np.testing.assert_allclose(float(stored[name]),value,rtol=2e-13,atol=2e-13)
            computed[(seed,mid,j)] = expected
            compared_summary_rows += 1
    assert compared_summary_rows == 496
    assert summary_keys == {(seed,mid,j) for seed,mid in all_groups for j in range(62)}
    optima_compared = 0
    optima_keys = set()
    for filename in ("optima_summary.csv","pooled_optima_summary.csv"):
        for stored in load_csv(filename):
            seed = stored["master_seed"] if stored["master_seed"] == "pooled" else int(stored["master_seed"])
            mid = int(stored["model_id"])
            assert (seed,mid) not in optima_keys
            optima_keys.add((seed,mid))
            assert int(stored["n"]) == int(stored["p"]) == (15 if mid == 1 else 20)
            assert stored["model"] == ("BinaryTree" if mid == 1 else "Exponential")
            assert float(stored["gamma"]) == 1. and int(stored["test_size"]) == 100
            vectors = all_groups[(seed,mid)]
            assert int(stored["repetitions"]) == len(vectors)
            jstar = int(np.flatnonzero(grid == 1.)[0])
            jbest = int(np.argmin(np.mean(vectors[:,:,0],axis=0)))
            jcond = int(np.argmin(np.mean(vectors[:,:,2],axis=0)))
            target,best = computed[(seed,mid,jstar)],computed[(seed,mid,jbest)]
            expected = dict(theory_optimal_lambda=1., theory_optimal_risk=target["theory"],
                test_mean_at_theory_optimum=target["test_mean"], test_se_at_theory_optimum=target["test_se"],
                test_ci_low_at_theory_optimum=target["test_ci_low"], test_ci_high_at_theory_optimum=target["test_ci_high"],
                test_minus_theory=target["test_minus_theory"], absolute_test_theory_gap=abs(target["test_minus_theory"]),
                z_test_theory=target["test_minus_theory"]/target["test_se"],
                conditional_mean_at_theory_optimum=target["conditional_mean"],
                conditional_se_at_theory_optimum=target["conditional_se"],
                population_mean_at_theory_optimum=target["population_mean"],
                population_se_at_theory_optimum=target["population_se"],
                test_minus_conditional=target["test_minus_conditional"],
                test_minus_conditional_se=target["test_minus_conditional_se"],
                paired_difference_se=target["test_minus_conditional_se"],
                conditional_minus_theory=target["conditional_minus_theory"],
                observed_optimal_lambda=float(grid[jbest]), observed_optimal_test_mean=best["test_mean"],
                observed_optimal_test_se=best["test_se"], relative_deviation=abs(float(grid[jbest])-1.),
                observed_minus_theory_optimum_test_mean=best["test_mean"]-target["test_mean"],
                theory_risk_at_observed_optimum=best["theory"],
                theory_regret_at_observed_optimum=best["theory"]-target["theory"],
                conditional_optimal_lambda=float(grid[jcond]))
            for name,value in expected.items():
                max_error = max(max_error,abs(float(stored[name])-value))
                np.testing.assert_allclose(float(stored[name]),value,rtol=2e-13,atol=2e-13)
            assert float(stored["conditional_optimal_lambda"]) == 1.0
            optima_compared += 1
    assert optima_compared == 8
    assert optima_keys == set(all_groups)

    # Reconstruct author spectra without using their library constructors.
    direct_tree = np.asarray([2.]*8+[4.]*4+[8.]*2+[16.])
    direct_tree /= direct_tree.mean()
    direct_exp = -np.log((np.arange(20)+0.5)/20)
    for mid,t in ((1,direct_tree),(2,direct_exp)):
        actual = np.asarray([float(row["eigenvalue"]) for row in load_csv("population_spectra.csv") if int(row["model_id"]) == mid])
        np.testing.assert_array_equal(actual,t)
        for seed in config["master_seeds"]:
            theory = ref.general_theoretical_risk(1.,grid,t,1.)
            np.testing.assert_allclose(groups[(seed,mid)][:,:,3],np.broadcast_to(theory,(500,62)),rtol=2e-13,atol=2e-13)
        assert grid[int(np.argmin(theory))] == 1.0

    # Replay first repetition of each actual random stream and compare every
    # lambda to direct primal solves and matrix trace conditional risk.
    fit_errors, risk_errors, replay_errors = [], [], []
    for seed in config["master_seeds"]:
        for mid,t in ((1,direct_tree),(2,direct_exp)):
            n = p = len(t)
            rng = np.random.default_rng(np.random.SeedSequence([seed,mid]))
            x = rng.standard_normal((n,p))*np.sqrt(t)
            beta = rng.standard_normal(p)/np.sqrt(p)
            y = x@beta+rng.standard_normal(n)
            test_x = rng.standard_normal((100,p))*np.sqrt(t)
            test_y = test_x@beta+rng.standard_normal(100)
            s,q,dual = ref.spectrum(x)
            spectral_fit = ref.fit_from_spectrum(x,y,s,q,dual,grid)
            gram = x.T@x/n
            for j,lam in enumerate(grid):
                inverse = np.linalg.solve(gram+lam*np.eye(p),np.eye(p))
                direct_fit = np.linalg.solve(gram+lam*np.eye(p),x.T@y/n)
                direct_conditional = 1+lam*lam/p*np.dot(t,np.diag(inverse@inverse))+np.dot(t,np.diag(inverse@gram@inverse))/n
                direct_test = np.mean((test_y-test_x@direct_fit)**2)
                direct_population = 1+np.dot(t,(direct_fit-beta)**2)
                actual = groups[(seed,mid)][0,j,:3]
                expected = [direct_test,direct_population,direct_conditional]
                np.testing.assert_allclose(spectral_fit[:,j],direct_fit,rtol=2e-10,atol=2e-10)
                np.testing.assert_allclose(actual,expected,rtol=2e-10,atol=2e-10)
                fit_errors.append(float(np.max(np.abs(spectral_fit[:,j]-direct_fit))))
                risk_errors.append(float(abs(actual[2]-direct_conditional)))
                replay_errors.append(float(np.max(np.abs(actual-expected))))
    old_checks = ref.run_covariance_checks()
    result = dict(status="passed", verified_at_utc=datetime.now(timezone.utc).isoformat(),
        execution_seconds=time.perf_counter()-start, raw_rows=rows_read,
        unique_seed_model_repetition_lambda_combinations=len(seen), raw_groups=len(groups),
        repetitions_per_group=500, lambdas_per_group=62, independent_summary_rows_compared=compared_summary_rows,
        independent_optima_rows_compared=optima_compared, max_summary_absolute_difference=max_error,
        full_grid_first_repetition_replays=6, direct_matrix_fits_compared=372,
        max_fit_absolute_difference=max(fit_errors), max_conditional_risk_absolute_difference=max(risk_errors),
        max_raw_replay_absolute_difference=max(replay_errors), author_population_spectra="passed",
        source_and_output_sha256="passed", existing_covariance_checks=old_checks,
        interpretation="Verification confirms arithmetic, data completeness and implementation agreement; it does not establish an inferential test for an argmin selected on the same scan.")
    (ROOT/"verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
