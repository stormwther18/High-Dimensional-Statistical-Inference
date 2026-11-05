"""Summarize paired B-plan experiments and create publication-ready vector figures.

Run from anywhere: python Code/analyze_results.py --root /path/to/B_ridge_averaging
Only the standard library, NumPy and Matplotlib are required.  All risk estimates
are kept separate from the independently evaluated population risk.  In
particular, conditional prior risks are used only for the three oracle families
and the two single-ridge oracles, never as a risk target for data-selected weights.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


METHOD_LABELS = {
    "ridge_cont_oracle": "Continuous ridge oracle",
    "ridge_grid_oracle": "Grid ridge oracle",
    "oracle_convex": "Convex oracle",
    "oracle_signed_L3": r"Signed oracle ($\|\pi\|_1\leq3$)",
    "ridge_gcv": "Ridge-GCV",
    "gcv_ma": "Naive GCV-MA",
    "cross_gcv_ma": "Cross-residual MA",
    "cross_gcv_signed_L3": "Signed cross-residual MA",
    "loo_ma": "LOO-MA",
    "equal_ma": "Equal-weight MA",
    "minnorm": "Minimum norm",
    "bayes_oracle": "Generalized Bayes oracle",
    "gcv_aggregate": "Naive aggregate GCV",
    "gcv_cross": "Cross-residual estimate",
    "fixed_average": "Fixed ridge average",
    "single_lambda_1": r"Single ridge ($\lambda=1$)",
}
COLORS = {
    "ridge_cont_oracle": "#333333", "ridge_grid_oracle": "#999999",
    "oracle_convex": "#0072B2", "oracle_signed_L3": "#D55E00",
    "ridge_gcv": "#009E73", "gcv_ma": "#CC79A7",
    "cross_gcv_ma": "#0072B2", "loo_ma": "#E69F00",
    "cross_gcv_signed_L3": "#D55E00",
    "equal_ma": "#56B4E9", "minnorm": "#888888", "bayes_oracle": "#333333",
    "gcv_aggregate": "#CC79A7", "gcv_cross": "#0072B2",
}
INTEGER_FIELDS = {"n", "p", "rep", "seed"}
TEXT_FIELDS = {"config_id", "method", "model", "pair_id"}
CONFIG_FIELDS = ["config_id", "n", "p", "gamma", "a", "b", "delta"]
ORACLE_METHODS = {
    "ridge_cont_oracle", "ridge_grid_oracle", "oracle_convex", "oracle_signed_L3",
    "bayes_oracle",
}


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        raw = list(csv.DictReader(stream))
    rows = []
    for original in raw:
        row = {}
        for key, value in original.items():
            if key in TEXT_FIELDS:
                row[key] = value
            elif value in ("", None, "None"):
                row[key] = float("nan")
            elif key == "solver_success":
                row[key] = str(value).lower() in {"true", "1", "1.0"}
            elif key in INTEGER_FIELDS:
                row[key] = int(value)
            else:
                try:
                    row[key] = float(value)
                except ValueError:
                    row[key] = value
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def stats(values) -> tuple[int, float, float]:
    x = np.asarray(list(values), float)
    x = x[np.isfinite(x)]
    if not len(x):
        return 0, float("nan"), float("nan")
    return len(x), float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("nan")


def groups(rows, keys):
    output = defaultdict(list)
    for row in rows:
        output[tuple(row[k] for k in keys)].append(row)
    return output


def config_fields(row):
    return {key: row[key] for key in CONFIG_FIELDS}


def summarize_main(raw, destination: Path):
    summary, calibration = [], []
    for (config, method), subset in sorted(groups(raw, ["config_id", "method"]).items()):
        result = config_fields(subset[0]) | {"method": method}
        for field, label in [("risk", "risk"), ("excess", "excess"),
                             ("conditional_risk", "conditional_risk"),
                             ("risk_estimate", "estimate"), ("l1_norm", "l1_norm"),
                             ("selected_lambda", "selected_lambda")]:
            count, mean, se = stats(row.get(field, np.nan) for row in subset)
            result[label + "_count"] = count
            result[label + "_mean"] = mean
            result[label + "_se"] = se
        result["repetitions"] = len(subset)
        critical_minnorm = method == "minnorm" and subset[0]["p"] == subset[0]["n"]
        result["mean_uncertainty_valid"] = not critical_minnorm
        if critical_minnorm:
            result["risk_se"] = result["excess_se"] = float("nan")
            result["mean_interpretation"] = "Sample mean only; interpolation-boundary minimum-norm risk has no finite population expectation in this Gaussian model."
        result["solver_failures"] = sum(not row.get("solver_success", True) for row in subset)
        result["solver_gap_max"] = max((row.get("solver_gap", 0) for row in subset), default=0)
        summary.append(result)
        valid = [row for row in subset if np.isfinite(row.get("risk_estimate", np.nan))]
        if valid:
            risk = np.array([r["risk"] for r in valid])
            estimate = np.array([r["risk_estimate"] for r in valid])
            ratio = float(estimate.mean() / risk.mean())
            _, bias, se_bias = stats(estimate - risk)
            # Paired delta-method MC standard error of ratio of sample means.
            _, _, ratio_scaled_se = stats(estimate - ratio * risk)
            calibration.append(config_fields(subset[0]) | {
                "method": method, "repetitions": len(valid),
                "mean_estimate": float(estimate.mean()), "mean_risk": float(risk.mean()),
                "ratio_of_means": ratio, "ratio_mc_se": ratio_scaled_se / risk.mean(),
                "mean_paired_bias": bias, "paired_bias_mc_se": se_bias,
            })
    write_csv(destination / "main_summary.csv", summary)
    write_csv(destination / "calibration.csv", calibration)
    return summary, calibration


def summarize_pairs(raw, destination: Path):
    """Use identical training repetitions; never subtract unpaired group means."""
    indexed = {(r["config_id"], r["rep"], r["method"]): r for r in raw}
    paired, oracle = [], []
    methods = sorted({row["method"] for row in raw})
    config_sets = groups(raw, ["config_id"])
    for (config,), subset in sorted(config_sets.items()):
        cfg = config_fields(subset[0])
        reps = sorted({r["rep"] for r in subset})
        for reference in ["ridge_cont_oracle", "oracle_convex", "bayes_oracle",
                          "ridge_gcv", "gcv_ma", "cross_gcv_ma"]:
            for method in methods:
                if method == reference:
                    continue
                matched = [(indexed[(config, rep, method)], indexed[(config, rep, reference)])
                           for rep in reps if (config, rep, method) in indexed
                           and (config, rep, reference) in indexed]
                if not matched:
                    continue
                count, mean, se = stats(m["risk"] - r["risk"] for m, r in matched)
                if "minnorm" in {method, reference} and cfg["p"] == cfg["n"]:
                    se = float("nan")
                paired.append(cfg | {"method": method, "reference": reference,
                    "repetitions": count, "mean_risk_difference": mean, "mc_se": se,
                    "mc_lower_95": mean - 1.96 * se, "mc_upper_95": mean + 1.96 * se,
                    "negative_favors_method": True})
        for method in ["ridge_grid_oracle", "oracle_convex", "oracle_signed_L3"]:
            matched = [(indexed[(config, rep, method)], indexed[(config, rep, "ridge_cont_oracle")])
                       for rep in reps if (config, rep, method) in indexed
                       and (config, rep, "ridge_cont_oracle") in indexed]
            valid = [(m, r) for m, r in matched if np.isfinite(m["conditional_risk"])
                     and np.isfinite(r["conditional_risk"])]
            if not valid:
                continue
            gains = np.array([r["conditional_risk"] - m["conditional_risk"] for m, r in valid])
            relative = np.array([100 * (r["conditional_risk"] - m["conditional_risk"])
                                 / r["conditional_risk"] for m, r in valid])
            count, gain, se = stats(gains)
            _, pct, pct_se = stats(relative)
            oracle.append(cfg | {"method": method, "reference": "ridge_cont_oracle",
                "repetitions": count, "mean_conditional_gain": gain, "mc_se": se,
                "mc_lower_95": gain - 1.96 * se, "mc_upper_95": gain + 1.96 * se,
                "mean_per_rep_relative_gain_pct": pct, "relative_gain_mc_se": pct_se,
                "min_per_rep_gain": float(gains.min()), "max_per_rep_gain": float(gains.max()),
                "positive_favors_method": True})
    write_csv(destination / "paired_risk.csv", paired)
    write_csv(destination / "oracle_gains.csv", oracle)
    return paired, oracle


def configure_plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 10, "legend.fontsize": 8, "xtick.labelsize": 9,
        "ytick.labelsize": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": .18, "lines.linewidth": 1.35,
        "savefig.bbox": "tight", "savefig.pad_inches": .08,
    })
    return plt


def save_figure(fig, path):
    for extension in ["pdf", "svg", "png"]:
        fig.savefig(path.with_suffix("." + extension), dpi=160)


def panel_grid(plt, rows):
    av = sorted({r["a"] for r in rows})
    gv = sorted({r["gamma"] for r in rows})
    fig, axes = plt.subplots(len(av), len(gv), figsize=(10.2, 3.0 * len(av)), squeeze=False,
                             sharex=True)
    for i, a in enumerate(av):
        for j, gamma in enumerate(gv):
            axes[i, j].set_title(rf"$a={a:g}$, $p/n={gamma:g}$")
            axes[i, j].set_xticks([-.3, 0., .3])
            axes[i, j].set_xticklabels([r"$-0.3$", r"$0$", r"$+0.3$"])
            if i == len(av) - 1:
                axes[i, j].set_xlabel(r"Signal alignment $b-a$")
    return fig, axes, av, gv


def draw_main_figures(summary, calibration, oracle, destination: Path):
    plt = configure_plots()
    destination.mkdir(parents=True, exist_ok=True)
    fig, axes, av, gv = panel_grid(plt, oracle)
    for i, a in enumerate(av):
        for j, gamma in enumerate(gv):
            ax = axes[i, j]
            ax.axhline(0, color="#666666", linewidth=.7)
            for method in ["ridge_grid_oracle", "oracle_convex", "oracle_signed_L3"]:
                points = sorted([r for r in oracle if r["a"] == a and r["gamma"] == gamma
                                 and r["method"] == method], key=lambda r: r["delta"])
                ax.errorbar([r["delta"] for r in points], [r["mean_conditional_gain"] for r in points],
                    yerr=[r["mc_se"] for r in points], marker="o", markersize=4, capsize=2,
                    color=COLORS[method], label=METHOD_LABELS[method])
            if j == 0:
                ax.set_ylabel("Conditional risk gain\n(positive is better)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, -.01))
    fig.suptitle("Model-class gains over continuously tuned nonnegative ridge", y=1.01)
    fig.tight_layout(rect=(0, .055, 1, .98))
    save_figure(fig, destination / "oracle_gains")
    plt.close(fig)

    selected = ["ridge_gcv", "gcv_ma", "cross_gcv_ma", "cross_gcv_signed_L3",
                "loo_ma", "ridge_cont_oracle", "bayes_oracle"]
    fig, axes, av, gv = panel_grid(plt, summary)
    for i, a in enumerate(av):
        for j, gamma in enumerate(gv):
            ax = axes[i, j]
            for method in selected:
                points = sorted([r for r in summary if r["a"] == a and r["gamma"] == gamma
                                 and r["method"] == method], key=lambda r: r["delta"])
                if not points:
                    continue
                y = np.array([r["excess_mean"] for r in points])
                se = np.array([r["excess_se"] for r in points])
                if np.any(y - se <= 0):
                    raise ValueError("Nonpositive lower MC error bar on excess-risk log scale")
                ax.errorbar([r["delta"] for r in points], y, yerr=se, markersize=3.5,
                    marker="x" if method == "bayes_oracle" else "o", capsize=2,
                    color=COLORS[method], label=METHOD_LABELS[method],
                    linestyle=":" if method == "bayes_oracle" else
                              "--" if method == "ridge_cont_oracle" else "-",
                    alpha=.9)
            ax.set_yscale("log")
            if j == 0:
                ax.set_ylabel("Mean excess prediction risk\n(log scale)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, -.03))
    fig.suptitle("Realized population-risk evaluation on paired training data", y=1.01)
    fig.tight_layout(rect=(0, .115, 1, .98))
    save_figure(fig, destination / "selected_risk")
    plt.close(fig)

    if calibration:
        fig, axes, av, gv = panel_grid(plt, calibration)
        for i, a in enumerate(av):
            for j, gamma in enumerate(gv):
                ax = axes[i, j]
                ax.axhline(1, color="#555555", linestyle="--", linewidth=.9)
                for method in ["ridge_gcv", "gcv_ma", "cross_gcv_ma", "cross_gcv_signed_L3", "loo_ma"]:
                    points = sorted([r for r in calibration if r["a"] == a and r["gamma"] == gamma
                                     and r["method"] == method], key=lambda r: r["delta"])
                    if not points:
                        continue
                    ax.errorbar([r["delta"] for r in points], [r["ratio_of_means"] for r in points],
                        yerr=[r["ratio_mc_se"] for r in points], marker="o", markersize=4, capsize=2,
                        color=COLORS[method], label=METHOD_LABELS[method])
                if j == 0:
                    ax.set_ylabel("Mean estimate / mean total risk")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, -.01))
        fig.suptitle("Risk calibration after data-dependent tuning", y=1.01)
        fig.tight_layout(rect=(0, .085, 1, .98))
        save_figure(fig, destination / "calibration")
        plt.close(fig)


def summarize_tails(raw, destination: Path, figures: Path, make_plot=True):
    output = []
    for (config,), subset in sorted(groups([r for r in raw if r["method"] == "minnorm"], ["config_id"]).items()):
        x = np.array([r["risk"] for r in subset])
        output.append(config_fields(subset[0]) | {"method": "minnorm", "repetitions": len(x),
            "sample_mean": float(x.mean()), "median": float(np.median(x)),
            "p90": float(np.quantile(x, .9)), "p99": float(np.quantile(x, .99)),
            "maximum": float(x.max()), "critical_interpolation": subset[0]["p"] == subset[0]["n"],
            "quantile_method": "NumPy linear interpolation"})
    write_csv(destination / "tail_summary.csv", output)
    if output and make_plot:
        plt = configure_plots()
        fig, axes, av, gv = panel_grid(plt, output)
        for i, a in enumerate(av):
            for j, gamma in enumerate(gv):
                ax = axes[i, j]
                points = sorted([r for r in output if r["a"] == a and r["gamma"] == gamma],
                                key=lambda r: r["delta"])
                for field, color in [("median", "#0072B2"), ("p90", "#009E73"),
                                     ("p99", "#E69F00"), ("maximum", "#D55E00")]:
                    ax.plot([r["delta"] for r in points], [r[field] for r in points],
                            marker="o", markersize=4, label=field, color=color)
                ax.set_yscale("log")
                if j == 0:
                    ax.set_ylabel("Minimum-norm total risk\n(log scale)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, -.01))
        fig.suptitle("Minimum-norm tail behavior: descriptive sample quantiles", y=1.01)
        fig.tight_layout(rect=(0, .055, 1, .98))
        save_figure(fig, figures / "minnorm_tail")
        plt.close(fig)
    return output


def summarize_weights(path: Path, results: Path):
    """Stream the larger weight file rather than retaining all rows in memory."""
    if not path.exists():
        return
    accumulators = defaultdict(lambda: np.zeros(7))
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["config_id"], row["method"], float(row["lambda"]))
            w = float(row["weight"])
            d = float(row["residual_denominator"])
            accumulators[key] += [1, w, w * w, abs(w), abs(w) > 1e-6, w < -1e-6, d]
    output = []
    for (config, method, lam), x in sorted(accumulators.items()):
        n, total, square, absolute, active, negative, den = x
        variance = max(0., (square - total * total / n) / (n - 1)) if n > 1 else np.nan
        output.append({"config_id": config, "method": method, "lambda": lam,
            "repetitions": int(n), "mean_weight": total / n,
            "weight_mc_se": math.sqrt(variance / n), "mean_absolute_weight": absolute / n,
            "nonzero_frequency": active / n, "negative_frequency": negative / n,
            "mean_residual_denominator": den / n, "nonzero_tolerance": 1e-6})
    write_csv(results / "weights_summary.csv", output)


def summarize_mp(path: Path, results: Path, figures: Path):
    if not path.exists():
        return []
    raw = load_csv(path)
    output = []
    for (pair, n, method), subset in sorted(groups(raw, ["pair_id", "n", "method"]).items()):
        _, est, est_se = stats(r["estimate"] for r in subset)
        _, risk, risk_se = stats(r["risk"] for r in subset)
        _, bias, bias_se = stats(r["estimate"] - r["risk"] for r in subset)
        output.append({"pair_id": pair, "n": n, "method": method, "repetitions": len(subset),
            "estimate_mean": est, "estimate_se": est_se, "risk_mean": risk, "risk_se": risk_se,
            "paired_bias": bias, "paired_bias_se": bias_se,
            "limit_risk": subset[0]["limit_risk"], "limit_gcv": subset[0]["limit_gcv"]})
    write_csv(results / "mp_summary.csv", output)
    plt = configure_plots()
    pairs = sorted({r["pair_id"] for r in output})
    fig, axes = plt.subplots(1, len(pairs), figsize=(5.2 * len(pairs), 3.5), squeeze=False)
    for ax, pair in zip(axes.ravel(), pairs):
        subset = [r for r in output if r["pair_id"] == pair]
        methods = sorted({r["method"] for r in subset})
        for method in methods:
            points = sorted([r for r in subset if r["method"] == method], key=lambda r: r["n"])
            ax.errorbar([r["n"] for r in points], [r["estimate_mean"] for r in points],
                yerr=[r["estimate_se"] for r in points], marker="o", markersize=4, capsize=2,
                label=METHOD_LABELS.get(method, method), color=COLORS.get(method))
        points = sorted([r for r in subset if r["method"] == methods[0]], key=lambda r: r["n"])
        ax.errorbar([r["n"] for r in points], [r["risk_mean"] for r in points],
            yerr=[r["risk_se"] for r in points], color="#222222", marker="s", markersize=4,
            label="Actual population risk", capsize=2)
        ax.axhline(points[0]["limit_risk"], color="#222222", linestyle="--", linewidth=.9,
                   label="True-risk limit")
        ax.axhline(points[0]["limit_gcv"], color=COLORS["gcv_ma"], linestyle=":", linewidth=1.2,
                   label="Naive-GCV limit")
        ax.set_xlabel(r"Sample size $n=p$")
        ax.set_ylabel("Total risk or risk estimate")
        pair_label = {"0": r"$\lambda=(0.1,1)$", "1": r"$\lambda=(0.01,10)$"}.get(str(pair), f"pair {pair}")
        ax.set_title(f"Fixed equal weights; {pair_label}")
        ax.set_xscale("log")
        ax.minorticks_off()
        ns = sorted({r["n"] for r in points})
        ax.set_xticks(ns, labels=[str(n) for n in ns])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, -.16), ncol=3, frameon=False)
    fig.tight_layout()
    save_figure(fig, figures / "mp_validation")
    plt.close(fig)
    return output


def summarize_de(path: Path, results: Path, figures: Path):
    if not path.exists():
        return []
    raw = load_csv(path)
    output = []
    for (model, gamma, n, method), subset in sorted(groups(raw, ["model", "gamma", "n", "method"]).items()):
        exact = np.array([r["exact_prior_risk"] for r in subset])
        de = np.array([r["de_risk"] for r in subset])
        _, exact_mean, exact_se = stats(exact)
        _, de_mean, de_se = stats(de)
        _, diff, diff_se = stats(exact - de)
        _, relative, relative_se = stats((exact - de) / de)
        output.append({"model": model, "gamma": gamma, "n": n, "method": method,
            "repetitions": len(subset), "exact_prior_risk_mean": exact_mean,
            "exact_prior_risk_se": exact_se, "de_risk_mean": de_mean,
            "de_risk_se": de_se, "mean_exact_minus_de": diff,
            "difference_mc_se": diff_se, "mean_relative_error": relative,
            "relative_error_mc_se": relative_se})
    write_csv(results / "de_summary.csv", output)
    plt = configure_plots()
    models = sorted({r["model"] for r in output})
    fig, axes = plt.subplots(1, len(models), figsize=(5.1 * len(models), 3.3), squeeze=False)
    for ax, model in zip(axes.ravel(), models):
        for (gamma, method), subset in sorted(groups([r for r in output if r["model"] == model],
                                                     ["gamma", "method"]).items()):
            points = sorted(subset, key=lambda r: r["n"])
            ax.errorbar([r["n"] for r in points], [100 * r["mean_relative_error"] for r in points],
                yerr=[100 * r["relative_error_mc_se"] for r in points], marker="o", capsize=2,
                label=rf"$p/n={gamma:g}$; {METHOD_LABELS.get(method, method)}")
        ax.axhline(0, color="#333333", linewidth=.8)
        ax.set_xlabel(r"Sample size $n$")
        ax.set_ylabel("Exact conditional risk minus DE (%)")
        ax.set_title(model)
        ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figures / "de_validation")
    plt.close(fig)
    return output


def diagnostics(raw, summary, oracle, destination):
    keys = [(r["config_id"], r["rep"], r["method"]) for r in raw]
    counts = [r["repetitions"] for r in summary]
    data_methods = [r for r in raw if r["method"] not in ORACLE_METHODS]
    leaked_conditional = int(sum(np.isfinite(r.get("conditional_risk", np.nan)) for r in data_methods))
    metadata = {
        "raw_rows": len(raw), "configurations": len({r["config_id"] for r in raw}),
        "methods": sorted({r["method"] for r in raw}),
        "independent_training_repetitions": len({(r["config_id"], r["rep"]) for r in raw}),
        "duplicate_method_rep_keys": len(keys) - len(set(keys)),
        "balanced_method_repetitions": len(set(counts)) == 1,
        "repetitions_per_config_method": sorted(set(counts)),
        "risk_is_finite": all(np.isfinite(r["risk"]) for r in raw),
        "risk_at_least_noise_variance_one": all(r["risk"] >= 1 - 1e-9 for r in raw),
        "solver_failures": sum(not r.get("solver_success", True) for r in raw),
        "finite_conditional_risk_for_nonoracle_methods": leaked_conditional,
        "uncertainty": "SE across independent training repetitions; paired differences use shared repetitions. MC 95% intervals are normal approximations, with no multiplicity correction.",
        "calibration": "Ratio of mean in-training risk estimate to mean independently evaluated total population risk. Ratio SE uses paired delta method.",
        "gain_sign": "Positive conditional oracle gain favors averaging over the continuous nonnegative-ridge oracle.",
        "plots": "All error bars are +/- one Monte Carlo SE. Main empirical-risk figure uses log excess risk; minimum-norm and equal-weight comparators are retained in CSV tables.",
        "minimum_norm_interpolation": "For p=n the Gaussian minimum-norm population risk has an infinite expectation. Its sample mean is descriptive only; ordinary mean SEs and normal intervals are deliberately omitted. tail_summary.csv contains descriptive quantiles and maxima.",
    }
    for method in ["oracle_convex", "oracle_signed_L3"]:
        rows = [r for r in oracle if r["method"] == method]
        if rows:
            best = max(rows, key=lambda r: r["mean_conditional_gain"])
            control = [r for r in rows if abs(r["delta"]) < 1e-8]
            metadata[method] = {
                "largest_conditional_gain": best,
                "isotropic_control_max_abs_mean_gain": max(abs(r["mean_conditional_gain"]) for r in control) if control else None,
            }
    (destination / "analysis_diagnostics.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    results, figures = args.root / "Results", args.root / "Figures"
    raw = load_csv(results / "main_raw.csv")
    if not raw:
        raise ValueError("main_raw.csv has no records")
    summary, calibration = summarize_main(raw, results)
    paired, oracle = summarize_pairs(raw, results)
    metadata = diagnostics(raw, summary, oracle, results)
    if metadata["duplicate_method_rep_keys"] or not metadata["risk_is_finite"]:
        raise ValueError("Invalid or duplicate raw experiment records")
    if metadata["finite_conditional_risk_for_nonoracle_methods"]:
        raise ValueError("Conditional risk for data-selected weights must remain undefined")
    if not args.no_plots:
        draw_main_figures(summary, calibration, oracle, figures)
        summarize_mp(results / "mp_validation.csv", results, figures)
        summarize_de(results / "de_validation.csv", results, figures)
    summarize_tails(raw, results, figures, make_plot=not args.no_plots)
    summarize_weights(results / "weights.csv", results)
    print(json.dumps({key: metadata[key] for key in ["raw_rows", "configurations",
        "independent_training_repetitions", "solver_failures", "repetitions_per_config_method"]}, indent=2))


if __name__ == "__main__":
    main()
