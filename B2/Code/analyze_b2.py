"""Summarize, verify, and plot the B2 formal experiment."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

import run_b2


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "Results"
FIGURES = ROOT / "Figures"


METHODS = {
    "five-fold adaptive": ("cv5_selected_excess", "cv5_selected_gcv_score"),
    "full grid": ("full_grid_excess", "full_grid_gcv_score"),
    "fixed lower 1": ("fixed_one_excess", "fixed_one_gcv_score"),
    "single split": ("split_selected_excess", "split_selected_gcv_score"),
    "Ridge-GCV": ("ridge_gcv_excess", "ridge_gcv_score"),
    "LOO-MA": ("loo_ma_excess", "loo_ma_gcv_score"),
}


def mean_se(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


def summarize(selected):
    keys = ["config_id", "n", "p", "gamma", "cov", "snr"]
    summary_rows, paired_rows = [], []
    for key, group in selected.groupby(keys, sort=True):
        base = dict(zip(keys, key))
        for method, (risk_col, score_col) in METHODS.items():
            mean, se = mean_se(group[risk_col])
            score_mean = float(group[score_col].mean())
            summary_rows.append({
                **base, "method": method, "repetitions": len(group),
                "mean_excess": mean, "se_excess": se,
                "mean_gcv_score": score_mean,
                "calibration_ratio": score_mean / (1.0 + mean),
                "calibration_abs_error": abs(score_mean / (1.0 + mean) - 1.0),
            })
        adaptive = group["cv5_selected_excess"]
        for baseline, (risk_col, _) in METHODS.items():
            if baseline == "five-fold adaptive":
                continue
            diff = adaptive - group[risk_col]
            mean, se = mean_se(diff)
            critical = float(t.ppf(0.975, len(diff) - 1))
            low, high = mean - critical * se, mean + critical * se
            status = "inconclusive"
            if high < 0:
                status = "adaptive lower"
            elif low > 0:
                status = "adaptive higher"
            paired_rows.append({
                **base, "contrast": f"five-fold adaptive minus {baseline}",
                "repetitions": len(diff), "mean_diff": mean, "se_diff": se,
                "ci95_low": low, "ci95_high": high, "status": status,
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(paired_rows)


def aggregate(summary, paired):
    wide = summary.pivot(index=["config_id", "n", "p", "gamma", "cov", "snr"],
                         columns="method", values=["mean_excess", "calibration_abs_error"])
    wide.columns = [f"{a}__{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    wide["risk_ratio_vs_full"] = (
        wide["mean_excess__five-fold adaptive"] / wide["mean_excess__full grid"]
    )
    wide["risk_ratio_vs_fixed1"] = (
        wide["mean_excess__five-fold adaptive"] / wide["mean_excess__fixed lower 1"]
    )
    counts = []
    for gamma, group in wide.groupby("gamma"):
        pfull = paired[(paired.gamma == gamma) &
                       (paired.contrast == "five-fold adaptive minus full grid")]
        pfixed = paired[(paired.gamma == gamma) &
                        (paired.contrast == "five-fold adaptive minus fixed lower 1")]
        counts.append({
            "gamma": gamma, "scenarios": len(group),
            "median_ratio_vs_full": group.risk_ratio_vs_full.median(),
            "range_ratio_vs_full_low": group.risk_ratio_vs_full.min(),
            "range_ratio_vs_full_high": group.risk_ratio_vs_full.max(),
            "median_ratio_vs_fixed1": group.risk_ratio_vs_fixed1.median(),
            "range_ratio_vs_fixed1_low": group.risk_ratio_vs_fixed1.min(),
            "range_ratio_vs_fixed1_high": group.risk_ratio_vs_fixed1.max(),
            "full_significantly_lower": int((pfull.status == "adaptive lower").sum()),
            "full_significantly_higher": int((pfull.status == "adaptive higher").sum()),
            "fixed1_significantly_lower": int((pfixed.status == "adaptive lower").sum()),
            "fixed1_significantly_higher": int((pfixed.status == "adaptive higher").sum()),
            "median_calibration_error_adaptive": group["calibration_abs_error__five-fold adaptive"].median(),
            "median_calibration_error_full": group["calibration_abs_error__full grid"].median(),
        })
    return wide, pd.DataFrame(counts)


def selection_counts(selected):
    overall = (selected.groupby("chosen_log10_lower_cv5").size()
               .rename("count").reset_index())
    overall["proportion"] = overall["count"] / len(selected)
    by_scenario = (selected.groupby(["config_id", "n", "gamma", "cov", "snr",
                                     "chosen_log10_lower_cv5"])
                   .size().rename("count").reset_index())
    by_scenario["proportion"] = by_scenario["count"] / 200.0
    return overall, by_scenario


def make_figures(wide, selected):
    FIGURES.mkdir(exist_ok=True)
    colors = {1.0: "#305f87", 5.0: "#9e2a2b"}
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.5), sharex=True)
    for row, n in enumerate((100, 200)):
        for col, cov in enumerate(("iid", "toeplitz")):
            ax = axes[row, col]
            data = wide[(wide.n == n) & (wide["cov"] == cov)]
            for snr in (1.0, 5.0):
                d = data[data.snr == snr].sort_values("gamma")
                ax.plot(d.gamma, d.risk_ratio_vs_full, marker="o", color=colors[snr],
                        label=f"SNR={snr:g}: vs full")
                ax.plot(d.gamma, d.risk_ratio_vs_fixed1, marker="s", ls="--",
                        color=colors[snr], label=f"SNR={snr:g}: vs fixed 1")
            ax.axhline(1.0, color="black", lw=0.8)
            ax.set_title(f"n={n}, {cov}")
            ax.set_ylim(0, 1.35)
            ax.set_ylabel("risk ratio")
            ax.set_xticks((0.95, 1.0, 1.05, 1.5))
            ax.set_xlabel("p/n")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"B2_risk_ratios.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)

    order = [-4.0, -3.0, -2.0, -1.0, 0.0]
    labels = [r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$1$"]
    grouped = (selected.groupby(["snr", "gamma", "chosen_log10_lower_cv5"])
               .size().unstack(fill_value=0).reindex(columns=order, fill_value=0))
    grouped = grouped.div(grouped.sum(axis=1), axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    palette = ["#8c1d40", "#c65f4a", "#e7a35c", "#6e9ab8", "#2f6b4f"]
    for ax, snr in zip(axes, (1.0, 5.0)):
        d = grouped.loc[snr]
        bottom = np.zeros(len(d))
        for floor, label, color in zip(order, labels, palette):
            vals = d[floor].to_numpy()
            ax.bar(np.arange(len(d)), vals, bottom=bottom, label=label, color=color)
            bottom += vals
        ax.set_xticks(np.arange(len(d)), [f"{x:g}" for x in d.index])
        ax.set_xlabel("p/n")
        ax.set_title(f"SNR={snr:g}")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("selection proportion")
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"B2_selection_frequencies.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    x = wide["calibration_abs_error__full grid"]
    y = wide["calibration_abs_error__five-fold adaptive"]
    ax.scatter(x, y, c=wide.gamma, cmap="viridis", s=45, edgecolor="black", linewidth=0.3)
    limit = max(float(x.max()), float(y.max())) * 1.05
    ax.plot([0, limit], [0, limit], ls="--", color="black", lw=0.8)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("full-grid absolute calibration error")
    ax.set_ylabel("five-fold absolute calibration error")
    ax.set_title("32 scenarios")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"B2_calibration.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def verify(floor_raw, selected):
    assert len(selected) == 6400
    assert len(floor_raw) == 32000
    assert selected.groupby("config_id").size().eq(200).all()
    assert floor_raw.groupby(["config_id", "log10_lower"]).size().eq(200).all()
    assert floor_raw.gcv_qp_success.all() and floor_raw.oracle_qp_success.all()
    assert selected.loo_qp_success.all()
    assert floor_raw.gcv_qp_gap.max() < 1e-3
    assert floor_raw.oracle_qp_gap.max() < 1e-3
    assert selected.loo_qp_gap.max() < 1e-3

    matched = selected[["config_id", "rep", "chosen_log10_lower_cv5",
                        "cv5_selected_excess"]].merge(
        floor_raw[["config_id", "rep", "log10_lower", "gcv_ma_excess"]],
        left_on=["config_id", "rep", "chosen_log10_lower_cv5"],
        right_on=["config_id", "rep", "log10_lower"], how="left", validate="one_to_one"
    )
    selection_error = float(np.max(np.abs(
        matched.cv5_selected_excess - matched.gcv_ma_excess
    )))
    assert selection_error < 1e-12

    replay_floor, replay_selected = run_b2.task((run_b2.configurations()[0], 0))
    recorded = floor_raw[(floor_raw.config_id == 0) & (floor_raw.rep == 0)].sort_values("log10_lower")
    replay = pd.DataFrame(replay_floor).sort_values("log10_lower")
    replay_error = float(np.max(np.abs(
        recorded.gcv_ma_excess.to_numpy() - replay.gcv_ma_excess.to_numpy()
    )))
    selected_error = abs(
        float(selected[(selected.config_id == 0) & (selected.rep == 0)].cv5_selected_excess.iloc[0])
        - float(replay_selected["cv5_selected_excess"])
    )
    assert replay_error < 1e-12 and selected_error < 1e-12
    return {
        "floor_rows": len(floor_raw), "selected_rows": len(selected),
        "scenarios": int(selected.config_id.nunique()),
        "replications_per_scenario": 200,
        "all_qp_success": True,
        "max_gcv_qp_gap": float(floor_raw.gcv_qp_gap.max()),
        "max_oracle_qp_gap": float(floor_raw.oracle_qp_gap.max()),
        "max_loo_qp_gap": float(selected.loo_qp_gap.max()),
        "selection_lookup_max_abs_error": selection_error,
        "replay_floor_max_abs_error": replay_error,
        "replay_selected_abs_error": selected_error,
    }


def main():
    floor_raw = pd.read_csv(RESULTS / "B2_floor_raw.csv")
    selected = pd.read_csv(RESULTS / "B2_selected_raw.csv")
    verification = verify(floor_raw, selected)
    summary, paired = summarize(selected)
    wide, aggregate_table = aggregate(summary, paired)
    overall_counts, scenario_counts = selection_counts(selected)

    summary.to_csv(RESULTS / "B2_method_summary.csv", index=False)
    paired.to_csv(RESULTS / "B2_paired_contrasts.csv", index=False)
    wide.to_csv(RESULTS / "B2_scenario_summary.csv", index=False)
    aggregate_table.to_csv(RESULTS / "B2_gamma_aggregate.csv", index=False)
    overall_counts.to_csv(RESULTS / "B2_selection_counts.csv", index=False)
    scenario_counts.to_csv(RESULTS / "B2_selection_counts_by_scenario.csv", index=False)
    (RESULTS / "B2_verification.json").write_text(
        json.dumps(verification, indent=2), encoding="utf-8"
    )
    make_figures(wide, selected)
    print(json.dumps(verification, indent=2))
    print(aggregate_table.to_string(index=False))
    print(overall_counts.to_string(index=False))


if __name__ == "__main__":
    main()
