"""Recalculate all report tables and figures from the recorded simulations."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import a2_core


ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "Results"
F = ROOT / "Figures"
METHODS = ("GCV-MA", "Ridge-GCV", "LOO-MA", "Oracle-MA")


def se(values):
    x = np.asarray(values, float)
    return float(x.std(ddof=1) / np.sqrt(len(x)))


def summarize_a2(raw):
    assert len(raw) == 18000
    assert set(raw.method) == set(a2_core.METHODS)
    assert raw.groupby(["gamma", "lo", "method"]).size().eq(150).all()
    assert raw.groupby(["gamma", "lo"]).dgp_id.nunique().eq(1).all()
    rows = []
    for (gamma, lo, method), g in raw.groupby(["gamma", "lo", "method"]):
        rows.append(dict(gamma=gamma, lower=10**lo, log10_lower=lo, method=method,
                         n=len(g), mean_excess=g.excess.mean(), se_excess=se(g.excess),
                         median_excess=g.excess.median(), mean_total_risk=g.risk.mean(),
                         mean_gcv_score=g.gcv_score.mean(),
                         calibration_ratio=g.gcv_score.mean()/g.risk.mean(),
                         mean_test_mse=g.test_mse.mean(),
                         mean_effective_m=g.effective_m.mean()))
    return pd.DataFrame(rows).sort_values(["gamma", "log10_lower", "method"])


def compare_reference(new, old):
    both = pd.concat([new.assign(series="independent"), old.assign(series="week1")])
    rows=[]
    for (series,gamma,lo,method),g in both.groupby(["series","gamma","lo","method"]):
        if method not in METHODS: continue
        rows.append(dict(series=series,gamma=gamma,log10_lower=lo,method=method,
                         mean_excess=g.excess.mean(),se_excess=se(g.excess),
                         calibration_ratio=g.gcv_score.mean()/g.risk.mean()))
    return pd.DataFrame(rows).sort_values(["gamma","log10_lower","method","series"])


def paired_a2(raw):
    rows=[]
    for gamma in (1.,1.5):
        x=raw[(raw.gamma==gamma)&(raw.method.isin(METHODS))]
        pivot=x.pivot(index="rep",columns=["lo","method"],values="excess")
        for lo in (-4.,-3.,-2.,-1.):
            for method in METHODS:
                diff=pivot[(lo,method)]-pivot[(0.,method)]
                rows.append(dict(gamma=gamma,method=method,log10_lower=lo,
                                 contrast="lower minus 1",mean_diff=diff.mean(),
                                 se_diff=se(diff),n=len(diff)))
        for lo in (-4.,-3.,-2.,-1.,0.):
            for baseline in ("Ridge-GCV","LOO-MA","Oracle-MA"):
                diff=pivot[(lo,"GCV-MA")]-pivot[(lo,baseline)]
                rows.append(dict(gamma=gamma,method="GCV-MA vs "+baseline,
                                 log10_lower=lo,contrast="same grid",mean_diff=diff.mean(),
                                 se_diff=se(diff),n=len(diff)))
    return pd.DataFrame(rows)


def summarize_b(raw, selections):
    grid=raw[raw.gcv_ma_excess.notna()].copy()
    rows=[]; paired=[]
    for gamma in (1.,1.5):
        g=grid[grid.gamma==gamma]
        s=selections[selections.gamma==gamma].set_index("rep")
        assert len(s)==100 and g.groupby("log10_lower").size().eq(100).all()
        wide=g.pivot(index="rep",columns="log10_lower",values="gcv_ma_excess")
        vals={"five-fold adaptive":s.cv5_selected_excess,
              "single-split adaptive":s.selected_excess,
              "full grid":wide[-4.], "fixed lower 1":wide[0.]}
        calibration={
            "five-fold adaptive":s.cv5_selected_gcv_score.mean()/(s.cv5_selected_excess.mean()+1),
            "single-split adaptive":s.selected_gcv_score.mean()/(s.selected_excess.mean()+1),
            "full grid":g[g.log10_lower==-4.].gcv_ma_score.mean()/(wide[-4.].mean()+1),
            "fixed lower 1":g[g.log10_lower==0.].gcv_ma_score.mean()/(wide[0.].mean()+1),
        }
        for label,v in vals.items():
            rows.append(dict(gamma=gamma,method=label,n=len(v),mean_excess=v.mean(),
                             se_excess=se(v),median_excess=v.median(),
                             calibration_ratio=calibration[label]))
        for label in ("Ridge-GCV","LOO-MA"):
            v=raw[(raw.gamma==gamma)&(raw.method==label)].set_index("rep").excess
            assert len(v)==100
            rows.append(dict(gamma=gamma,method=label,n=len(v),mean_excess=v.mean(),
                             se_excess=se(v),median_excess=v.median(),
                             calibration_ratio=np.nan))
        for baseline in ("full grid","fixed lower 1","single-split adaptive"):
            diff=vals["five-fold adaptive"]-vals[baseline]
            paired.append(dict(gamma=gamma,contrast="five-fold adaptive minus "+baseline,
                               n=len(diff),mean_diff=diff.mean(),se_diff=se(diff),
                               ci95_low=diff.mean()-1.984*se(diff),
                               ci95_high=diff.mean()+1.984*se(diff)))
    return pd.DataFrame(rows),pd.DataFrame(paired)


def figures(a2,b):
    F.mkdir(exist_ok=True)
    methods=("GCV-MA","Ridge-GCV","LOO-MA","Oracle-MA")
    colors={"GCV-MA":"#9e2a2b","Ridge-GCV":"#305f87",
            "LOO-MA":"#2f7654","Oracle-MA":"#666666"}
    fig,axes=plt.subplots(2,2,figsize=(10,6),sharex=True)
    for j,gamma in enumerate((1.,1.5)):
        data=a2[a2.gamma==gamma]
        for method in methods:
            d=data[data.method==method].sort_values("log10_lower")
            axes[0,j].errorbar(d.log10_lower,d.mean_excess,yerr=1.96*d.se_excess,
                               marker="o",capsize=2,label=method,color=colors[method])
            axes[1,j].plot(d.log10_lower,d.calibration_ratio,marker="o",label=method,
                           color=colors[method])
        axes[0,j].set_title(f"p/n = {gamma:g}")
        axes[0,j].set_yscale("log")
        axes[1,j].axhline(1,color="black",lw=.8,ls="--")
        axes[1,j].set_xlabel("log10 lower ridge parameter")
        axes[1,j].set_ylim(0,1.2)
    axes[0,0].set_ylabel("Mean excess risk (95% MC interval)")
    axes[1,0].set_ylabel("Mean GCV score / mean total risk")
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",ncol=4,bbox_to_anchor=(.5,1.02))
    fig.tight_layout(rect=(0,0,1,.94))
    for ext in ("pdf","png"):fig.savefig(F/f"A2_boundary_replication.{ext}",dpi=220,bbox_inches="tight")
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    names=("full grid","single-split adaptive","five-fold adaptive","fixed lower 1")
    for j,gamma in enumerate((1.,1.5)):
        d=b[b.gamma==gamma].set_index("method").loc[list(names)]
        axes[j].bar(range(4),d.mean_excess,color=["#9e2a2b","#7f7f7f","#305f87","#2f7654"])
        axes[j].errorbar(range(4),d.mean_excess,yerr=1.96*d.se_excess,fmt="none",color="black",capsize=3)
        axes[j].set_xticks(range(4),("full","split","5-fold","fixed 1"))
        axes[j].set_title(f"p/n = {gamma:g}")
        axes[j].set_ylabel("Mean excess risk (95% MC interval)")
    fig.tight_layout()
    for ext in ("pdf","png"):fig.savefig(F/f"B_pilot_risk.{ext}",dpi=220,bbox_inches="tight")
    plt.close(fig)


def main():
    raw=pd.read_csv(R/"A2_raw.csv")
    old=pd.read_csv(R/"A2_reference_boundary_raw.csv")
    braw=pd.read_csv(R/"B_pilot_raw.csv")
    bs=pd.read_csv(R/"B_pilot_selections.csv")
    assert set(old.family)=={"boundary"} and len(old)==18000
    assert raw.rep.nunique()==150 and old.rep.nunique()==150
    assert raw.qp_success.all()
    assert raw.qp_gap.max()<1e-3
    cfg=[c for c in a2_core.configurations() if c["family"]=="boundary"][0]
    replay,_=a2_core.one_task((cfg,0))
    recorded=raw[(raw.config_id==cfg["config_id"])&(raw.rep==0)].set_index("method")
    replay_error=max(abs(recorded.loc[r["method"],"excess"]-r["excess"]) for r in replay)
    assert replay_error<1e-9
    a2=summarize_a2(raw)
    ref=compare_reference(raw,old)
    paired=paired_a2(raw)
    b,bpaired=summarize_b(braw,bs)
    a2.to_csv(R/"A2_summary.csv",index=False)
    ref.to_csv(R/"A2_reference_comparison.csv",index=False)
    paired.to_csv(R/"A2_paired.csv",index=False)
    b.to_csv(R/"B_pilot_summary.csv",index=False)
    bpaired.to_csv(R/"B_pilot_paired.csv",index=False)
    counts=[]
    for rule,column in (("single-split","chosen_log10_lower"),("five-fold","chosen_log10_lower_cv5")):
        c=bs.groupby(["gamma",column]).size().rename("count").reset_index().rename(columns={column:"chosen_log10_lower"})
        c["rule"]=rule
        counts.append(c)
    pd.concat(counts).to_csv(R/"B_pilot_selection_counts.csv",index=False)
    figures(a2,b)
    checks={"A2_replay_max_abs_excess_error":replay_error,"A2_qp_success":True,
            "A2_max_qp_gap":float(raw.qp_gap.max()),"A2_rows":len(raw),
            "B_pilot_rows":len(braw),"B_pilot_selected":len(bs)}
    (R/"verification.json").write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))


if __name__=="__main__":main()
