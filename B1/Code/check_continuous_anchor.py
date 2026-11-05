"""Remove a finite-grid confound by appending the continuous oracle penalty.

This is an oracle-only diagnostic on the SAME main designs, not new datasets
and not a data-realizable method. No beta/noise realization enters selection.
"""
from __future__ import annotations
import csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from ridge_core import prepare_design,prior_risk_matrix,solve_weights
from run_experiment import population
ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'Results'
    manifest=json.loads((out/'manifest.json').read_text())
    with (out/'main_raw.csv').open() as f:
        raw=[r for r in csv.DictReader(f) if r['method']=='ridge_cont_oracle']
    cfgs={c['config_id']:c for c in manifest['configurations']}
    rows=[]
    with threadpool_limits(limits=1):
        for r in raw:
            cfg=cfgs[r['config_id']];p,n=cfg['p'],cfg['n']
            rng=np.random.default_rng(int(r['seed']))
            mu,psi=population(p,cfg['a'],cfg['b'])
            X=rng.normal(size=(n,p))*np.sqrt(mu)
            design=prepare_design(X,mu,psi)
            lamc=float(r['selected_lambda'])
            lam=np.unique(np.r_[np.logspace(-4,2,21),cfg['gamma'],lamc])
            C=prior_risk_matrix(design,lam)
            w,diag=solve_weights(C)
            assert diag['success']
            risk=1+float(w@C@w)
            gain=float(r['conditional_risk'])-risk
            assert gain>-2e-7
            rows.append(dict(config_id=r['config_id'],rep=int(r['rep']),seed=int(r['seed']),
                             a=cfg['a'],gamma=cfg['gamma'],delta=cfg['delta'],
                             continuous_lambda=lamc,augmented_convex_risk=risk,gain=gain,
                             normalized_solver_gap=diag['normalized_kkt_gap'],
                             raw_solver_gap=diag['kkt_gap']))
    with (out/'anchor_diagnostic.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    grouped=defaultdict(list)
    for r in rows:grouped[r['config_id']].append(r)
    summary=[]
    for cfg,rr in grouped.items():
        x=np.array([r['gain'] for r in rr])
        summary.append(dict(config_id=cfg,a=rr[0]['a'],gamma=rr[0]['gamma'],delta=rr[0]['delta'],
                            mean_gain=float(x.mean()),mc_se=float(x.std(ddof=1)/np.sqrt(len(x))),
                            min_gain=float(x.min()),max_gain=float(x.max()),
                            max_raw_solver_gap=max(r['raw_solver_gap'] for r in rr)))
    with (out/'anchor_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
