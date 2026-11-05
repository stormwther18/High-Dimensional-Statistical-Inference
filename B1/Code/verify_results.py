"""Audit saved results, oracle ordering, GCV transformation, and seed replay."""
from __future__ import annotations
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from run_experiment import one_rep

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'Results'
    manifest=json.loads((out/'manifest.json').read_text())
    with (out/'main_raw.csv').open() as f: rows=list(csv.DictReader(f))
    group=defaultdict(dict)
    for row in rows:
        group[(row['config_id'],int(row['rep']))][row['method']]=row
        assert math.isfinite(float(row['risk'])) and float(row['risk'])>=1
        assert float(row['solver_violation'])<1e-7
        assert float(row['l1_norm'])<=3+1e-7
    assert len(group)==manifest['training_datasets']
    assert len(rows)==12*manifest['training_datasets']
    assert all(row['solver_success']=='True' for row in rows)
    assert all(len(methods)==12 for methods in group.values())
    seeds={int(next(iter(x.values()))['seed']) for x in group.values()}
    assert len(seeds)==len(group)
    worst_order=0.;isotropic_max=0.
    for methods in group.values():
        r=lambda name:float(methods[name]['conditional_risk'])
        for left,right in [('bayes_oracle','oracle_signed_L3'),
                           ('oracle_signed_L3','oracle_convex'),
                           ('oracle_convex','ridge_grid_oracle'),
                           ('ridge_cont_oracle','ridge_grid_oracle')]:
            violation=r(left)-r(right)
            worst_order=max(worst_order,violation)
            assert violation<2e-6,(left,right,violation)
        if abs(float(methods['oracle_convex']['delta']))<1e-12:
            isotropic_max=max(isotropic_max,abs(r('oracle_convex')-r('ridge_cont_oracle')),
                              abs(r('oracle_signed_L3')-r('ridge_cont_oracle')),
                              abs(r('bayes_oracle')-r('ridge_cont_oracle')))
    assert isotropic_max<2e-6

    weight_groups=defaultdict(list)
    with (out/'weights.csv').open() as f:
        for row in csv.DictReader(f):
            weight_groups[(row['config_id'],int(row['rep']),row['method'])].append(row)
    max_transform_error=max_score_error=0.
    mechanism=[]
    for cfg,rep in group:
        cross=weight_groups[(cfg,rep,'cross_gcv_ma')]
        naive=weight_groups[(cfg,rep,'gcv_ma')]
        u=np.array([float(r['weight']) for r in cross])
        d=np.array([float(r['residual_denominator']) for r in cross])
        pi=np.array([float(r['weight']) for r in naive])
        predicted=u/d;predicted/=predicted.sum()
        err=float(np.max(np.abs(pi-predicted)))
        max_transform_error=max(max_transform_error,err)
        score1=float(group[(cfg,rep)]['gcv_ma']['risk_estimate'])
        score2=float(group[(cfg,rep)]['cross_gcv_ma']['risk_estimate'])
        serr=abs(score1-score2)
        max_score_error=max(max_score_error,serr)
        mechanism.append(dict(config_id=cfg,rep=rep,weight_l1_distance=float(np.abs(pi-u).sum()),
                              denominator_min=float(d.min()),denominator_max=float(d.max()),
                              shared_minimum_score=score2,score_difference=serr,
                              transform_max_error=err))
    assert max_transform_error<1e-11 and max_score_error<1e-9
    with (out/'anchor_diagnostic.csv').open() as f: anchor=list(csv.DictReader(f))
    assert len(anchor)==len(group)
    assert all(float(r['gain'])>-2e-7 for r in anchor)
    strong_anchor=[r for r in anchor if float(r['delta'])>0]
    max_strong_anchor_gain=max(abs(float(r['gain'])) for r in strong_anchor)
    assert max_strong_anchor_gain<1e-10
    with (out/'mechanism.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(mechanism[0]));writer.writeheader();writer.writerows(mechanism)
    replay_error=0.
    with threadpool_limits(limits=1):
        for cfg in manifest['configurations']:
            replay,_=one_rep(cfg,0,manifest['root_seed'])
            for row in replay:
                old=group[(cfg['config_id'],0)][row['method']]
                for key in ['risk','risk_estimate','conditional_risk','l1_norm']:
                    x,y=float(old[key]),float(row[key])
                    if math.isnan(x) and math.isnan(y):continue
                    err=abs(x-y);replay_error=max(replay_error,err)
                    assert err<1e-8*max(1,abs(x)),(cfg,row['method'],key,err)
    result=dict(passed=True,training_datasets=len(group),method_records=len(rows),
                distinct_seeds=len(seeds),worst_oracle_order_violation=worst_order,
                isotropic_oracle_max_difference=isotropic_max,
                max_gcv_weight_transform_error=max_transform_error,
                max_equivalent_objective_difference=max_score_error,
                replayed_datasets=len(manifest['configurations']),max_replay_abs_error=replay_error,
                solver_failures=sum(r['solver_success']!='True' for r in rows),
                max_normalized_solver_gap=max(float(r['solver_gap']) for r in rows),
                max_raw_solver_gap=max(float(r['solver_gap_raw']) for r in rows),
                max_constraint_violation=max(float(r['solver_violation']) for r in rows))
    result['continuous_anchor_datasets']=len(anchor)
    result['strong_alignment_max_abs_anchor_gain']=max_strong_anchor_gain
    (out/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
