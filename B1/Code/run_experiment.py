"""Paired finite-dimensional source-alignment experiment for ridge averaging.

All selectors use training X,y only. The known population/prior is used solely
by explicitly named oracle methods and by independent analytic evaluation.
Run from any directory: python Code/run_experiment.py --repetitions 100
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import minimize_scalar
from threadpoolctl import threadpool_limits
from ridge_core import (prepare_design, candidate_path, single_prior_risk,
                        solve_weights, realized_risk)

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['config_id','n','p','gamma','a','b','delta','rep','seed','method',
          'risk','excess','risk_estimate','conditional_risk','l1_norm',
          'selected_lambda','solver_success','solver_gap','solver_gap_raw',
          'solver_violation','optimizer_success','seconds']
WEIGHT_FIELDS = ['config_id','rep','seed','method','lambda','weight','residual_denominator']


def configurations():
    out = []
    for a in (1.5, 2.0):
        for gamma in (0.8, 1.0, 1.5):
            for delta in (-0.3, 0.0, 0.3):
                out.append(dict(config_id=f'a{a:.1f}_g{gamma:.1f}_d{delta:+.1f}',
                                n=100,p=round(100*gamma),gamma=gamma,a=a,
                                b=round(a+delta,4),delta=delta))
    return out


def population(p, a, b):
    index = np.arange(1,p+1,dtype=float)
    mu = index**(-a)
    mu /= mu.mean()
    psi = index**(a-b)
    psi /= np.dot(mu,psi)  # E beta' Sigma beta = 1; noise variance = 1.
    return mu,psi


def continuous_oracle(design, gamma, isotropic):
    if isotropic:
        return gamma, single_prior_risk(design,gamma)
    # Search every detected local minimum on a broad log grid, rather than
    # presuming the anisotropic risk curve is globally unimodal.
    grid = np.linspace(np.log(1e-6),np.log(1e4),101)
    f = lambda z: float(single_prior_risk(design,float(np.exp(z))))
    vals = np.array([f(z) for z in grid])
    candidates = [(vals[0],grid[0]),(vals[-1],grid[-1])]
    for j in range(1,len(grid)-1):
        if vals[j] <= vals[j-1] and vals[j] <= vals[j+1]:
            opt = minimize_scalar(f,bounds=(grid[j-1],grid[j+1]),
                                  method='bounded',options={'xatol':1e-10})
            if not opt.success:
                raise ArithmeticError('Continuous oracle search failed')
            candidates.append((float(opt.fun),float(opt.x)))
    best = min(candidates)
    return float(np.exp(best[1])),best[0]


def diagnostic_value(diag, *names, default=0.):
    for name in names:
        if name in diag:
            return diag[name]
    return default


def one_rep(cfg,rep,root_seed):
    t0=time.perf_counter()
    config_index=configurations().index(cfg)
    seed=int(np.random.SeedSequence([root_seed,0,config_index,rep]).generate_state(1)[0])
    rng=np.random.default_rng(seed)
    n,p=cfg['n'],cfg['p']
    mu,psi=population(p,cfg['a'],cfg['b'])
    X=rng.normal(size=(n,p))*np.sqrt(mu)
    beta=rng.normal(size=p)*np.sqrt(psi)
    y=X@beta+rng.normal(size=n)
    design=prepare_design(X,mu,psi,noise_variance=1.)
    lambdas=np.unique(np.r_[np.logspace(-4,2,21),cfg['gamma']])
    path=candidate_path(design,y,lambdas)
    C=path['prior_risk_matrix']
    W=path['corrected_matrix']
    LOO=path['loo_matrix']
    k=len(lambdas)
    unit=lambda j: np.eye(k)[j]
    methods={}
    methods['ridge_grid_oracle']=(unit(int(np.argmin(np.diag(C)))),{},'prior')
    wc,dc=solve_weights(C)
    methods['oracle_convex']=(wc,dc,'prior')
    ws,ds=solve_weights(C,signed_l1=3.)
    methods['oracle_signed_L3']=(ws,ds,'prior')
    methods['ridge_gcv']=(unit(int(np.argmin(np.diag(W)))),{},'cross')
    wx,dx=solve_weights(W)
    methods['cross_gcv_ma']=(wx,dx,'cross')
    # Exact variable transformation; avoids a redundant optimization.
    d=path['aggregate_denominator']
    wg=wx/d
    wg/=wg.sum()
    methods['gcv_ma']=(wg,dx,'aggregate')
    wl,dl=solve_weights(LOO)
    methods['loo_ma']=(wl,dl,'loo')
    wsn,dsn=solve_weights(W,signed_l1=3.)
    methods['cross_gcv_signed_L3']=(wsn,dsn,'cross')
    methods['equal_ma']=(np.full(k,1/k),{},'cross')
    rows=[]; weights=[]
    meta={key:cfg[key] for key in ['config_id','n','p','gamma','a','b','delta']}

    def record(method,coef,estimate=np.nan,cond=np.nan,w=None,diag=None,lam=np.nan):
        diag=diag or {}
        risk=float(realized_risk(coef,beta,mu,noise_variance=1.))
        success=bool(diagnostic_value(diag,'success','solver_success',default=True))
        gap=float(diagnostic_value(diag,'normalized_kkt_gap','gap','normalized_gap','solver_gap',default=0.))
        rows.append(dict(**meta,rep=rep,seed=seed,method=method,risk=risk,
                         excess=risk-1,risk_estimate=float(estimate),
                         conditional_risk=float(cond),
                         l1_norm=float(np.abs(w).sum()) if w is not None else 1.,
                         selected_lambda=float(lam),solver_success=success,
                         solver_gap=gap,solver_gap_raw=float(diag.get('kkt_gap',0.)),
                         solver_violation=float(diag.get('constraint_violation',0.)),
                         optimizer_success=bool(diag.get('optimizer_success',True)),
                         seconds=time.perf_counter()-t0))

    for method,(w,diag,kind) in methods.items():
        if kind=='prior':
            cond=float(1+w@C@w); estimate=np.nan
        else:
            cond=np.nan
            matrix=W if kind=='cross' else LOO
            if kind=='aggregate':
                estimate=float(np.mean((path['residuals']@w)**2)/(d@w)**2)
            else:
                estimate=float(w@matrix@w)
        coef=path['coefficients']@w
        active=np.flatnonzero(np.abs(w)>1e-7)
        lam=lambdas[active[0]] if len(active)==1 else np.nan
        record(method,coef,estimate,cond,w,diag,lam)
        weights.extend(dict(config_id=cfg['config_id'],rep=rep,seed=seed,
                            method=method,**{'lambda':float(lamj),'weight':float(wj),
                                             'residual_denominator':float(dj)})
                       for lamj,wj,dj in zip(lambdas,w,d))

    lamc,rc=continuous_oracle(design,cfg['gamma'],abs(cfg['delta'])<1e-12)
    pc=candidate_path(design,y,np.array([lamc]))
    record('ridge_cont_oracle',pc['coefficients'][:,0],cond=rc,lam=lamc)
    gridrisk=1+float(np.min(np.diag(C)))
    if rc>gridrisk+2e-7:
        raise ArithmeticError('Continuous search missed a better grid candidate')
    # Minimum-norm least squares: no response-dependent numerical tuning.
    coef=np.linalg.lstsq(X,y,rcond=None)[0]
    record('minnorm',coef)
    # Correct anisotropic Gaussian posterior mean; explicitly privileged oracle.
    XP=X*psi
    kernel=XP@X.T+np.eye(n)
    coef=XP.T@np.linalg.solve(kernel,y)
    B=(XP*mu)@XP.T
    cond=1+float(mu@psi)-float(np.trace(np.linalg.solve(kernel,B)))
    record('bayes_oracle',coef,cond=cond)
    return rows,weights


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repetitions',type=int,default=100)
    ap.add_argument('--seed',type=int,default=202611041)
    ap.add_argument('--config-limit',type=int,default=None)
    ap.add_argument('--output',type=Path,default=ROOT/'Results')
    args=ap.parse_args()
    if args.repetitions<1: ap.error('repetitions must be positive')
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    configs=configurations()[:args.config_limit]
    start=time.perf_counter()
    with (out/'main_raw.csv').open('w',newline='') as fr,(out/'weights.csv').open('w',newline='') as fw:
        wr=csv.DictWriter(fr,fieldnames=FIELDS);wr.writeheader()
        ww=csv.DictWriter(fw,fieldnames=WEIGHT_FIELDS);ww.writeheader()
        with threadpool_limits(limits=1):
            for cfg in configs:
                for rep in range(args.repetitions):
                    rows,weights=one_rep(cfg,rep,args.seed)
                    wr.writerows(rows);ww.writerows(weights)
                fr.flush();fw.flush()
                print(f"{cfg['config_id']}: {args.repetitions} repetitions; elapsed {time.perf_counter()-start:.1f}s",flush=True)
    manifest=dict(root_seed=args.seed,repetitions=args.repetitions,configurations=configs,
                  training_datasets=len(configs)*args.repetitions,
                  noise_variance=1.,predictive_snr=1.,signed_l1_bound=3.,
                  lambda_grid='21 log-spaced points in [1e-4,100], plus gamma',
                  continuous_oracle='[1e-6,1e4], 101 log probes, refine all detected local minima; exact lambda=gamma if isotropic',
                  risk_target='1+(beta_hat-beta)^T Sigma (beta_hat-beta), analytic population evaluation',
                  oracle_information='X and known Sigma/Psi only, never realized beta or epsilon',
                  prior_risk_matrix_excludes_noise=True,
                  selected_conditional_risk='NaN: fixed-weight formula is not substituted at y-selected weights',
                  seconds=time.perf_counter()-start,python=platform.python_version(),
                  numpy=np.__version__,scipy=scipy.__version__)
    manifest['raw_sha256']=hashlib.sha256((out/'main_raw.csv').read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    print(json.dumps({'done':True,'datasets':manifest['training_datasets'],'seconds':manifest['seconds']}),flush=True)


if __name__=='__main__':
    main()
