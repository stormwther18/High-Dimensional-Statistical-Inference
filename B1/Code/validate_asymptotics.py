"""Independent MP/GCV and bounded-spectrum deterministic-equivalent checks."""
from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]


def trace_limit(mu,gamma,lambdas):
    lam=np.asarray(lambdas,float)
    low=np.zeros_like(lam);high=1/lam
    for _ in range(90):
        v=(low+high)/2
        f=lam*v+gamma*np.mean(mu[:,None]*v/(1+mu[:,None]*v),axis=0)-1
        low=np.where(f<0,v,low);high=np.where(f>=0,v,high)
    v=(low+high)/2
    den=1+mu[:,None]*v
    vdot=-v/(lam+gamma*np.mean(mu[:,None]/den**2,axis=0))
    t=np.mean(mu[:,None]/den,axis=0)/lam
    tdot=-t/lam-vdot*np.mean(mu[:,None]**2/den**2,axis=0)/lam
    return t,tdot


def risk_matrix_limit(t,tdot,lam,gamma):
    K=len(lam);B=np.empty((K,K));V=np.empty((K,K))
    for j in range(K):
        for k in range(K):
            if j==k:
                B[j,k]=-tdot[j];V[j,k]=t[j]+lam[j]*tdot[j]
            else:
                B[j,k]=(t[j]-t[k])/(lam[k]-lam[j])
                V[j,k]=(lam[k]*t[k]-lam[j]*t[j])/(lam[k]-lam[j])
    return np.outer(lam,lam)*B+gamma*V


def mp_limits(lam,pi):
    lam=np.asarray(lam,float);pi=np.asarray(pi,float)
    t=(np.sqrt(1+4/lam)-1)/2
    td=-1/(lam**2*np.sqrt(1+4/lam))
    C=risk_matrix_limit(t,td,lam,1.)
    # Numerator integral of (s+1)u(s)^2.
    B=np.array([[-td[j] if j==k else (t[j]-t[k])/(lam[k]-lam[j])
                 for k in range(len(lam))] for j in range(len(lam))])
    V=C-np.outer(lam,lam)*B
    true=1+pi@C@pi
    aggregate=pi@(np.outer(lam,lam)*(B+V))@pi/(pi@(lam*t))**2
    return float(true),float(aggregate)


def mp_validation(out,reps,seed):
    fields=['n','rep','pair_id','method','estimate','risk','exact_prior_risk',
            'limit_risk','limit_gcv']
    pairs=[np.array([.1,1.]),np.array([.01,10.])]
    pi=np.array([.5,.5])
    with (out/'mp_validation.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for n in (100,300,600):
            for rep in range(reps):
                rng=np.random.default_rng(np.random.SeedSequence([seed,1,n,rep]))
                X=rng.normal(size=(n,n));beta=rng.normal(size=n)/np.sqrt(n)
                y=X@beta+rng.normal(size=n)
                s,U=eigh(X@X.T/n,check_finite=False)
                s=np.maximum(s,0.);uy=U.T@y;V=X.T@U/n
                for pair_id,lam in enumerate(pairs):
                    filt=lam/(s[:,None]+lam)
                    residual=U@(uy[:,None]*filt)
                    coef=V@(uy[:,None]/(s[:,None]+lam))
                    d=filt.mean(axis=0)
                    risk=1+float(np.sum((coef@pi-beta)**2))
                    avgq=(1/(s[:,None]+lam))@pi
                    prior=1+float(np.mean((filt@pi)**2+s*avgq**2))
                    naive=float(np.mean((residual@pi)**2)/(d@pi)**2)
                    cross=float(np.mean(((residual/d)@pi)**2))
                    rlim,glim=mp_limits(lam,pi)
                    for method,estimate in [('gcv_aggregate',naive),('gcv_cross',cross)]:
                        writer.writerow(dict(n=n,rep=rep,pair_id=pair_id,method=method,
                                             estimate=estimate,risk=risk,exact_prior_risk=prior,
                                             limit_risk=rlim,limit_gcv=glim))
            f.flush();print(f'MP n={n}: {reps} datasets',flush=True)


def de_validation(out,reps,seed):
    fields=['model','n','gamma','rep','method','exact_prior_risk','de_risk']
    lam=np.array([.1,1.,10.]);weights={'fixed_average':np.array([.2,.5,.3]),
                                    'single_lambda_1':np.array([0.,1.,0.])}
    with (out/'de_validation.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for model_id,model in enumerate(('identity','two_block')):
            for n in (100,200,400):
                for gamma in (.8,1.,1.5):
                    p=round(n*gamma)
                    mu=np.ones(p) if model=='identity' else np.r_[np.full(p//2,.5),np.full(p-p//2,1.5)]
                    t,td=trace_limit(mu,p/n,lam)
                    Climit=risk_matrix_limit(t,td,lam,p/n)
                    for rep in range(reps):
                        rng=np.random.default_rng(np.random.SeedSequence([seed,2,model_id,n,p,rep]))
                        X=rng.normal(size=(n,p))*np.sqrt(mu)
                        if p<=n:
                            s,V=eigh(X.T@X/n,check_finite=False)
                        else:
                            s,U=eigh(X@X.T/n,check_finite=False)
                            s=np.maximum(s,0.)
                            keep=s>np.finfo(float).eps*max(n,p)*max(s.max(),1)
                            s=s[keep];V=(X.T@U[:,keep])/np.sqrt(n*s)
                        s=np.maximum(s,0.)
                        spectral=np.sum(mu[:,None]*V**2,axis=0)
                        null_mass=max(float(mu.sum()-spectral.sum()),0.)
                        for method,w in weights.items():
                            q=(1/(s[:,None]+lam))@w
                            shrink=(lam/(s[:,None]+lam))@w
                            exact=1+(spectral@shrink**2+null_mass)/p+(spectral*s)@q**2/n
                            pred=1+w@Climit@w
                            writer.writerow(dict(model=model,n=n,gamma=p/n,rep=rep,method=method,
                                                 exact_prior_risk=float(exact),de_risk=float(pred)))
                    f.flush();print(f'DE {model} n={n} gamma={gamma}: {reps} datasets',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mp-repetitions',type=int,default=100)
    ap.add_argument('--de-repetitions',type=int,default=40)
    ap.add_argument('--seed',type=int,default=202611041)
    ap.add_argument('--output',type=Path,default=ROOT/'Results')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    with threadpool_limits(limits=1):
        mp_validation(args.output,args.mp_repetitions,args.seed)
        de_validation(args.output,args.de_repetitions,args.seed)
    info={'mp_training_datasets':3*args.mp_repetitions,
          'de_training_datasets':18*args.de_repetitions,'root_seed':args.seed,
          'mp_repetitions':args.mp_repetitions,'de_repetitions':args.de_repetitions,
          'seconds':time.perf_counter()-start}
    (args.output/'validation_manifest.json').write_text(json.dumps(info,indent=2))


if __name__=='__main__': main()
