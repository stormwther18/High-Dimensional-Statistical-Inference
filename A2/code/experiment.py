"""Reproducible ridge-parameter averaging experiments. No test data in tuning."""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse, json, time, sys, platform, hashlib
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.linalg import toeplitz
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
METHODS = ['Ridge-GCV','Ridge-LOO','Ridge-5CV','Equal','GCV-MA','LOO-MA',
           '5CV-MA','5CV-MA-Reg','MinNorm','Cp-known','Oracle-Ridge','Oracle-MA']

def simplex_qp(E, linear=None, eta=0., target=None):
    """Minimize ||Ew||² / rows + linear'w + eta ||w-target||² on simplex."""
    m = E.shape[1]
    Q = E.T @ E / len(E)
    c = np.zeros(m) if linear is None else np.asarray(linear, float).copy()
    if eta:
        Q += eta*np.eye(m)
        c -= 2*eta*(np.ones(m)/m if target is None else target)
    scale = max(float(np.median(np.diag(Q))), 1e-12)
    Q, c = Q/scale, c/scale
    fun = lambda w: float(w@Q@w+c@w)
    jac = lambda w: 2*Q@w+c
    starts = [np.ones(m)/m, np.eye(m)[np.argmin(np.diag(Q)+c)]]
    best = None
    for start in starts:
        sol = minimize(fun, start, jac=jac, bounds=[(0.,1.)]*m,
                       constraints={'type':'eq','fun':lambda w:w.sum()-1,
                                    'jac':lambda w:np.ones(m)}, method='SLSQP',
                       options={'ftol':1e-11,'maxiter':400})
        w = np.maximum(sol.x,0); w /= w.sum()
        grad = jac(w)
        gap = float(w@grad-grad.min()) # first-order optimality gap on simplex
        candidate = (fun(w),w,gap,bool(sol.success),int(sol.nit))
        if best is None or candidate[0] < best[0]: best = candidate
        if sol.success and gap < 1e-6: break
    return best[1], {'gap':best[2], 'success':best[3], 'iterations':best[4]}

def ridge_path(X,y,lambdas):
    n = len(y)
    U,s,Vt = np.linalg.svd(X,full_matrices=False)
    z = U.T@y
    h = s[:,None]**2/(s[:,None]**2+n*lambdas[None,:])
    B = Vt.T @ (s[:,None]*z[:,None]/(s[:,None]**2+n*lambdas[None,:]))
    perp = y-U@z
    E = perp[:,None] + U@((1-h)*z[:,None])
    leverage = (U**2)@h
    df = h.sum(axis=0)
    cutoff = max(X.shape)*np.finfo(float).eps*s.max()
    invs = np.divide(1.,s,out=np.zeros_like(s),where=s>cutoff)
    minnorm = Vt.T@(z*invs)
    # Normalized leave-one-out objective uses (n-1)*lambda, rather than n*lambda.
    hloo=s[:,None]**2/(s[:,None]**2+(n-1)*lambdas[None,:])
    eloo=perp[:,None]+U@((1-hloo)*z[:,None])
    loo=eloo/(1-(U**2)@hloo)
    return B,E,leverage,df,minnorm,loo

def fit_feasible(X,y,lambdas,fold_order,eta=.01):
    """Only observed training data enter feasible tuning; beta/Sigma are absent."""
    t0=time.perf_counter()
    B,E,lev,df,minnorm,loo = ridge_path(X,y,lambdas)
    shared_time = time.perf_counter()-t0
    n,m = E.shape; a=1-df/n
    assert np.all(a>0) and np.all(1-lev>0)
    oof = np.empty_like(E)
    cv0=time.perf_counter()
    for val in np.array_split(fold_order,5):
        train = np.ones(n,dtype=bool); train[val]=False
        # Fold fitting uses n_train*lambda, preserving the normalized objective.
        b,*_ = ridge_path(X[train],y[train],lambdas)
        oof[val] = y[val,None]-X[val]@b
    cvtime=time.perf_counter()-cv0
    weights={}; diagnostics={}; timings={}
    for name,score in [('Ridge-GCV',np.mean((E/a)**2,axis=0)),
                       ('Ridge-LOO',np.mean(loo**2,axis=0)),
                       ('Ridge-5CV',np.mean(oof**2,axis=0))]:
        weights[name]=np.eye(m)[np.argmin(score)]
        timings[name]=shared_time+(cvtime if name=='Ridge-5CV' else 0.)
    weights['Equal']=np.ones(m)/m; timings['Equal']=shared_time; timings['MinNorm']=shared_time
    for name,res in [('GCV-MA',E/a),('LOO-MA',loo),('5CV-MA',oof),('5CV-MA-Reg',oof)]:
        ts=time.perf_counter()
        w,diag=simplex_qp(res,eta=eta*np.mean(y*y) if name=='5CV-MA-Reg' else 0.)
        if name=='GCV-MA':
            # v=a*w/(a'w), inverse map from the convex fractional reparameterization.
            w=w/a; w=w/w.sum()
        weights[name]=w; diagnostics[name]=diag
        timings[name]=shared_time+(cvtime if name.startswith('5CV') else 0.)+time.perf_counter()-ts
    return B,E,df,minnorm,weights,diagnostics,timings,oof,loo

@lru_cache(maxsize=64)
def population(p,cov):
    if cov=='iid': S=np.eye(p)
    elif cov=='toeplitz': S=toeplitz(.7**np.arange(p))
    elif cov=='anisotropic':
        vals=np.geomspace(.05,5,p); vals/=vals.mean(); S=np.diag(vals)
    else: raise ValueError(cov)
    L=np.linalg.cholesky(S)
    assert np.max(np.abs(S-L@L.T))<1e-10
    return S,L

def response(X,beta,rng,noise,nonlinear=0.):
    if noise=='gaussian': e=rng.normal(size=len(X))
    elif noise=='t3': e=rng.standard_t(3,size=len(X))/np.sqrt(3)
    elif noise=='hetero': e=rng.normal(size=len(X))*np.sqrt(.25+.75*X[:,0]**2)
    else: raise ValueError(noise)
    return X@beta+nonlinear*(X[:,0]**2-1)/np.sqrt(2)+e

def one_task(task):
    cfg,rep = task
    threadpool_limits(1)
    n,p=cfg['n'],cfg['p']; m=cfg['m']
    # Identical DGP and folds across grid/eta ablations, independent across DGPs.
    seed=np.random.SeedSequence([20261007,cfg['dgp_id'],rep])
    rb,rx,re,rf,rt = [np.random.default_rng(s) for s in seed.spawn(5)]
    S,L=population(p,cfg['cov'])
    b=rb.normal(size=p)
    if cfg['signal']=='sparse':
        b[:]=0; idx=rb.choice(p,max(1,p//10),replace=False); b[idx]=rb.normal(size=len(idx))
    elif cfg['signal'] in ('strong','weak'):
        _,V=np.linalg.eigh(S); k=max(1,p//10)
        V=V[:,-k:] if cfg['signal']=='strong' else V[:,:k]; b=V@rb.normal(size=k)
    b*=np.sqrt(cfg['snr']/(b@S@b))
    X=rx.normal(size=(n,p))@L.T
    y=response(X,b,re,cfg['noise'],cfg['nonlinear'])
    lambdas=np.logspace(cfg['lo'],cfg['hi'],m)
    B,E,df,mn,ws,diags,times,oof,loo=fit_feasible(X,y,lambdas,rf.permutation(n),cfg['eta'])
    # Privileged references enter only here, after all feasible methods are fixed.
    ts=time.perf_counter(); w,diag=simplex_qp(E,linear=2*df/n)
    ws['Cp-known']=w; diags['Cp-known']=diag; times['Cp-known']=time.perf_counter()-ts
    D=B-b[:,None]; riskvec=np.sum(D*(S@D),axis=0)
    ws['Oracle-Ridge']=np.eye(m)[np.argmin(riskvec)]
    ts=time.perf_counter(); w,diag=simplex_qp(np.sqrt(p)*(L.T@D))
    ws['Oracle-MA']=w; diags['Oracle-MA']=diag; times['Oracle-MA']=time.perf_counter()-ts
    # Common independently generated test sample. Analytical risk is the primary metric.
    Xt=rt.normal(size=(1000,p))@L.T
    yt=response(Xt,b,rt,cfg['noise'],cfg['nonlinear'])
    rows=[]; wrows=[]
    for method in METHODS:
        bh=mn if method=='MinNorm' else B@ws[method]
        delta=bh-b
        ex=float(delta@S@delta+cfg['nonlinear']**2)
        row={**cfg,'rep':rep,'method':method,'excess':ex,'risk':ex+1,
             'coef_mse':float(delta@delta),'test_mse':float(np.mean((yt-Xt@bh)**2)),
             'df':float(min(n,p) if method=='MinNorm' else df@ws[method]),
             'effective_m':float(1. if method=='MinNorm' else 1/np.sum(ws[method]**2)),
             'time_sec':times.get(method,np.nan),
             'qp_gap':diags.get(method,{}).get('gap',0.),
             'qp_success':diags.get(method,{}).get('success',True),
             'gcv_score':float(np.mean((E@ws[method])**2)/(1-df@ws[method]/n)**2) if method!='MinNorm' else np.nan,
             'loo_score':float(np.mean((loo@ws[method])**2)) if method!='MinNorm' else np.nan,
             'cv_score':float(np.mean((oof@ws[method])**2)) if method!='MinNorm' else np.nan}
        rows.append(row)
        if method in ws:
            for j,wj in enumerate(ws[method]):
                wrows.append({'config_id':cfg['config_id'],'rep':rep,'method':method,
                              'lambda':float(lambdas[j]),'weight':float(wj)})
    oracle=next(x['excess'] for x in rows if x['method']=='Oracle-MA')
    assert all(x['excess']>=oracle-1e-5*(1+x['excess']) for x in rows if x['method']!='MinNorm'), (cfg,rep,oracle,[(x['method'],x['excess']) for x in rows],diags)
    return rows,wrows

def configurations():
    configs=[]; dgpids={}
    def add(family,n,gamma,cov='iid',snr=1,signal='dense',noise='gaussian',nonlinear=0.,m=21,lo=-4.,hi=2.,eta=.01,reps=120):
        p=round(n*gamma)
        dgp=(n,p,cov,snr,signal,noise,nonlinear)
        if dgp not in dgpids: dgpids[dgp]=len(dgpids)
        configs.append(dict(config_id=len(configs),dgp_id=dgpids[dgp],family=family,n=n,p=p,
                            gamma=p/n,cov=cov,snr=snr,signal=signal,noise=noise,
                            nonlinear=nonlinear,m=m,lo=lo,hi=hi,eta=eta,reps=reps))
    for cov in ('iid','toeplitz'):
        for snr in (1,5):
            for g in (.5,.8,.95,1.,1.05,1.2,1.5,2.,3.): add('main',100,g,cov,snr)
    for signal in ('dense','strong','weak','sparse'): add('stress',100,1.5,'anisotropic',signal=signal,reps=150)
    for noise in ('hetero','t3'): add('stress',100,1.5,'toeplitz',noise=noise,reps=150)
    add('stress',100,1.5,'toeplitz',signal='sparse',reps=150)
    add('stress',100,1.5,'toeplitz',nonlinear=1.,reps=150)
    for n in (60,200):
        for cov in ('iid','toeplitz'):
            for g in (.8,1.5,3.): add('sample',n,g,cov,reps=100)
    for m in (5,11,21,41): add('grid',100,1.5,'toeplitz',m=m,reps=120)
    for lo,hi in ((-1.5,.5),(0.,3.)): add('grid',100,1.5,'toeplitz',lo=lo,hi=hi,reps=120)
    for eta in (0.,.001,.01,.1,1.): add('penalty',100,1.5,'toeplitz',eta=eta,reps=120)
    for gamma in (1.,1.5):
        for lo in (-4.,-3.,-2.,-1.,0.): add('boundary',100,gamma,'toeplitz',lo=lo,reps=150)
    return configs

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--quick',action='store_true'); args=ap.parse_args()
    configs=configurations()
    if args.quick:
        configs=[configs[0],configs[6],configs[36]]
        for c in configs:c['reps']=3
    out=ROOT/'results'; out.mkdir(exist_ok=True)
    prefix='quick_' if args.quick else ''
    (out/(prefix+'config.json')).write_text(json.dumps(configs,indent=2))
    tasks=[(c,r) for c in configs for r in range(c['reps'])]
    rows=[]; wrows=[]; start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for k,(rs,ws) in enumerate(pool.map(one_task,tasks,chunksize=5),1):
            rows.extend(rs); wrows.extend(ws)
            if k%100==0 or k==len(tasks): print(f'{k}/{len(tasks)} replications; {time.perf_counter()-start:.1f}s',flush=True)
    pd.DataFrame(rows).to_csv(out/(prefix+'raw.csv'),index=False)
    pd.DataFrame(wrows).to_csv(out/(prefix+'weights.csv'),index=False)
    meta=dict(seed=20261007,replications=len(tasks),scenarios=len(configs),methods=METHODS,
              elapsed_sec=time.perf_counter()-start,python=sys.version,platform=platform.platform(),
              numpy=np.__version__,scipy=__import__('scipy').__version__,
              code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/(prefix+'metadata.json')).write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2),flush=True)

if __name__=='__main__': main()
