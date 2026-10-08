"""Numerical checks of invariants that could invalidate the scientific comparison."""
import json
import numpy as np
from experiment import ridge_path, simplex_qp, fit_feasible, population, ROOT
from scipy.optimize import minimize

def main():
    rng=np.random.default_rng(119)
    checks={}
    for p in (50,150,600):
        S,L=population(p,'toeplitz'); assert np.max(abs(S-L@L.T))<1e-10
    checks['covariance_generation']='passed at p=50,150,600'
    for n,p in [(23,9),(17,27),(43,57)]:
        X=rng.normal(size=(n,p)); y=rng.normal(size=n); lam=np.logspace(-3,1,7)
        B,E,lev,df,mn,loo=ridge_path(X,y,lam)
        direct=np.column_stack([np.linalg.solve(X.T@X+n*l*np.eye(p),X.T@y) for l in lam])
        assert np.max(np.abs(B-direct))<1e-9
        brute=np.zeros_like(loo)
        for i in range(n):
            take=np.arange(n)!=i
            bd=np.column_stack([np.linalg.solve(X[take].T@X[take]+(n-1)*l*np.eye(p),X[take].T@y[take]) for l in lam])
            brute[i]=y[i]-X[i]@bd
        err=float(np.max(np.abs(brute-loo))); assert err<1e-9
        assert np.max(np.abs(mn-np.linalg.pinv(X)@y))<1e-9
        a=1-df/n
        v,diag=simplex_qp(E/a); w=v/a; w/=w.sum()
        score=lambda t:np.mean((E@t)**2)/(a@t)**2
        raw=minimize(score,np.ones(len(lam))/len(lam),bounds=[(0,1)]*len(lam),
                     constraints={'type':'eq','fun':lambda t:t.sum()-1},method='SLSQP',
                     options={'ftol':1e-12,'maxiter':1000})
        assert abs(score(w)-score(raw.x))<1e-7
        assert diag['gap']<1e-5
        # All feasible selection functions accept observed data only.
        fs=fit_feasible(X,y,lam,rng.permutation(n))
        assert all(abs(t.sum()-1)<1e-10 and t.min()>-1e-10 for t in fs[4].values())
        checks[f'n{n}_p{p}']={'loo_max_error':err,'gcv_objective_difference':float(abs(score(w)-score(raw.x))), 'simplex_pass':True}
    f=ROOT/'results'/'raw.csv'
    if f.exists():
        import pandas as pd
        d=pd.read_csv(f)
        assert np.isfinite(d[['excess','risk','coef_mse','test_mse','qp_gap']]).all().all()
        checks['full_run']={'rows':len(d),'max_qp_gap':float(d.qp_gap.max()),
                            'solver_non_success':int((~d.qp_success).sum())}
        wide=d.pivot(index=['config_id','rep'],columns='method',values='excess')
        assert np.all(wide['Oracle-MA']<=wide['Oracle-Ridge']+1e-5*(1+wide['Oracle-Ridge']))
        checks['full_run']['oracle_dominance_pass']=True
        # Independent test MSE must agree with analytic total risk at aggregate scale.
        gaussian=d[(d.family=='main')&(d.method!='MinNorm')]
        cal=gaussian.groupby('method')[['test_mse','risk']].mean()
        ratios=cal.test_mse/cal.risk
        assert ((ratios>.95)&(ratios<1.05)).all()
        checks['full_run']['independent_test_to_analytic_risk']=ratios.to_dict()
        w=pd.read_csv(ROOT/'results'/'weights.csv')
        assert np.max(abs(w.groupby(['config_id','rep','method']).weight.sum()-1))<1e-10
        assert (w.weight>=0).all()
    (ROOT/'results'/'verification.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
