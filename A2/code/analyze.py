"""All figure data and report numbers are computed from raw.csv, never hard-coded."""
import os
os.environ.setdefault('MPLCONFIGDIR',str(__import__('pathlib').Path(__file__).resolve().parents[1]/'.mpl-cache'))
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import ROOT, METHODS

plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
                     'font.size':9,'axes.labelsize':10,'legend.fontsize':8,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'ps.fonttype':42,'savefig.bbox':'tight'})
COLORS=['#0072B2','#56B4E9','#009E73','#999999','#D55E00','#CC79A7','#E69F00','#000000','#7D3C98','#80A086','#333333','#6B705C']
STYLE={m:dict(color=COLORS[i],linestyle=['-','--',':','-.'][i%4],marker=['o','s','^','D'][i%4],markersize=3,linewidth=1.2) for i,m in enumerate(METHODS)}

def save(fig,name):
    fig.savefig(ROOT/'figures'/f'{name}.pdf')
    fig.savefig(ROOT/'figures'/f'{name}.png',dpi=180)
    plt.close(fig)

def boot_ratio(a,b,rng):
    idx=rng.integers(len(a),size=(2000,len(a)))
    vals=a[idx].mean(axis=1)/b[idx].mean(axis=1)
    return np.quantile(vals,[.025,.975])

def main():
    d=pd.read_csv(ROOT/'results'/'raw.csv'); cfg=pd.read_json(ROOT/'results'/'config.json')
    weights=pd.read_csv(ROOT/'results'/'weights.csv')
    rng=np.random.default_rng(701)
    summaries=[]; comparisons=[]
    for (cid,method),g in d.groupby(['config_id','method']):
        x=g.excess.to_numpy(); base=d[(d.config_id==cid)&(d.method=='Ridge-GCV')].sort_values('rep').excess.to_numpy()
        x=g.sort_values('rep').excess.to_numpy(); lo,hi=boot_ratio(x,base,rng)
        z={'config_id':cid,'method':method,'mean':x.mean(),'mcse':x.std(ddof=1)/np.sqrt(len(x)),
           'median':np.median(x),'q25':np.quantile(x,.25),'q75':np.quantile(x,.75),
           'ratio_ridge_gcv':x.mean()/base.mean(),'ratio_low':lo,'ratio_high':hi,
           'mean_coef_error':g.coef_mse.mean(),'mean_test_mse':g.test_mse.mean(),
           'mean_df':g.df.mean(),'mean_effective_m':g.effective_m.mean(),
           'median_time_ms':1000*g.time_sec.median(),'max_qp_gap':g.qp_gap.max(),
           'mean_total_risk':g.risk.mean()}
        summaries.append(z)
    s=pd.DataFrame(summaries).merge(cfg,on='config_id'); s.to_csv(ROOT/'results'/'summary.csv',index=False)
    pairs=[('GCV-MA','Ridge-GCV'),('LOO-MA','Ridge-LOO'),('5CV-MA','Ridge-5CV'),('5CV-MA-Reg','Ridge-5CV'),('Oracle-MA','Oracle-Ridge')]
    for cid,cg in d.groupby('config_id'):
        wide=cg.pivot(index='rep',columns='method',values='excess')
        for a,b in pairs:
            x,y=wide[a].to_numpy(),wide[b].to_numpy(); diff=x-y
            lo,hi=boot_ratio(x,y,rng)
            comparisons.append(dict(config_id=cid,averager=a,selector=b,ratio=x.mean()/y.mean(),
                                    ci_low=lo,ci_high=hi,paired_delta=diff.mean(),
                                    paired_mcse=diff.std(ddof=1)/np.sqrt(len(diff)),win_rate=float((x<y).mean())))
    cp=pd.DataFrame(comparisons).merge(cfg,on='config_id'); cp.to_csv(ROOT/'results'/'paired.csv',index=False)
    # 1: four DGP groups, mean risk with pointwise Monte Carlo uncertainty.
    fig,axs=plt.subplots(2,2,figsize=(8.1,5.4),sharex=True)
    selected=['Ridge-GCV','Equal','GCV-MA','LOO-MA','5CV-MA','Oracle-MA']
    for ax,(cov,snr) in zip(axs.flat,[(c,t) for c in ('iid','toeplitz') for t in (1,5)]):
        sub=s[(s.family=='main')&(s['cov']==cov)&(s.snr==snr)]
        for m in selected:
            g=sub[sub.method==m].sort_values('gamma'); ax.plot(g.gamma,g['mean'],label=m,**STYLE[m])
            ax.fill_between(g.gamma,np.maximum(g['mean']-1.96*g.mcse,1e-9),g['mean']+1.96*g.mcse,color=STYLE[m]['color'],alpha=.08)
        ax.axvline(1,color='grey',lw=.7,ls=':'); ax.set_yscale('log')
        ax.set_xlabel(r'$p/n$'); ax.set_ylabel('Mean excess prediction risk')
        ax.text(.03,.96,f'{cov}, SNR={snr}',transform=ax.transAxes,va='top')
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=6,bbox_to_anchor=(.5,1.03))
    fig.tight_layout(); save(fig,'fig01_main_risk')
    # 2: interpolation instability is reported through medians and IQR, not a hidden cap.
    fig,axs=plt.subplots(1,2,figsize=(8.1,2.7),sharey=True)
    for ax,cov in zip(axs,('iid','toeplitz')):
        for m in ['MinNorm','Ridge-GCV','LOO-MA']:
            g=s[(s.family=='main')&(s['cov']==cov)&(s.snr==1)&(s.method==m)].sort_values('gamma')
            ax.plot(g.gamma,g['median'],label=m,**STYLE[m]); ax.fill_between(g.gamma,g.q25,g.q75,color=STYLE[m]['color'],alpha=.14)
        ax.axvline(1,color='grey',ls=':',lw=.7); ax.set_yscale('log'); ax.set_xlabel(r'$p/n$'); ax.text(.03,.94,cov,transform=ax.transAxes,va='top')
    axs[0].set_ylabel('Median excess risk (IQR band)'); axs[1].legend(frameon=False); fig.tight_layout(); save(fig,'fig02_interpolation')
    # 3: fair matched-selection comparisons, pointwise paired-bootstrap confidence intervals.
    fig,axs=plt.subplots(2,2,figsize=(8.1,5.3),sharex=True)
    for ax,(cov,snr) in zip(axs.flat,[(c,t) for c in ('iid','toeplitz') for t in (1,5)]):
        sub=cp[(cp.family=='main')&(cp['cov']==cov)&(cp.snr==snr)]
        for m in ['GCV-MA','LOO-MA','5CV-MA','Oracle-MA']:
            g=sub[sub.averager==m].sort_values('gamma'); ax.plot(g.gamma,g.ratio,label=m,**STYLE[m])
            ax.fill_between(g.gamma,g.ci_low,g.ci_high,color=STYLE[m]['color'],alpha=.08)
        ax.axhline(1,color='grey',lw=.8); ax.set_xlabel(r'$p/n$'); ax.set_ylabel('Risk ratio to matched single ridge'); ax.text(.03,.94,f'{cov}, SNR={snr}',transform=ax.transAxes,va='top')
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=4,bbox_to_anchor=(.5,1.03)); fig.tight_layout(); save(fig,'fig03_matched')
    # 4: robustness, each of eight settings has 150 independent replicates.
    stress=cfg[cfg.family=='stress']; labels=['Aniso: dense','Aniso: strong','Aniso: weak','Aniso: sparse','Heteroskedastic','t(3) noise','Toeplitz: sparse','Nonlinear mean']
    fig,ax=plt.subplots(figsize=(8.1,3.8)); xs=np.arange(len(stress))
    for offset,m in zip([-.21,-.07,.07,.21],['GCV-MA','LOO-MA','5CV-MA-Reg','Equal']):
        g=s[(s.family=='stress')&(s.method==m)].sort_values('config_id')
        # t(3) has infinite fourth moment: no conventional mean-risk CI is claimed.
        yerr=np.vstack([g.ratio_ridge_gcv-g.ratio_low,g.ratio_high-g.ratio_ridge_gcv]); yerr[:,5]=np.nan
        ax.errorbar(xs+offset,g.ratio_ridge_gcv,yerr=yerr,label=m,fmt=STYLE[m]['marker'],color=STYLE[m]['color'],ms=4,capsize=2,lw=.8)
    ax.set_xticks(xs,labels,rotation=25,ha='right'); ax.axhline(1,color='grey',lw=.8); ax.set_ylabel('Risk ratio to Ridge-GCV'); ax.legend(frameon=False,ncol=4); fig.tight_layout(); save(fig,'fig04_robustness')
    # 5: sample size, same gamma and SNR; risk estimated analytically.
    fig,axs=plt.subplots(1,2,figsize=(8.1,3.1),sharey=True)
    for ax,cov in zip(axs,('iid','toeplitz')):
        sub=s[(s.family.isin(['main','sample']))&(s['cov']==cov)&(s.snr==1)&(s.gamma==1.5)]
        for m in ['Ridge-GCV','GCV-MA','LOO-MA','5CV-MA-Reg','Oracle-MA']:
            g=sub[sub.method==m].sort_values('n'); ax.errorbar(g.n,g['mean'],yerr=1.96*g.mcse,label=m,**STYLE[m],capsize=2)
        ax.set_xlabel('Training sample size n'); ax.set_ylabel('Mean excess prediction risk'); ax.text(.03,.96,cov,transform=ax.transAxes,va='top')
    axs[1].legend(frameon=False,ncol=1,fontsize=7); fig.tight_layout(); save(fig,'fig05_sample_size')
    # 6: candidate grid ablation; models trained on exactly the same seeds.
    grid=s[s.family=='grid'].copy(); gids=sorted(grid.config_id.unique()); glabel=['M=5','M=11','M=21','M=41','Narrow','Overshrink']
    fig,axs=plt.subplots(1,2,figsize=(8.1,3.1))
    for m in ['Ridge-GCV','Equal','GCV-MA','LOO-MA','5CV-MA-Reg','Oracle-MA']:
        g=grid[grid.method==m].sort_values('config_id'); axs[0].plot(range(len(g)),g['mean'],label=m,**STYLE[m]); axs[1].plot(range(len(g)),g.mean_effective_m,label=m,**STYLE[m])
    for ax in axs:ax.set_xticks(range(len(gids)),glabel,rotation=25,ha='right')
    axs[0].set_ylabel('Mean excess prediction risk'); axs[1].set_ylabel(r'Effective candidates $1/\sum w_m^2$'); axs[0].legend(frameon=False,fontsize=6.5,ncol=2); fig.tight_layout(); save(fig,'fig06_grid')
    # 7: weight paths. They show allocation, not uniquely identified scientific effects.
    wc=weights.merge(cfg[['config_id','family','cov','snr','gamma','eta']],on='config_id')
    fig,axs=plt.subplots(1,3,figsize=(8.1,3.1),sharey=True)
    for ax,m in zip(axs,['GCV-MA','LOO-MA','5CV-MA-Reg']):
        g=wc[(wc.family=='main')&(wc['cov']=='toeplitz')&(wc.snr==1)&(wc.method==m)]
        mat=g.groupby(['lambda','gamma']).weight.mean().unstack().sort_index()
        im=ax.imshow(mat.to_numpy(),origin='lower',aspect='auto',vmin=0,vmax=.55,cmap='viridis',extent=[-.5,len(mat.columns)-.5,-4,2])
        ax.set_xticks(range(len(mat.columns)),[str(x) for x in mat.columns],rotation=55); ax.set_xlabel(r'$p/n$'); ax.text(.03,.97,m,transform=ax.transAxes,color='white',va='top')
    axs[0].set_ylabel(r'$\log_{10}\lambda$'); fig.colorbar(im,ax=list(axs),label='Mean weight',fraction=.025,pad=.02); fig.subplots_adjust(bottom=.22,wspace=.15,right=.9); save(fig,'fig07_weights')
    # 8: penalty ablation (all settings share data, fold allocations, and candidate grid).
    fig,axs=plt.subplots(1,2,figsize=(8.1,2.9))
    pen=s[(s.family=='penalty')&(s.method=='5CV-MA-Reg')].sort_values('eta')
    xp=np.arange(len(pen)); labels=[str(x) for x in pen.eta]
    axs[0].errorbar(xp,pen['mean'],yerr=1.96*pen.mcse,marker='o',color='#0072B2',capsize=3)
    base=s[(s.family=='main')&(s['cov']=='toeplitz')&(s.snr==1)&(s.gamma==1.5)&(s.method=='Ridge-GCV')]['mean'].iloc[0]
    axs[0].axhline(base,color='grey',ls='--',label='Ridge-GCV'); axs[0].legend(frameon=False)
    axs[1].plot(xp,pen.mean_effective_m,'o-',color='#D55E00')
    for ax in axs:ax.set_xticks(xp,labels);ax.set_xlabel(r'Penalty scale $\eta$')
    axs[0].set_ylabel('Mean excess prediction risk'); axs[1].set_ylabel('Mean effective candidates'); fig.tight_layout(); save(fig,'fig08_penalty')
    # 9: runtime, all operational methods, on main runs only.
    runtime=d[(d.family=='main')&(d.method.isin(METHODS[:9]))].groupby('method').time_sec.median()*1000
    fig,ax=plt.subplots(figsize=(8.1,2.8)); ax.bar(range(len(runtime)),runtime.to_numpy(),color=[STYLE[m]['color'] for m in runtime.index]); ax.set_xticks(range(len(runtime)),runtime.index,rotation=25,ha='right'); ax.set_ylabel('Median fitting + tuning time (ms)'); fig.tight_layout(); save(fig,'fig09_runtime')
    # 10: diagnose the lower ridge boundary using both actual risk and estimated risk.
    fig,axs=plt.subplots(2,2,figsize=(8.1,5.1),sharex=True)
    for col,gamma in enumerate((1.,1.5)):
        for m in ['Ridge-GCV','GCV-MA','LOO-MA','Oracle-MA']:
            g=s[(s.family=='boundary')&(s.gamma==gamma)&(s.method==m)].sort_values('lo')
            axs[0,col].plot(g.lo,g['mean'],label=m,**STYLE[m])
        bd=d[(d.family=='boundary')&(d.gamma==gamma)]
        for m in ['Ridge-GCV','GCV-MA','LOO-MA']:
            q=bd[bd.method==m].groupby('lo')[['gcv_score','risk']].mean()
            axs[1,col].plot(q.index,q.gcv_score/q.risk,label=m,**STYLE[m])
        axs[0,col].text(.03,.96,fr'$p/n={gamma}$',transform=axs[0,col].transAxes,va='top')
        axs[0,col].set_yscale('log'); axs[0,col].set_ylabel('Mean excess prediction risk')
        axs[1,col].axhline(1,color='grey',lw=.8); axs[1,col].set_ylabel('Mean GCV score / mean total risk')
        axs[1,col].set_ylim(0,1.1)
        axs[1,col].set_xlabel(r'Lower boundary $\log_{10}\lambda_{\min}$')
    axs[0,1].legend(frameon=False,fontsize=7); fig.tight_layout(); save(fig,'fig10_boundary_calibration')
    # Main comparisons are equally weighted across 36 predeclared settings.
    mainpairs=cp[cp.family=='main']; aggregate={}
    for m in mainpairs.averager.unique():
        g=mainpairs[mainpairs.averager==m]
        aggregate[m]={'mean_scenario_ratio':float(g.ratio.mean()),'ratio_min':float(g.ratio.min()),
                      'ratio_max':float(g.ratio.max()),'better_scenarios':int((g.ratio<1).sum()),
                      'ci_below_one':int((g.ci_high<1).sum()),'ci_above_one':int((g.ci_low>1).sum())}
    claims={'main_matched_comparisons':aggregate,'total_replications':int(len(d)/len(METHODS)),
            'distinct_data_replications':int(cfg.groupby('dgp_id').reps.max().sum()),
            'settings':len(cfg),'stress_labels':dict(zip(stress.config_id.astype(str),labels if False else ['Aniso dense','Aniso strong','Aniso weak','Aniso sparse','Hetero','t3','Toeplitz sparse','Nonlinear'])),
            'max_qp_gap':float(d.qp_gap.max()),'solver_non_success':int((~d.qp_success).sum())}
    (ROOT/'results'/'claims.json').write_text(json.dumps(claims,indent=2))
    print(json.dumps(claims,indent=2))

if __name__=='__main__':main()
