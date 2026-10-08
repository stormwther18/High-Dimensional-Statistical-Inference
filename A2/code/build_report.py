"""Build manuscript, bibliography and tables from audited experiment outputs."""
import json,re
import pandas as pd
from experiment import ROOT

# Bibliographic facts verified against original PDFs / publisher and author pages.
# These are not claimed to be Google Scholar exports. URLs preserve provenance.
REFS=[
dict(key='breiman',author='Breiman, Leo',short='Breiman',year=1996,title='Stacked regressions',journal='Machine Learning',volume='24',pages='49--64',doi='10.1007/BF00117832',url='https://link.springer.com/article/10.1007/BF00117832'),
dict(key='buckland',author='Buckland, S. T. and Burnham, K. P. and Augustin, N. H.',short='Buckland et al.',year=1997,title='Model selection: An integral part of inference',journal='Biometrics',volume='53',pages='603--618',doi='10.2307/2533961',url='https://www.tandfonline.com/servlet/linkout?type=Scholix&url=https%3A%2F%2Fdx.doi.org%2F10.2307%252F2533961'),
dict(key='dw',author='Dobriban, Edgar and Wager, Stefan',short='Dobriban and Wager',year=2018,title='High-dimensional asymptotics of prediction: Ridge regression and classification',journal='The Annals of Statistics',volume='46',pages='247--279',doi='10.1214/17-AOS1549',url='https://projecteuclid.org/journals/annals-of-statistics/volume-46/issue-1/High-dimensional-asymptotics-of-prediction-Ridge-regression-and-classification/10.1214/17-AOS1549.full'),
dict(key='golub',author='Golub, Gene H. and Heath, Michael and Wahba, Grace',short='Golub et al.',year=1979,title='Generalized cross-validation as a method for choosing a good ridge parameter',journal='Technometrics',volume='21',pages='215--223',doi='10.1080/00401706.1979.10489751',url='https://pages.stat.wisc.edu/~wahba/ftp1/oldie/index.html'),
dict(key='hansen',author='Hansen, Bruce E.',short='Hansen',year=2007,title='Least squares model averaging',journal='Econometrica',volume='75',pages='1175--1189',doi='10.1111/j.1468-0262.2007.00785.x',url='https://users.ssc.wisc.edu/~behansen/papers/ecnmt_07.pdf'),
dict(key='jma',author='Hansen, Bruce E. and Racine, Jeffrey S.',short='Hansen and Racine',year=2012,title='Jackknife model averaging',journal='Journal of Econometrics',volume='167',pages='38--46',doi='10.1016/j.jeconom.2011.06.019',url='https://users.ssc.wisc.edu/~behansen/papers/joe_12.pdf'),
dict(key='hastie',author='Hastie, Trevor and Montanari, Andrea and Rosset, Saharon and Tibshirani, Ryan J.',short='Hastie et al.',year=2022,title='Surprises in high-dimensional ridgeless least squares interpolation',journal='The Annals of Statistics',volume='50',pages='949--986',doi='10.1214/21-AOS2133',url='https://arxiv.org/abs/1903.08560'),
dict(key='hoerl',author='Hoerl, Arthur E. and Kennard, Robert W.',short='Hoerl and Kennard',year=1970,title='Ridge regression: Biased estimation for nonorthogonal problems',journal='Technometrics',volume='12',pages='55--67',doi='10.1080/00401706.1970.10488634',url='https://www.tandfonline.com/doi/abs/10.1080/00401706.1970.10488634'),
dict(key='liu',author='Liu, Qingfeng and Okui, Ryo',short='Liu and Okui',year=2013,title='Heteroscedasticity-robust {C}p model averaging',journal='The Econometrics Journal',volume='16',pages='463--472',doi='10.1111/ectj.12009',url='https://academic.oup.com/ectj/article-abstract/16/3/463/5061230'),
dict(key='patil',author='Patil, Pratik and Wei, Yuting and Rinaldo, Alessandro and Tibshirani, Ryan',short='Patil et al.',year=2021,title='Uniform consistency of cross-validation estimators for high-dimensional ridge regression',journal='Proceedings of Machine Learning Research',volume='130',pages='3178--3186',url='https://proceedings.mlr.press/v130/patil21a.html'),
dict(key='wan',author='Wan, Alan T. K. and Zhang, Xinyu and Zou, Guohua',short='Wan et al.',year=2010,title='Least squares model averaging by {M}allows criterion',journal='Journal of Econometrics',volume='156',pages='277--283',doi='10.1016/j.jeconom.2009.10.030',url='https://www.sciencedirect.com/science/article/pii/S0304407609002838'),
dict(key='yao',author='Yao, Yuling and Vehtari, Aki and Simpson, Daniel and Gelman, Andrew',short='Yao et al.',year=2018,title='Using stacking to average {B}ayesian predictive distributions (with discussion)',journal='Bayesian Analysis',volume='13',pages='917--1003',doi='10.1214/17-BA1091',url='https://aaltodoc.aalto.fi/items/5baa1af7-7be4-40ab-9973-b24a02d36062'),
dict(key='yu',author='Yu, Dalei and Rao, Weidong and Yin, Xiaoxiao',short='Yu et al.',year=2018,title='岭回归中基于广义交叉核实法的最优模型平均估计',journal='系统科学与数学',volume='38',pages='652--661',doi='10.12341/jssms13408',url='https://sysmath.cjoe.ac.cn/jweb_xtkxysx/CN/10.12341/jssms13408'),
dict(key='zhao',author='Zhao, Shangwei and Liao, Jun and Yu, Dalei',short='Zhao et al.',year=2020,title='Model averaging estimator in ridge regression and its large sample properties',journal='Statistical Papers',volume='61',pages='1719--1739',doi='10.1007/s00362-018-1002-4',url='https://link.springer.com/article/10.1007/s00362-018-1002-4'),
dict(key='atanasov',author='Atanasov, Alexander and Zavatone-Veth, Jacob A. and Pehlevan, Cengiz',short='Atanasov et al.',year=2025,title='Risk and cross validation in ridge regression with correlated samples',journal='Proceedings of Machine Learning Research',volume='267',pages='1887--1930',url='https://proceedings.mlr.press/v267/atanasov25a.html')]

def bibliography():
    lines=[r'\begin{thebibliography}{99}']
    bib=[]
    for r in sorted(REFS,key=lambda r:r['author'].lower()):
        authors=r['author'].replace(' and ', '; ')
        title=r['title']; url='https://doi.org/'+r['doi'] if 'doi' in r else r['url']
        if r['key']=='yu':authors='喻达磊、饶炜东、尹潇潇'
        lines.append(r'\bibitem['+r['short']+'('+str(r['year'])+')]{'+r['key']+'}\n'+
                     authors+f" ({r['year']}). {title}. "+r'\textit{'+r['journal']+'}, '+r['volume']+', '+r['pages']+'. '+r'\url{'+url+'}.')
        fields=['author','title','journal','year','volume','pages','doi','url']
        bib.append('@article{'+r['key']+',\n'+',\n'.join(f'  {k} = {{{r[k]}}}' for k in fields if k in r)+'\n}')
    lines.append(r'\end{thebibliography}')
    (ROOT/'references.bib').write_text('\n\n'.join(bib),encoding='utf-8')
    (ROOT/'results'/'references_verified.json').write_text(json.dumps(REFS,ensure_ascii=False,indent=2))
    return '\n'.join(lines)

def main():
    s=pd.read_csv(ROOT/'results'/'summary.csv'); cp=pd.read_csv(ROOT/'results'/'paired.csv')
    raw=pd.read_csv(ROOT/'results'/'raw.csv'); claims=json.loads((ROOT/'results'/'claims.json').read_text())
    agg=claims['main_matched_comparisons']; repl={}
    for k,name in [('ORACLE','Oracle-MA'),('LOO','LOO-MA'),('CV','5CV-MA'),('GCV','GCV-MA')]:
        repl[k]=f"{agg[name]['mean_scenario_ratio']:.3f}"
        repl[k+'_RANGE']=f"{agg[name]['ratio_min']:.3f}--{agg[name]['ratio_max']:.3f}"
        repl[k+'_BETTER']=str(agg[name]['better_scenarios'])
    repl.update(RUNS=f"{claims['total_replications']:,}",DISTINCT=f"{claims['distinct_data_replications']:,}",
                SCENARIOS=str(claims['settings']),ROWS=f"{len(raw):,}",GAP=f"${claims['max_qp_gap']:.3g}$")
    def value(cov,gamma,method):
        return s[(s.family=='main')&(s['cov']==cov)&(s.snr==1)&(s.gamma==gamma)&(s.method==method)]['mean'].iloc[0]
    for prefix,cov,gamma in [('IID_ONE','iid',1.),('TOEP','toeplitz',1.5)]:
        for suffix,m in [('RIDGE','Ridge-GCV'),('GCV','GCV-MA'),('LOO','LOO-MA')]:repl[prefix+'_'+suffix]=f'{value(cov,gamma,m):.3f}'
    for key,m in [('GRID_LOO','LOO-MA'),('GRID_ORACLE','Oracle-MA')]:
        q=s[(s.family=='grid')&(s.method==m)&(s.lo==-4)].sort_values('m');repl[key]='、'.join(f'{x:.3f}' for x in q['mean'])
    pen=s[(s.family=='penalty')&(s.method=='5CV-MA-Reg')].sort_values('eta')
    repl['PEN_RISK']='、'.join(f'{x:.3f}' for x in pen['mean']);repl['PEN_M0']=f'{pen.mean_effective_m.iloc[0]:.2f}';repl['PEN_M1']=f'{pen.mean_effective_m.iloc[-1]:.2f}'
    bd=raw[(raw.family=='boundary')&(raw.gamma==1)&(raw.method=='GCV-MA')]
    repl['BOUND_GAMMA']='1'
    for suffix,lo in [('0',-4),('1',0)]:
        t=bd[bd.lo==lo];repl['BOUND_RISK'+suffix]=f'{t.excess.mean():.3f}';repl['BOUND_CAL'+suffix]=f'{t.gcv_score.mean()/t.risk.mean():.3f}'
    methods=['Ridge-GCV','Equal','GCV-MA','LOO-MA','5CV-MA','Oracle-MA']
    lines=[r'\begin{table}[!htbp]\centering\small',r'\caption{信噪比 1 的代表主场景。单元格为平均超额风险（Monte Carlo 标准误），每场景 120 次重复。Oracle-MA 仅作不可实施的诊断参照。}\label{tab:main}',r'\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{llrrrrrr}\toprule',r'设计 & $p/n$ & Ridge-GCV & Equal & GCV-MA & LOO-MA & 5CV-MA & Oracle-MA\\\midrule']
    for cov in ['iid','toeplitz']:
        for g in [.5,1.,1.5,3.]:
            sub=s[(s.family=='main')&(s['cov']==cov)&(s.snr==1)&(s.gamma==g)].set_index('method')
            vals=[r'\shortstack{'+f"{sub.loc[m,'mean']:.3f}"+r'\\('+f"{sub.loc[m,'mcse']:.3f}"+')}' for m in methods]
            lines.append(('独立' if cov=='iid' else '相关')+f' & {g:g} & '+' & '.join(vals)+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{table}'];repl['TABLE_MAIN']='\n'.join(lines)
    lines=[r'\begin{table}[!htbp]\centering\small',r'\caption{相关设计、信噪比 1 的同准则成对比较。差值为平均器减去单一选参器，括号为成对差值的 Monte Carlo 标准误；比值区间使用 2000 次成对 bootstrap。}\label{tab:paired}',r'\begin{tabular}{llrrl}\toprule',r'$p/n$ & 平均器／选参器 & 风险比 & 风险差（标准误） & 比值区间\\\midrule']
    for g in [.5,1.,1.5,3.]:
        for a,b in [('GCV-MA','Ridge-GCV'),('LOO-MA','Ridge-LOO'),('5CV-MA','Ridge-5CV')]:
            q=cp[(cp.family=='main')&(cp['cov']=='toeplitz')&(cp.snr==1)&(cp.gamma==g)&(cp.averager==a)].iloc[0]
            lines.append(f"{g:g} & {a}/{b} & {q.ratio:.3f} & {q.paired_delta:.3f} ({q.paired_mcse:.3f}) & [{q.ci_low:.3f}, {q.ci_high:.3f}]"+r'\\')
    lines += [r'\bottomrule\end{tabular}\end{table}'];repl['TABLE_PAIRED']='\n'.join(lines)
    repl['BIBLIOGRAPHY']=bibliography()
    text=(ROOT/'report_template.tex').read_text()
    for key,v in repl.items():text=text.replace('@@'+key+'@@',v)
    assert not re.search(r'@@\w+@@',text)
    cited=set()
    for v in re.findall(r'\\cite[tp](?:\[[^]]*\])?\{([^}]*)\}',text):cited.update(v.split(','))
    assert cited=={r['key'] for r in REFS}
    (ROOT/'report.tex').write_text(text,encoding='utf-8')
    (ROOT/'results'/'report_values.json').write_text(json.dumps({k:v for k,v in repl.items() if not k.startswith('TABLE') and k!='BIBLIOGRAPHY'},ensure_ascii=False,indent=2))
    print(f'Built report.tex with {len(cited)} verified references and 10 data-driven figures.')

if __name__=='__main__':main()
