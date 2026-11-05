"""Build the standalone LaTeX research report from verified experiment CSVs."""
from __future__ import annotations
import csv
import json
import subprocess
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'Report'
def load(name):
    with (ROOT/'Results'/name).open() as f:return list(csv.DictReader(f))
def number(row,key):return float(row[key])
def fmt(x,digits=4):
    x=float(x)
    if not np.isfinite(x):return '--'
    if abs(x)>=10000:
        m,e=f'{x:.2e}'.split('e');return rf'{m}\times10^{{{int(e)}}}'
    return f'{x:.{digits}f}'
def sci(x):
    m,e=f'{float(x):.2e}'.split('e')
    return rf'{m}\times10^{{{int(e)}}}'


def main():
    main=load('main_summary.csv');gains=load('oracle_gains.csv');cal=load('calibration.csv')
    mp=load('mp_summary.csv');de=load('de_summary.csv');pairs=load('paired_risk.csv')
    verify=json.loads((ROOT/'Results/verification.json').read_text())
    manifest=json.loads((ROOT/'Results/manifest.json').read_text())
    validation=json.loads((ROOT/'Results/validation_manifest.json').read_text())
    assert verify['passed'] and verify['solver_failures']==0
    keyed={(r['config_id'],r['method']):r for r in main}
    gainkey={(r['config_id'],r['method']):r for r in gains}
    calkey={(r['config_id'],r['method']):r for r in cal}
    configs=manifest['configurations']
    weak=[r for r in gains if float(r['delta'])<0 and r['method']=='oracle_convex']
    strong=[r for r in gains if float(r['delta'])>0 and r['method']=='oracle_signed_L3']
    weakrange=(min(number(r,'mean_conditional_gain') for r in weak),max(number(r,'mean_conditional_gain') for r in weak))
    strongrange=(min(number(r,'mean_conditional_gain') for r in strong),max(number(r,'mean_conditional_gain') for r in strong))
    maintext=rf"""
\begin{{center}}
{{\LARGE\bfseries 多岭参数平均的风险、信号谱对齐与选权代价\par}}
\vspace{{6pt}}
{{\large B 计划首轮研究报告与复现实验\par}}
\vspace{{4pt}}{{\small 2026 年 11 月 4 日\quad 代码、原始结果与数学证明配套交付}}
\end{{center}}
\begin{{abstract}}
本研究从同一训练数据、多个正则参数的岭回归平均出发，建立一般系数先验下的精确条件风险矩阵，
给出各向同性情况下可由单参数 resolvent 差商计算的高维极限，
并区分连续单岭基准、平均家族的可达收益和训练数据选权造成的损失。
主实验包含 $18$ 个配置、每配置 $100$ 次独立重复，共 $1\,800$ 份训练数据、$21\,600$ 条方法记录；
另以 $300$ 份数据验证普通 GCV 的极限反例，以 $720$ 份数据验证有界谱下的确定性等价。
各向同性控制中，连续单岭与先验平均 oracle 混合风险一致；
弱谱方向的信号方差提高时，凸平均存在小幅结构收益；强谱方向更突出时，受约束负权重可扩大可达收益。
但本次实际选权没有实现这些 oracle 收益：校正与 LOO 显著改善普通 GCV 平均，仍未优于单岭调参。
这支持将后续问题聚焦于稳定选权，而不能宣称已提出普遍优于岭回归的新方法。
\end{{abstract}}

\section{{已完成的工作与主要结论}}
\begin{{enumerate}}
\item \textbf{{风险公式已实现并核验。}}一般先验协方差 $\Psi$ 下直接计算完整交叉风险，保留 $p>n$ 的样本零空间；
各向同性极限的方差交叉项包含 $S=X^\mathsf TX/n$，没有将其错误替换为偏差项。
\item \textbf{{信号结构确实影响平均的可达收益，但幅度小。}}
当 $b-a=-0.3$，六个配置的凸平均条件风险收益范围为
${fmt(weakrange[0],6)}$ 至 ${fmt(weakrange[1],6)}$；
当 $b-a=+0.3$，负权重 oracle 的收益范围为
${fmt(strongrange[0],6)}$ 至 ${fmt(strongrange[1],6)}$。
这些是总风险的绝对差，不能误读成百分数。各向同性控制的最大逐次风险偏差小于 $10^{{-12}}$。
\item \textbf{{普通 GCV 的失准有明确代数机制。}}
普通聚合 GCV 与逐候选交叉残差准则可由同一个单纯形二次规划求解，
但前者将解按 $1/(1-\mathrm{{df}}_j/n)$ 再加权，可能放大高自由度候选。
两者最小分数相同，并不意味着拟合出的模型或实际风险相同。
\item \textbf{{有限样本选权仍是主要代价。}}
在全部 $18$ 个配置中，本次交叉残差、LOO 及受约束负权重选权的平均风险仍高于连续单岭 oracle；
不能将固定权重校准有效直接解释为自适应选权已经达到最优。
\end{{enumerate}}

本文的风险恒等式、代数推导和模拟用于课程项目的可核验推广。
多岭参数平均、source condition、正则外推与交叉风险校正均已有相关研究\cite{{zhao,richards,bach,bellec}}；
本报告不据此宣称论文级原创性已确认。

\clearpage
\section{{实验设计、信息范围与计算方法}}
总体特征基中令 $\Sigma=\operatorname{{diag}}(\mu_1,\ldots,\mu_p)$，
系数先验为 $\beta\sim N(0,\Psi)$，$\Psi=\operatorname{{diag}}(\psi_1,\ldots,\psi_p)$，
训练特征行为 $N(0,\Sigma)$，噪声为独立 $N(0,1)$。
主实验固定 $n=100$、$p/n\in\{{0.8,1,1.5\}}$、$a\in\{{1.5,2\}}$，并取
\begin{{equation}}\label{{eq:experiment_population}}
 \mu_i=\frac{{p\,i^{{-a}}}}{{\sum_{{\ell=1}}^p\ell^{{-a}}}},\qquad
 \psi_i=\frac{{i^{{a-b}}}}{{\sum_{{\ell=1}}^p\mu_\ell\ell^{{a-b}}}},\qquad
 b-a\in\{{-0.3,0,0.3\}}.
\end{{equation}}
于是 $\operatorname{{tr}}\Sigma=p$，$\mathbb E[\beta^\mathsf T\Sigma\beta]=1$，总体预测信噪比始终为 $1$；
$b=a$ 时 $\Psi=I_p/p$。
主实验的幂律谱属于固定有限维机制研究；有界谱渐近验证单独进行，未混用理论适用范围。

候选参数为 $[10^{{-4}},10^2]$ 上 $21$ 个对数等距点，再加入 $p/n$（去重后 $21$ 或 $22$ 点），
确保各向同性时已包含精确最优单岭参数。
连续单岭 oracle 在 $[10^{{-6}},10^4]$ 上先检查 $101$ 个对数点，再精化所有检测到的局部极小值；
各向同性时直接使用 $\lambda_*=p/n$。它是宽区间内的数值基准，未证明任意各向异性风险曲线全局单峰。
负权重同时约束 $\boldsymbol1^\mathsf T\pi=1$、$\lVert\pi\rVert_1\le3$。

\textbf{{信息范围。}}先验平均 oracle 只使用 $X$ 以及已知的 $\Sigma,\Psi$；
实际的 Ridge-GCV、普通 GCV 平均、交叉残差平均及 LOO 平均只使用 $X,y$。
广义 Bayes oracle 还知道真实各向异性先验，作为更宽方法类的参照。
真实 $\beta$ 只进入独立评估
\begin{{equation}}\label{{eq:experiment_evaluation}}
 R_{{\mathcal D}}=1+(\widehat\beta-\beta)^\mathsf T\Sigma(\widehat\beta-\beta),
\end{{equation}}
不参与这些算法的训练。该式将新测试点精确积分掉，因此没有额外有限测试集噪声。
各配置内所有方法共享同一训练数据，差值标准误按配对重复计算；根种子为 $202611041$，
主实验、MP 验证、确定性等价验证采用不同种子分支。
图中误差线为 $\pm1$ 个 Monte Carlo 标准误，表中为均值 $\pm\mathrm{{SE}}$。
配对 $95\%$ 区间为正态近似的逐项描述，没有多重比较校正。
"""
    # Full 18-configuration conditional oracle table.
    maintext+=r"""
\clearpage
\section{平均家族的结构收益}
表\ref{tab:oracle}报告给定设计后、对系数与训练噪声平均的风险。
收益定义为连续单岭风险减去对应平均风险；正值表示平均更好。
该目标适用于只依赖设计与生成参数的 oracle，不能用来替代数据自适应选权的实际评估。
\begin{table}[htbp]\centering\small
\caption{全部主配置的条件 oracle 收益；后两列以 $10^{-3}$ 为单位}\label{tab:oracle}
\renewcommand{\arraystretch}{1.12}
\begin{tabular}{cccccc}\toprule
$a$ & $p/n$ & $b-a$ & 连续单岭总风险 & 凸平均收益 $\pm\mathrm{SE}$ & 负权平均收益 $\pm\mathrm{SE}$\\\midrule
"""
    for cfg in configs:
        cid=cfg['config_id'];r=keyed[(cid,'ridge_cont_oracle')]
        c=gainkey[(cid,'oracle_convex')];s=gainkey[(cid,'oracle_signed_L3')]
        maintext+=rf"{cfg['a']:g} & {cfg['gamma']:g} & ${cfg['delta']:+.1f}$ & ${fmt(r['conditional_risk_mean'])}$ & ${fmt(1000*number(c,'mean_conditional_gain'),3)}\pm{fmt(1000*number(c,'mc_se'),3)}$ & ${fmt(1000*number(s,'mean_conditional_gain'),3)}\pm{fmt(1000*number(s,'mc_se'),3)}$ \\"+'\n'
    maintext+=r"""\bottomrule\end{tabular}
\par\vspace{3pt}{\footnotesize 每格 $100$ 份独立设计。均值与标准误按逐次配对风险差计算。
极微小的负权额外收益仍须与优化间隙比较；例如弱谱方向下负权相对凸平均约 $10^{-6}$ 量级的差异不作实质改善结论。}
\end{table}
\textbf{各向同性控制。}当 $b=a$，连续单岭、包含其参数的凸平均及负权平均 oracle 均恢复同一风险，
符合最优线性/Bayes 基准。
当 $b<a$，凸平均在本次六个配置中有严格的正收益；当 $b>a$，有限凸候选族未胜过连续单岭，
但受约束负权重获得正收益。
负的凸平均收益混合了候选网格与滤波器形状限制，不能据此证明无限密网格的凸平均也存在严格劣势。
这些观察是本次有限参数范围内的机制证据，不是关于任意 source condition 的“当且仅当”定理。

\textbf{排除网格混杂的补充诊断。}对相同的 $1\,800$ 个设计，将各自连续最优单岭参数再加入 oracle 候选，
只重新求解凸平均；它不改变实际选权方法，也不增加独立数据量。
当 $b-a=+0.3$ 时，全部 $600$ 个增广凸平均解均回到该连续单岭顶点，收益在机器精度内为零。
因此原表中这些配置的轻微负收益应归于有限网格；现有诊断支持“增广候选凸平均没有额外收益”，
仍不构成关于任意正则参数连续混合的全局不可能性定理。
完整逐次结果见 \texttt{anchor\_diagnostic.csv}。

\begin{figure}[p]\centering
\includegraphics[width=\textwidth]{oracle_gains.pdf}
\caption{相对连续单岭的条件风险收益。横轴为信号谱对齐差 $b-a$，
灰线为离散网格单岭，蓝线为凸平均，橙线为 $\ell_1\le3$ 的负权平均。
正值表示收益；误差线为配对均值的 $\pm1\mathrm{SE}$。部分误差线小于标记大小。}
\label{fig:oracle}
\end{figure}
\clearpage
\section{实际选权没有自动实现 oracle 收益}
以插值附近的配置 $a=1.5,p/n=1,b-a=-0.3$ 为例，表\ref{tab:actual}同时展示实际预测风险与训练内分数。
其余配置全部保存在原始和汇总 CSV，并见图\ref{fig:actual}。
即使使用完全相同的最小分数，普通 GCV 与交叉残差准则也可能给出风险截然不同的实际模型。
\begin{table}[htbp]\centering\small
\caption{代表配置的总预测风险与校准；$100$ 次配对重复}\label{tab:actual}
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{lccc}\toprule
方法 & 总风险 $\pm\mathrm{SE}$ & 平均分数/平均风险 & 平均 $\lVert\pi\rVert_1$\\\midrule
"""
    labels={'ridge_cont_oracle':'连续单岭 oracle','ridge_grid_oracle':'网格单岭 oracle',
            'oracle_convex':'凸平均 oracle','oracle_signed_L3':'负权平均 oracle',
            'ridge_gcv':'单岭 GCV','gcv_ma':'普通 GCV 平均','cross_gcv_ma':'交叉残差平均',
            'loo_ma':'LOO 平均','cross_gcv_signed_L3':'负权交叉残差平均',
            'equal_ma':'等权平均','minnorm':'最小范数','bayes_oracle':'广义 Bayes oracle'}
    cid='a1.5_g1.0_d-0.3'
    for method in labels:
        r=keyed[(cid,method)]
        value=fmt(r['risk_mean'])
        se=number(r,'risk_se')
        riskstr=f'${value}\\pm{fmt(se)}$' if np.isfinite(se) else f'${value}$（样本均值）'
        cr=calkey.get((cid,method))
        ratio=fmt(cr['ratio_of_means'],3) if cr else '--'
        l1=fmt(r['l1_norm_mean'],3) if method not in ['ridge_cont_oracle','minnorm','bayes_oracle'] else '--'
        maintext+=rf"{labels[method]} & {riskstr} & {ratio} & {l1} \\"+'\n'
    maintext+=r"""\bottomrule\end{tabular}
\par\vspace{3pt}{\footnotesize oracle 的已知生成参数风险不作为训练内分数。
最小范数在 $p=n$ 下的理论期望风险为无穷，样本均值仅作描述，不报告普通均值 SE。}
\end{table}

\textbf{可实现收益与估计代价。}交叉残差与 LOO 明显缓解了普通 GCV 平均的风险膨胀，
但仍未达到本次连续单岭基准；受约束负权重的实际选权反而更不稳定。
该结果与“扩大函数家族就能自动提高预测表现”的推断不一致。
校正矩阵能估计固定模型的风险，不意味着对许多权重取最小值后仍完全校准；
本次自适应分数普遍存在向下的选择偏差。
配对差值、比值的 Monte Carlo 标准误以及逐次权重均已保存，避免只比较独立组均值。
相对同样仅用训练数据调参的 Ridge-GCV，本次交叉残差平均与 LOO 平均均没有出现配对 $95\%$ 区间完全低于零的配置；
分别有 $12$ 个和 $11$ 个配置的区间完全高于零。
这些是未做多重校正的 Monte Carlo 描述，不能读作普适劣势定理。

\textbf{机制核验。}全部 $1\,800$ 次重复中，普通 GCV 权重均由交叉残差最优权重按命题 3 精确转换。
两种准则最小值的最大绝对差小于 $10^{-12}$，而预测风险可以相差很大。
这将“指标最小化成功”和“实际预测模型好”明确区分。
\begin{figure}[p]\centering
\includegraphics[width=\textwidth]{selected_risk.pdf}
\caption{所有主配置的实际平均超额风险，纵轴为对数尺度。
所有方法共享各次训练数据，oracle 曲线使用已知生成参数。
误差线为训练重复均值的 $\pm1\mathrm{SE}$。连续单岭与广义 Bayes 在各向同性控制中重合。}
\label{fig:actual}
\end{figure}
\begin{figure}[p]\centering
\includegraphics[width=\textwidth]{calibration.pdf}
\caption{自适应选权后的校准比，即训练内分数的重复均值除以独立评估总风险的重复均值。
虚线 $1$ 表示均值尺度校准；它不表示每份数据上都准确。误差线使用配对 delta 方法。}
\label{fig:calibration}
\end{figure}
\clearpage
\section{独立公式验证与数值审计}
\subsection{固定等权平均的 MP 反例}
取单位协方差、各向同性系数、噪声方差 $1$、$p=n$，比较两组固定候选参数。
每个规模 $n\in\{100,300,600\}$ 使用 $100$ 份独立数据；两组参数共享这些数据。
表\ref{tab:mp}展示 $n=600$ 的结果，与解析极限一致。
\begin{table}[htbp]\centering\small
\caption{固定等权平均：$n=600$ 的验证结果与理论极限}\label{tab:mp}
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{cccccc}\toprule
参数对 & 实际风险 & 普通 GCV & 交叉残差 & 风险极限 & 普通 GCV 极限\\\midrule
"""
    for pair_id,lab in [('0','(0.1,1)'),('1','(0.01,10)')]:
        rs=[r for r in mp if r['pair_id']==pair_id and int(r['n'])==600]
        ordinary=next(r for r in rs if r['method']=='gcv_aggregate')
        cross=next(r for r in rs if r['method']=='gcv_cross')
        maintext+=rf"${lab}$ & ${fmt(ordinary['risk_mean'])}$ & ${fmt(ordinary['estimate_mean'])}$ & ${fmt(cross['estimate_mean'])}$ & ${fmt(ordinary['limit_risk'])}$ & ${fmt(ordinary['limit_gcv'])}$ \\"+'\n'
    maintext+=r"""\bottomrule\end{tabular}\end{table}
此处没有优化权重，所以固定权重的 GCV 失准与数据自适应选择偏差已经被分开。
这也说明不能仅凭单岭 GCV 的一致性就推断平均后普通 GCV 一致。

\subsection{有界谱下的高维风险验证}
另取单位谱和等比例两块谱 $\{0.5,1.5\}$，$n\in\{100,200,400\}$、
$p/n\in\{0.8,1,1.5\}$，每配置 $40$ 次，共 $720$ 份独立设计。
比较单岭 $\lambda=1$ 与三参数 $\lambda=(0.1,1,10)$、权重 $(0.2,0.5,0.3)$ 的固定平均。
以实际样本设计的精确先验平均风险作为参照，使用总体谱方程计算确定性等价。
"""
    er=[abs(number(r,'mean_relative_error')) for r in de if int(r['n'])==400]
    maintext+=rf"在 $n=400$ 的 $12$ 个方法配置中，组均值的最大绝对相对偏差为 ${100*max(er):.3f}\%$。"+'\n'
    maintext+=rf"""
\subsection{{复现与求解精度}}
保存结果共含 $1\,800$ 个不同主实验种子。
每个配置的首个重复均独立重跑，风险与保存值的最大绝对差为 ${fmt(verify['max_replay_abs_error'],2)}$。
全部方法记录的求解检查通过，最大归一化一阶间隙为 ${sci(verify['max_normalized_solver_gap'])}$，
最大约束误差为 ${sci(verify['max_constraint_violation'])}$。
一次初始负权重求解未达到间隙阈值，已使用相同数据种子进一步求解并重新核验；没有替换训练数据。
一般先验风险与直接矩阵逆计算、$p>n$ 零空间、显式删点 LOO、Bayes 基准及 GCV 换元均有独立测试。
检验代码与机器可读审计记录一起交付。
\begin{{figure}}[p]\centering
\includegraphics[width=\textwidth]{{mp_validation.pdf}}
\caption{{固定权重 MP 验证：普通 GCV 与交叉残差估计随规模变化的均值及标准误，虚线为解析极限。
具体曲线含义见图例；这里没有数据选权。}}\label{{fig:mp}}
\end{{figure}}
\begin{{figure}}[p]\centering
\includegraphics[width=\textwidth]{{de_validation.pdf}}
\caption{{有界总体谱下的确定性等价验证。相对误差为精确条件风险减去谱方程预测，再除以预测值。
误差线为 $40$ 次独立设计上均值的 $\pm1\mathrm{{SE}}$；单个有限样本仍存在波动。}}\label{{fig:de}}
\end{{figure}}
\clearpage
\section{{研究边界与下一步应验收什么}}
本阶段完成了理论基线、source alignment 的最小模拟、普通 GCV 失准机制和独立验证。
有限幂律模型中已观测到方法类收益，但幅度远小于实际权重估计损失。
因此后续优先问题应是稳定选权：例如限制有效候选维数、收缩到单岭或使用独立选权数据，
再检查是否能在明确场景中实现相对单岭的配对风险改善。
这些是后续待检验方案，本报告没有将其当作已实现结果。

本次没有证明一般各向异性双参数风险的完整确定性等价，
也没有证明候选数任意增长、正则下界趋零或无约束负权重下的统一一致性。
数学附录只在明确的固定候选集、正则参数远离零及有界谱条件下陈述选权保证。
所有模拟均为高斯设计、线性响应；结果尚未扩展到重尾数据或真实数据。
最小范数在插值点的重尾样本另存分位数与最大值，不能用其不稳定样本均值包装风险改进。

\section{{交付清单与复现入口}}
\begin{{itemize}}
\item \texttt{{Code/}}：风险与求解模块、主实验、独立渐近验证、统计绘图、结果核查、报告生成与单元测试。
\item \texttt{{Results/}}：逐次风险、逐候选权重、配对差值、条件 oracle 收益、校准、机制记录、种子清单与求解审计。
\item \texttt{{Figures/}}：六组矢量 PDF 图，另附 SVG 与 PNG；最小范数的尾部分布作为补充材料。
\item \texttt{{Report/}}：本 PDF、可编辑完整 \LaTeX{{}} 源文件、数学片段和文献条件说明。
\item 项目根目录的 \texttt{{README.md}}、\texttt{{requirements.txt}}、\texttt{{run\_all.sh}} 给出完整复现顺序。
\end{{itemize}}
本次执行使用 Python {manifest['python']}、NumPy {manifest['numpy']}、SciPy {manifest['scipy']}。
所需依赖均在清单中固定版本，报告由 XeLaTeX 实际编译。

\clearpage
\appendix
"""
    theory=(REPORT/'theory.tex').read_text()
    preamble=r"""\documentclass[UTF8,fontset=fandol,11pt,a4paper]{ctexart}
\usepackage[margin=2.15cm,headheight=15pt,headsep=11pt,footskip=22pt]{geometry}
\usepackage{amsmath,amssymb,newtxtext,newtxmath,booktabs,tabularx,graphicx,enumitem,xurl,fancyhdr}
\usepackage[unicode,hidelinks]{hyperref}
\graphicspath{{../Figures/}}
\makeatletter
\setlength{\@fptop}{0pt}\setlength{\@fpsep}{18pt}\setlength{\@fpbot}{0pt plus 1fil}
\makeatother
\setlength{\parindent}{2em}\setlength{\parskip}{3pt}\setlength{\emergencystretch}{2em}
\linespread{1.08}\renewcommand{\arraystretch}{1.15}
\setlist[itemize]{leftmargin=1.5em,itemsep=3pt,topsep=3pt}
\setlist[enumerate]{leftmargin=1.7em,itemsep=4pt,topsep=3pt}
\ctexset{section={format=\large\bfseries,beforeskip=1.2ex,afterskip=.7ex},subsection={format=\normalsize\bfseries,beforeskip=1ex,afterskip=.5ex}}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small B：多岭参数平均与信号谱对齐}\fancyhead[R]{\small 2026 年 11 月 4 日}\fancyfoot[C]{\small\thepage}
\hypersetup{pdftitle={多岭参数平均的风险、信号谱对齐与选权代价},pdfauthor={高维统计课程项目}}
\begin{document}
"""
    bib=r"""
\clearpage
\begin{thebibliography}{7}\small
\bibitem{dw} Dobriban, E. and Wager, S. (2018). \href{https://arxiv.org/abs/1507.03003}{High-Dimensional Asymptotics of Prediction: Ridge Regression and Classification}. \emph{Annals of Statistics}, 46(1), 247--279.
\bibitem{richards} Richards, D., Mourtada, J. and Rosasco, L. (2021). \href{https://proceedings.mlr.press/v130/richards21b.html}{Asymptotics of Ridge(less) Regression under General Source Condition}. \emph{AISTATS}, PMLR 130, 3889--3897.
\bibitem{bellec} Bellec, P. C., Du, J.-H., Koriyama, T., Patil, P. and Tan, K. (2025). \href{https://arxiv.org/html/2310.01374v2}{Corrected generalized cross-validation for finite ensembles of penalized estimators}. \emph{JRSSB}, 87(2), 289--318.
\bibitem{bach} Bach, F. (2021). \href{https://www.di.ens.fr/~fbach/Richardson_SIAM_2021.pdf}{On the Effectiveness of Richardson Extrapolation in Data Science}. \emph{SIAM Journal on Mathematics of Data Science}, 3(4), 1251--1277.
\bibitem{wu} Wu, D. and Xu, J. (2020). \href{https://proceedings.neurips.cc/paper/2020/hash/72e6d3238361fe70f22fb0ac624a7072-Abstract.html}{On the Optimal Weighted $\ell_2$ Regularization in Overparameterized Linear Regression}. \emph{NeurIPS} 33.
\bibitem{lin} Lin, L., Wu, J., Kakade, S. M., Bartlett, P. L. and Lee, J. D. (2024). \href{https://proceedings.neurips.cc/paper_files/paper/2024/hash/6fcb1afcc1e9c2c82c8ddddf03bcf0f6-Abstract.html}{Scaling Laws in Linear Regression: Compute, Parameters, and Data}. \emph{NeurIPS} 37.
\bibitem{zhao} Zhao, S., Liao, J. and Yu, D. (2020; online 2018). \href{https://link.springer.com/article/10.1007/s00362-018-1002-4}{Model averaging estimator in ridge regression and its large sample properties}. \emph{Statistical Papers}, 61, 1719--1739.
\end{thebibliography}
\end{document}
"""
    tex=REPORT/'B_研究报告.tex'
    tex.write_text(preamble+maintext+theory+bib)
    for run in range(2):
        log=REPORT/f'compile_pass{run+1}.txt'
        result=subprocess.run(['xelatex','-interaction=nonstopmode','-halt-on-error',tex.name],
                              cwd=REPORT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        log.write_text(result.stdout)
        if result.returncode:raise RuntimeError(f'LaTeX compilation failed: {log}')
    print(str(REPORT/'B_研究报告.pdf'))


if __name__=='__main__':main()
