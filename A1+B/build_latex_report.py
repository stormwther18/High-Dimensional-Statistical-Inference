"""Write a standalone academic LaTeX report from the verified experiment data.

Only the standard library is required. Figure coordinates are embedded in the
final .tex file; compiling that file needs no external images or CSV files.
"""
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / 'latex' / 'A1_第1项_多随机种子复验_规范LaTeX版.tex'

def read(name):
    with (ROOT/'data'/name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

opt = sorted(read('optima_summary.csv'),key=lambda r:(int(r['model_id']),int(r['master_seed'])))
pooled = read('pooled_optima_summary.csv')
cfg = json.loads((ROOT/'config.json').read_text())
check = json.loads((ROOT/'verification.json').read_text())
assert check['status']=='passed'
seeds={str(seed):f'S{i+1}' for i,seed in enumerate(cfg['master_seeds'])}
summary=[]
for r in opt:
    summary.append(rf"{r['model']} & {seeds[r['master_seed']]} & "
       rf"${float(r['test_mean_at_theory_optimum']):.5f}\pm {float(r['test_se_at_theory_optimum']):.5f}$ & "
       rf"${float(r['theory_optimal_risk']):.5f}$ & ${float(r['test_minus_theory']):+.5f}$ & "
       rf"${float(r['observed_optimal_lambda']):.5f}$ \\")
poolrows=[]
for r in pooled:
    poolrows.append(rf"{r['model']} & ${float(r['test_mean_at_theory_optimum']):.5f}\pm {float(r['test_se_at_theory_optimum']):.5f}$ & "
        rf"${float(r['test_minus_theory']):+.5f}$ & ${float(r['test_minus_conditional']):+.5f}$ & "
        rf"${float(r['conditional_minus_theory']):+.5f}$ \\")

template=r'''\documentclass[UTF8,fontset=fandol,a4paper,zihao=-4]{ctexart}
% Compile with XeLaTeX twice. This document is fully self-contained.
\usepackage[margin=2.3cm,headheight=16pt,headsep=12pt,footskip=24pt]{geometry}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{newtxtext,newtxmath}
\usepackage{booktabs,array,tabularx}
\usepackage{graphicx,xcolor}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepgfplotslibrary{groupplots,fillbetween}
\usepackage{caption}
\usepackage{fancyhdr}
\usepackage{enumitem}
\usepackage{xurl}
\usepackage[unicode,hidelinks]{hyperref}
\usepackage{bookmark}
\DeclareMathOperator{\tr}{tr}
\DeclareMathOperator{\Cov}{Cov}
\DeclareMathOperator{\SE}{SE}
\DeclareMathOperator*{\argmin}{arg\,min}
\newcommand{\E}{\mathbb E}
\newcommand{\RDE}{R^{\mathrm{DE}}}
\newcommand{\norm}[1]{\left\lVert #1\right\rVert}
\newcommand{\Tbar}{\overline T}
\newcommand{\Pbar}{\overline P}
\newcommand{\Cbar}{\overline C}
\numberwithin{equation}{section}
\linespread{1.13}
\setlength{\parindent}{2em}
\setlength{\parskip}{0.22em}
\setlength{\emergencystretch}{2em}
\setlength{\abovedisplayskip}{8pt plus 2pt minus 2pt}
\setlength{\belowdisplayskip}{8pt plus 2pt minus 2pt}
\setlength{\textfloatsep}{12pt}
\setlength{\intextsep}{10pt}
\captionsetup{font=small,labelfont=bf,labelsep=quad,skip=6pt}
\ctexset{
 section={format=\large\bfseries,beforeskip=1.2ex,afterskip=.7ex},
 subsection={format=\normalsize\bfseries,beforeskip=1.0ex,afterskip=.4ex}}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small 高维岭回归：多随机种子复验}
\fancyhead[R]{\small A1 / 第 1 项验收报告}
\fancyfoot[C]{\small\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\setlist[itemize]{leftmargin=2em,itemsep=2pt,topsep=3pt}
\hypersetup{pdftitle={A1 多随机种子正则扫描复验验收报告},
 pdfsubject={固定实际维数样本比为1的岭回归风险稳定性},pdfauthor={高维统计课程项目}}

%%FIGURES%%

\begin{document}
\begin{center}
 {\LARGE\bfseries 高维岭回归正则扫描的多随机种子复验\par}
 \vspace{5pt}
 {\large A1 第 1 项：理论风险、测试误差与最优点稳定性\par}
 \vspace{4pt}
 {\small 高维统计课程项目\qquad 2026 年 10 月 22 日}
\end{center}
\vspace{-3pt}
\begin{abstract}
针对 BinaryTree 与 Exponential 协方差谱模型，在实际维数样本比
$\gamma=p/n=1$ 下，分别以三个独立随机种子复验岭回归正则扫描。
每组进行 $500$ 次重复、每次使用 $100$ 个测试点，统一扫描 $62$ 个正则参数。
观测最优参数相对理论最优值 $\lambda^*=1$ 的最大偏离为 $6.37\%$；
合并种子后，两模型的观测最优点均为 $1$。
精确条件风险揭示小幅有限样本近似差异，合并 Exponential 的测试风险也存在可见偏离。
本文保留这些差异并分析其来源；全部汇总均已由逐重复原始数据独立复算。
\end{abstract}

\section{实验设计与主要结果}\label{sec:results}
本次对应周报\cite{weekly}“下周期可验收目标”第 $1$ 项。
BinaryTree 沿用深度 $d=4$ 的作者代码谱，取 $n=p=2^4-1=15$；
Exponential 使用率参数为 $1$ 的中点分位谱，取 $n=p=20$，不再强制归一化谱均值。
两模型均设 $\alpha^2=1$、噪声方差为 $1$，每组重复数 $B=500$，测试样本数 $m=100$。
公共正则网格为
\begin{equation}\label{eq:grid}
 \mathcal L=\left\{\left(0.1+\frac{j}{60}(\sqrt{10}-0.1)\right)^2:
 j=0,\ldots,60\right\}\cup\{1\},\qquad |\mathcal L|=62.
\end{equation}
三个主种子记为 $\mathrm{S1}=2026102201$、$\mathrm{S2}=2026102202$、
$\mathrm{S3}=2026102203$。各模型以主种子和模型编号共同构造独立随机流；
同次重复内所有 $\lambda$ 共享训练与测试数据。共完成 $6$ 组、$3{,}000$ 次独立重复，
保存 $186{,}000$ 行逐重复与网格点记录。

\begin{table}[htbp]
 \centering\small
 \caption{理论最优点 $\lambda^*=1$ 处的六组风险及观测最优参数}
 \label{tab:main}
 \setlength{\tabcolsep}{4.2pt}\renewcommand{\arraystretch}{1.16}
 \begin{tabular}{llrrrr}
 \toprule
 模型 & 种子 & $\Tbar(1)\pm\widehat{\SE}_T(1)$
 & $R_1^{\mathrm{DE}}$ & $\Tbar(1)-R_1^{\mathrm{DE}}$ & $\widehat\lambda$\\
 \midrule
 %%SUMMARY%%
 \bottomrule
 \end{tabular}
 \par\vspace{3pt}
 \begin{minipage}{0.98\textwidth}\footnotesize
 注：风险均含不可约噪声；$R_\lambda^{\mathrm{DE}}$ 为有限总体谱确定性等价，
 $\Tbar$ 为测试 MSE 的重复均值，$\widehat{\SE}_T$ 为该均值的标准误。
 $\widehat\lambda$ 为全网格测试均值的最小点，定义见式\eqref{eq:argmin}。
 \end{minipage}
\end{table}

各组 $\widehat\lambda\in\{0.93633750,1,1.03771566\}$，
合并每模型 $1{,}500$ 次重复后，两模型的最小点均为 $1$。
该结果支持本次配置下最优区域的稳定性；风险数值的有限样本偏离另见第\ref{sec:error}节。

\clearpage
\section{模型、风险口径与标准误}\label{sec:definitions}
\subsection{岭回归与三个风险量}
每次重复 $r$ 独立生成
\begin{equation}\label{eq:model}
 X_r=Z_r\Sigma^{1/2},\qquad y_r=X_rw_r+\varepsilon_r,\qquad
 w_r\sim N\!\left(0,\frac{\alpha^2}{p}I_p\right),
 \quad\varepsilon_r\sim N(0,I_n),
\end{equation}
其中 $Z_r$ 的元素独立服从 $N(0,1)$，且 $Z_r,w_r,\varepsilon_r$ 相互独立。
训练和测试响应共享本次真实系数 $w_r$；测试特征与噪声独立于训练数据。
令 $S_r=X_r^{\mathsf T}X_r/n$，采用无截距岭估计
\begin{align}
 \widehat w_{r,\lambda}
 &=\argmin_{b\in\mathbb R^p}
 \left\{\frac{1}{n}\norm{y_r-X_rb}_2^2+\lambda\norm b_2^2\right\},\label{eq:objective}\\
 &=A_{r,\lambda}\frac{X_r^{\mathsf T}y_r}{n},\qquad
 A_{r,\lambda}=(S_r+\lambda I_p)^{-1}.\label{eq:fit}
\end{align}
因此，未按 $n$ 归一化的残差平方和对应惩罚系数 $n\lambda$。
为区分理论近似与有限样本模拟，定义
\begin{align}
 T_r(\lambda)
 &=\frac{1}{m}\sum_{i=1}^m
 \left[y_{ri}^{\mathrm{test}}-(x_{ri}^{\mathrm{test}})^{\mathsf T}
 \widehat w_{r,\lambda}\right]^2,\label{eq:test}\\
 P_r(\lambda)
 &=1+(\widehat w_{r,\lambda}-w_r)^{\mathsf T}
 \Sigma(\widehat w_{r,\lambda}-w_r),\label{eq:population}\\
 C_r(\lambda)&=\E\!\left[P_r(\lambda)\mid X_r\right].\label{eq:condition}
\end{align}
$T_r$ 为有限测试集 MSE，$P_r$ 为实际拟合模型的总体预测风险；
$C_r$ 进一步平均真实系数与训练噪声的随机性，是给定 $X_r$ 的精确条件风险。
由\cite{dw}第 $2$ 节的风险分解，有
\begin{equation}\label{eq:exact}
 C_r(\lambda)=1+\frac{\alpha^2\lambda^2}{p}\tr(\Sigma A_{r,\lambda}^2)
 +\frac{1}{n}\tr(\Sigma A_{r,\lambda}S_rA_{r,\lambda}).
\end{equation}

\subsection{重复均值及其标准误}
对预先指定的 $\lambda$，定义
\begin{align}
 \Tbar(\lambda)&=\frac1B\sum_{r=1}^B T_r(\lambda),\label{eq:mean}\\
 \widehat{\SE}_T(\lambda)
 &=\left\{\frac{1}{B(B-1)}\sum_{r=1}^B
 [T_r(\lambda)-\Tbar(\lambda)]^2\right\}^{1/2}.\label{eq:se}
\end{align}
标准误以 $B$ 次独立重复为单位。同次重复内的网格点彼此相关，
不能当作额外独立重复。$\Pbar$、$\Cbar$ 及其标准误同理定义。
配对差 $T_r-C_r$ 使用差值自身的样本标准差计算标准误。

\clearpage
\section{理论曲线及最优参数}\label{sec:theory}
\subsection{有限总体谱的确定性等价}
设总体协方差的特征值为 $t_1,\ldots,t_p$，记
$H_p=p^{-1}\sum_{j=1}^p\delta_{t_j}$。
本文将有限总体谱 $H_p$ 代入一般协方差风险公式\cite{dw}，
得到确定性等价 $R_\lambda^{\mathrm{DE}}$。
负实轴上的伴随 Stieltjes 变换 $v_\lambda=v(-\lambda)>0$ 满足
\begin{equation}\label{eq:stieltjes}
 \lambda v_\lambda+\gamma\int\frac{t v_\lambda}{1+t v_\lambda}\,\mathrm dH_p(t)=1,
 \qquad 0<v_\lambda<\lambda^{-1}.
\end{equation}
令
\begin{align}
 d_\lambda&=\int\frac{t^2}{(1+t v_\lambda)^2}\,\mathrm dH_p(t),&
 j_\lambda&=\int\frac{t}{(1+t v_\lambda)^2}\,\mathrm dH_p(t),\label{eq:integrals}\\
 D_\lambda&=1-\gamma v_\lambda^2d_\lambda>0.\label{eq:stability}
\end{align}
数值实现使用避免相近数相减的非负分量形式：
\begin{equation}\label{eq:de}
 R_\lambda^{\mathrm{DE}}
 =1+\underbrace{\frac{\alpha^2j_\lambda}{D_\lambda}}_{\text{收缩偏差贡献}}
 +\underbrace{\frac{\gamma v_\lambda^2d_\lambda}{D_\lambda}}_{\text{训练噪声方差贡献}}.
\end{equation}
它是有限样本风险的理论近似，并不等于精确的 $\E_X[C_r(\lambda)]$。
本次 $n=p=15$ 或 $20$ 的复验只评价固定配置的数值表现，
不据此作任意样本规模或任意增维谱序列的无偏保证。

\subsection{一般协方差下的条件风险最优点}
由式\eqref{eq:exact}，并利用 $A_{r,\lambda}$ 与 $S_r$ 可交换，
对 $\lambda$ 求导得
\begin{equation}\label{eq:derivative}
 C_r'(\lambda)=2\left(\frac{\alpha^2\lambda}{p}-\frac1n\right)
 \tr\!\left[\Sigma S_r(S_r+\lambda I_p)^{-3}\right].
\end{equation}
本次两种 $\Sigma$ 均正定；若 $S_r$ 存在非零方向，则上式迹项严格为正。
因此，每个固定训练设计的唯一最优参数为
\begin{equation}\label{eq:optimum}
 \lambda^*=\frac{p}{n\alpha^2}=\frac\gamma{\alpha^2}=1.
\end{equation}
六组保存的条件风险均值也都在 $\lambda=1$ 取得网格最小值。
有限测试均值的观测最优参数定义为
\begin{equation}\label{eq:argmin}
 \widehat\lambda=\argmin_{\lambda\in\mathcal L}\Tbar(\lambda).
\end{equation}
随种子变化的是式\eqref{eq:argmin}，而非式\eqref{eq:optimum}。
两者需要分别解释：前者包含模拟波动和网格选择，后者是本模型的条件风险结论。

\clearpage
\section{风险曲线的多种子稳定性}\label{sec:curves}
\begin{figure}[htbp]
 \centering
 \RiskCurvesFigure
 \caption{两模型的测试风险正则扫描。上排为完整范围，下排放大最优区域。
 彩色实线为三个种子的 $\Tbar(\lambda)$，黑色虚线为 $R_\lambda^{\mathrm{DE}}$，
 竖直点线为 $\lambda=1$，标记点为各组 $\widehat\lambda$。
 本图仅绘重复均值曲线，不绘置信带。}
 \label{fig:curves}
\end{figure}
图\ref{fig:curves}中，三个种子均显示风险随正则强度先下降、后上升的形状，
且低谷位于 $\lambda=1$ 附近。BinaryTree 三个种子的观测最优参数分别为
$1.03771566$、$1$、$1$；Exponential 分别为
$1.03771566$、$0.93633750$、$1.03771566$。
不同种子支持相同的最优区域结论，合并后两模型均取 $\widehat\lambda=1$。

从 $\lambda=1$ 改选每组观测最小点，测试 MSE 均值最多下降
$3.8052\times10^{-4}$；低谷的平坦性解释了为何小幅均值波动也会移动观测最优参数。
风险值在纵向的差异仍应与理论近似差异区分，不能仅凭曲线形状相似而忽略。

\clearpage
\section{理论差值的分解与偏离分析}\label{sec:error}
\begin{figure}[htbp]
 \centering
 \ErrorComponentsFigure
 \caption{固定 $\lambda=1$ 的风险差分量。各点为独立重复的均值差，
 误差棒为该分量的均值 $\pm1.96\widehat{\SE}$；
 $\Tbar-\Cbar$ 使用同次重复配对差的标准误。
 所有区间均为点态近似区间。}
 \label{fig:error}
\end{figure}
\begin{table}[htbp]
 \centering\small
 \caption{合并三个种子后在 $\lambda=1$ 处的风险分解（每模型 $B=1{,}500$）}
 \label{tab:pool}
 \setlength{\tabcolsep}{4.5pt}\renewcommand{\arraystretch}{1.15}
 \begin{tabular}{lrrrr}
 \toprule
 模型 & $\Tbar\pm\widehat{\SE}_T$ & $\Tbar-R_1^{\mathrm{DE}}$
 & $\Tbar-\Cbar$ & $\Cbar-R_1^{\mathrm{DE}}$\\
 \midrule
 %%POOL%%
 \bottomrule
 \end{tabular}
\end{table}
六个单种子在 $\lambda=1$ 处均有
$|\Tbar-R_1^{\mathrm{DE}}|/\widehat{\SE}_T\le1.732$。
但合并 Exponential 的差为 $+0.01785721$，标准误为 $0.00770400$，
描述性比例为 $2.3179$；其普通点态区间
$[1.49178617,1.52198587]$ 不含理论值 $1.48902882$。

合并条件风险相对理论的差分别为
$0.00694870$（BinaryTree，$\widehat{\SE}_C=0.00072092$）和
$0.00582561$（Exponential，$\widehat{\SE}_C=0.00056242$），
约占各自理论风险的 $0.45\%$ 和 $0.39\%$。
它们提示小幅有限样本近似差异，且仍含训练设计的抽样误差。
风险差可进一步严格分解为
\begin{equation}\label{eq:decomp}
 \Tbar-R_\lambda^{\mathrm{DE}}
 =(\Tbar-\Pbar)+(\Pbar-\Cbar)+(\Cbar-R_\lambda^{\mathrm{DE}}),
\end{equation}
其中前两项分别为有限测试样本波动、系数与训练噪声抽样波动；最后一项满足
\begin{equation}\label{eq:bias}
 \Cbar-R_\lambda^{\mathrm{DE}}
 =(\Cbar-\E_XC_r)+(\E_XC_r-R_\lambda^{\mathrm{DE}}).
\end{equation}
故不能把测试与理论的全部差异归因于测试集太小，也不能将 $\Cbar-R_\lambda^{\mathrm{DE}}$
视为没有抽样误差的偏差真值。

\clearpage
\section{观测最优点的漂移及其风险代价}\label{sec:argmin}
\begin{figure}[htbp]
 \centering
 \OptimaFigure
 \caption{观测最优参数的位移与理论风险增量。
 BT1--BT3 表示 BinaryTree 的 S1--S3；EX1--EX3 表示 Exponential 的 S1--S3。
 左图为 $\widehat\lambda-1$，右图为 $R_{\widehat\lambda}^{\mathrm{DE}}-R_1^{\mathrm{DE}}$。
 每点对应一次完整种子组的网格最小值，不表示最优点的置信区间。}
 \label{fig:argmin}
\end{figure}
定义相对位移和确定性理论风险增量
\begin{equation}\label{eq:regret}
 \delta_\lambda=\frac{|\widehat\lambda-\lambda^*|}{\lambda^*},\qquad
 \mathcal E_{\mathrm{DE}}=R_{\widehat\lambda}^{\mathrm{DE}}-R_{\lambda^*}^{\mathrm{DE}}.
\end{equation}
六组最大相对位移为 $6.3663\%$，最大理论风险增量为
$3.36398\times10^{-4}$，低于 $0.000337$。
这表明观测最优点的位移对应很小的理论风险代价，与图\ref{fig:curves}中的平坦低谷一致。

\subsection{关于选择偏差和区间的解释}
由于 $\widehat\lambda$ 与 $\Tbar(\widehat\lambda)$ 由同一批测试均值选出，
观测最小风险具有向下的选择偏差。它相对 $\Tbar(1)$ 的小幅下降只是描述性比较，
不能解释为正式的性能优胜；若需作选参后的统计推断，应使用独立验证数据或适当的选择校正。
同样，固定 $\lambda$ 的均值 $\pm1.96\widehat{\SE}$ 只是近似点态区间，
既不是整条曲线的同时置信带，也不是选择后风险的有效置信区间。

\subsection{可验收结论}
在本次固定小样本配置和公共网格下，三个种子支持相同的风险曲线形状与最优区域；
每模型合并后，测试均值最优点均为 $\lambda=1$。
与此同时，风险值存在小幅有限样本近似差异，合并 Exponential 的偏离尤其需要保留。
因此，最优区域的稳定性和风险近似的数值精度是两个应分别陈述的结论。

\clearpage
\section{复算核验与交付对应}\label{sec:verification}
\subsection{已经执行的独立核验}
独立核验脚本直接从原始逐重复 CSV 重算全部汇总，未调用实验脚本的汇总函数：
\begin{itemize}
 \item 核对 $186{,}000$ 个唯一的种子、模型、重复与网格组合，无缺失和重复；
 每组均有 $500$ 次重复和 $62$ 个正则点。
 \item 重算 $496$ 行扫描汇总与 $8$ 行最优点汇总（均含合并结果），
 最大绝对差为 $1.64\times10^{-14}$。
 \item 重放 $6$ 组首个重复的全部网格，与 $372$ 次直接矩阵求解比较，
 精确条件风险最大绝对差为 $2.32\times10^{-13}$。
 \item 原有总体谱、Stieltjes 方程、一般协方差条件风险及零空间核验均通过；
 源代码与数据的 SHA256 校验一致。
\end{itemize}
另有独立审查使用标准库统计函数复算六组结果，数值一致。
算术核验确认实现与数据完整性，不替代对选择偏差或模型近似误差的统计解释。

\subsection{交付文件与验收标准}
\begin{table}[htbp]
 \centering\small
 \caption{验收要求与可核验产物的对应}
 \label{tab:deliverables}
 \renewcommand{\arraystretch}{1.2}
 \begin{tabularx}{\textwidth}{@{}p{.31\textwidth}X@{}}
 \toprule
 验收要求 & 对应文件或正文位置\\
 \midrule
 两模型各三个独立种子 & \path{config.json}；\path{data/raw_repetitions.csv}\\
 统一实际 $\gamma$、网格、重复和测试规模 &
 \path{config.json}；\path{data/lambda_grid.csv}；式\eqref{eq:grid}\\
 MSE、SE、理论差值与观测最优参数 &
 表\ref{tab:main}；\path{data/optima_summary.csv}\\
 所有数值可从原始记录复算 &
 \path{verify_results.py}；\path{verification.json}\\
 跨种子稳定性及明显偏离的分析 &
 第\ref{sec:curves}--\ref{sec:argmin}节；\path{methods_review.md}\\
 可编辑的规范数学排版 & 本报告的独立 \LaTeX{} 源文件，含全部内联绘图坐标\\
 \bottomrule
 \end{tabularx}
\end{table}
实验实际运行环境为 Python %%PYTHON%%、NumPy %%NUMPY%%。
本次数学排版修订保留同一组已核验数据，不重新抽样。
交付的源文件使用 \texttt{XeLaTeX} 连续编译两遍即可生成 PDF，
公式由数学环境排版，图形由 PGFPlots 原生绘制，编译时无需外部 CSV 或图片。

\begin{thebibliography}{9}
\small
\bibitem{dw}
Dobriban E, Wager S. High-dimensional asymptotics of prediction:
Ridge regression and classification[J]. The Annals of Statistics,
2018, 46(1): 247--279. 第 2 节、定理 2.1 及其证明.
\bibitem{weekly}
第 3 组. 高维统计课程项目周报[R]. 报告周期：2026 年 9 月 18 日至 10 月 8 日.
第 4 页，“下周期可验收目标”第 1 项.
\bibitem{implementation}
A1 项目实现. \path{A1_ridge_reproduction.py}；
本次扩展：\path{run_experiment.py}、\path{verify_results.py}.
运行配置、完整数值与独立核验记录见随附材料.
\end{thebibliography}
\end{document}
'''

figures=(ROOT/'latex_figures.tex').read_text(encoding='utf-8')
text=(template.replace('%%FIGURES%%',figures)
      .replace('%%SUMMARY%%','\n'.join(summary))
      .replace('%%POOL%%','\n'.join(poolrows))
      .replace('%%PYTHON%%',cfg['python_version'])
      .replace('%%NUMPY%%',cfg['numpy_version']))
assert '%%SUMMARY%%' not in text and '\\includegraphics' not in text and '\\input{' not in text
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text(text,encoding='utf-8')
print(OUT)
