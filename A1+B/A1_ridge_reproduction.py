"""A1: ridge prediction risk, conditional risk, and independent-test simulation.

Requires Python >= 3.10 and NumPy. All outputs go beside this script.
Run: python3 A1_ridge_reproduction.py
Checks only: python3 A1_ridge_reproduction.py --check-only
New-model preview: python3 A1_ridge_reproduction.py --experiments paper --paper-repetitions 20

Ridge convention: minimize ||y-Xw||^2/n + lambda*||w||^2.
Noise variance is 1 and Var(w_j)=alpha2/p. Models: Identity, BinaryTree,
and Exponential. --experiments paper adds the author's covariance experiments
without rerunning or replacing the existing identity-covariance results.
The new checks are defined here for local execution; no checks run on import.
The original A1_ridge_risk.py is preserved as the earlier conditional-risk demo.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
TEX = ROOT / 'A1_高维岭回归预测风险_文字综述.tex'
MAIN_GAMMAS = (0.5, 1.0, 2.0)
GAMMA_GRID = (0.25, 0.5, 0.75, 0.9, 0.95, 1., 1.05, 1.1, 1.25, 1.5, 1.75, 2., 2.25, 2.5)
FIXED_LAMBDAS = (0.1, 1.0, 4.0)
AUTHOR_CODE = 'https://github.com/dobriban/high-dim-risk-experiments'
COV_BEGIN = '% BEGIN A1 COVARIANCE RESULTS'
COV_END = '% END A1 COVARIANCE RESULTS'


def theoretical_components(gamma, lam, alpha2=1.0):
    """Return asymptotic bias and variance, excluding test-noise variance 1."""
    gamma, lam = np.broadcast_arrays(np.asarray(gamma, float), np.asarray(lam, float))
    if np.any(gamma <= 0) or np.any(lam <= 0) or alpha2 <= 0:
        raise ValueError('gamma, lambda, alpha2 must all be positive')
    b = 1 - gamma + lam
    delta = np.sqrt(b*b + 4*gamma*lam)
    # Select stable root expression without cancellation in either sign regime.
    h = np.empty_like(b)
    positive = b >= 0
    h[positive] = 2 / (b[positive] + delta[positive])
    h[~positive] = (-b[~positive] + delta[~positive]) / (2*gamma[~positive]*lam[~positive])
    q = (gamma*h*h+h) / delta
    bias = alpha2 * lam*lam*q
    variance = gamma * (h-lam*q)
    return bias, variance


def theoretical_risk(gamma, lam, alpha2=1.0):
    """MP closed form; derivative in the calculation is with respect to z."""
    bias, variance = theoretical_components(gamma, lam, alpha2)
    return 1 + bias + variance


def binary_tree_spectrum(depth):
    """Author's ridge_hier_sim.m: p=2**depth-1, mean eigenvalue=1.

    Work in the covariance eigenbasis, as in the original MATLAB code.
    Gaussian isotropic w and the ridge penalty are rotation invariant, so this
    is risk-equivalent to the tree covariance with the same eigenvalues.
    """
    if depth < 1:
        raise ValueError('BinaryTree depth must be >=1')
    eigenvalues = np.concatenate([np.full(2**(depth-j), 2.**j)
                                 for j in range(1, depth+1)])
    return eigenvalues/eigenvalues.mean()


def exponential_spectrum(p, rate=1.):
    """Midpoint exponential quantiles from the author's ridge_exp_sim.m.

    Do not normalize the finite-p mean: the author uses rate, not a second
    trace normalization. Features remain Gaussian, not exponentially drawn.
    """
    if p < 1 or rate <= 0:
        raise ValueError('p and exponential rate must be positive')
    return -np.log((np.arange(p)+.5)/p)/rate


def general_theoretical_components(gamma, lambdas, eigenvalues, alpha2=1.):
    """General-covariance deterministic equivalent using H_p, not sample S.

    Solve lambda*v + gamma*mean(t*v/(1+t*v)) = 1 on (0,1/lambda).
    This is the negative-real Silverstein equation in compute_ST.m; bisection
    avoids interpolation from a v-grid and differentiation by finite differences.
    Return bias, variance, v, v_prime and equation residuals. v_prime means
    derivative with respect to z at z=-lambda, hence is positive.
    """
    lam = np.atleast_1d(np.asarray(lambdas, float))
    t = np.asarray(eigenvalues, float)
    if gamma <= 0 or alpha2 <= 0 or np.any(lam <= 0):
        raise ValueError('gamma, lambda and alpha2 must be positive')
    if t.ndim != 1 or not len(t) or np.any(t <= 0) or not np.all(np.isfinite(t)):
        raise ValueError('Population eigenvalues must be finite and positive')
    low, high = np.zeros_like(lam), 1/lam
    for _ in range(90):
        mid = (low+high)/2
        tv = t[:, None]*mid
        equation = lam*mid + gamma*np.mean(tv/(1+tv), axis=0)-1
        low = np.where(equation < 0, mid, low)
        high = np.where(equation >= 0, mid, high)
    v = (low+high)/2
    denominator = 1+t[:, None]*v
    d = np.mean((t[:, None]/denominator)**2, axis=0)
    j = np.mean(t[:, None]/denominator**2, axis=0)
    stability = 1-gamma*v*v*d
    if np.any(stability <= 0):
        raise ArithmeticError('Invalid branch of the companion transform')
    v_prime = v*v/stability
    # Positive bias/variance form of Theorem 2.1, avoids cancellation in R.
    bias = alpha2*j/stability
    variance = gamma*v*v*d/stability
    residual = lam*v+gamma*np.mean(t[:, None]*v/denominator, axis=0)-1
    return bias, variance, v, v_prime, residual


def general_theoretical_risk(gamma, lambdas, eigenvalues, alpha2=1.):
    bias, variance, _, _, _ = general_theoretical_components(
        gamma, lambdas, eigenvalues, alpha2)
    return 1+bias+variance


def spectrum(x):
    """Diagonalize the smaller Gram matrix, keeping a fitting basis."""
    n, p = x.shape
    if p <= n:
        eigenvalues, basis = np.linalg.eigh(x.T @ x / n)
        dual = False
    else:
        eigenvalues, basis = np.linalg.eigh(x @ x.T / n)
        dual = True
    if eigenvalues.min() < -1e-10:
        raise ArithmeticError('A Gram matrix should be positive semidefinite')
    return np.maximum(eigenvalues, 0), basis, dual


def conditional_risk(eigenvalues, n, p, lambdas, alpha2=1.0):
    """Exact E[(y0-x0.T*w_hat)^2 | X], including p-n null directions."""
    lambdas = np.atleast_1d(np.asarray(lambdas, float))
    if np.any(lambdas <= 0):
        raise ValueError('All lambdas must be positive')
    s = eigenvalues[:, None]
    bias = alpha2/p * np.sum((lambdas / (s+lambdas))**2, axis=0)
    bias += alpha2 * max(p-n, 0)/p
    variance = np.sum(s / (s+lambdas)**2, axis=0)/n
    return 1 + bias + variance


def covariance_spectral_weights(x, s, basis, dual, population_eigenvalues):
    """diag(Q.T Sigma Q) and Sigma-weighted nullspace mass.

    For dual fitting Q=X.T U/sqrt(n*s) spans the positive sample spectrum.
    Missing mass is tr(Sigma)-sum(weights), not p-n when Sigma is general.
    """
    n, p = x.shape
    t = np.asarray(population_eigenvalues, float)
    if len(t) != p:
        raise ValueError('Population spectrum dimension does not match X')
    if not dual:
        return np.sum(t[:, None]*basis**2, axis=0), 0.
    tolerance = np.finfo(float).eps*max(n, p)*max(float(s.max()), 1.)
    positive = s > tolerance
    q = (x.T @ basis[:, positive])/np.sqrt(n*s[positive])
    weights = np.zeros_like(s)
    weights[positive] = np.sum(t[:, None]*q*q, axis=0)
    missing = float(t.sum()-weights.sum())
    if missing < -1e-8*t.sum():
        raise ArithmeticError('Covariance mass lost in dual spectral basis')
    return weights, max(missing, 0.)


def weighted_conditional_risk(s, weights, null_mass, n, p, lambdas, alpha2=1.):
    """Exact conditional prediction risk for general diagonal Sigma."""
    lam = np.atleast_1d(np.asarray(lambdas, float))
    if np.any(lam <= 0):
        raise ValueError('All lambdas must be positive')
    denominator = s[:, None]+lam
    bias = alpha2/p*(np.sum(weights[:, None]*(lam/denominator)**2, axis=0)+null_mass)
    variance = np.sum(weights[:, None]*s[:, None]/denominator**2, axis=0)/n
    return 1+bias+variance


def fit_from_spectrum(x, y, eigenvalues, basis, dual, lambdas):
    """Fit every lambda on identical data; avoids an explicit matrix inverse."""
    n = x.shape[0]
    if dual:
        projected_y = basis.T @ y
        return (x.T @ basis) @ (projected_y[:, None] / (n*(eigenvalues[:, None]+lambdas)))
    projected_rhs = basis.T @ (x.T @ y / n)
    return basis @ (projected_rhs[:, None] / (eigenvalues[:, None]+lambdas))


def summarize(values):
    mean = values.mean(axis=0)
    se = values.std(axis=0, ddof=1)/np.sqrt(len(values))
    return mean, se


def simulate_group(rng, n, gamma, lambdas, reps, test_size, alpha2,
                   population_eigenvalues=None, model='Identity'):
    """Draw X,w,noise, fit, and test independently in every repetition."""
    p = max(1, round(gamma*n)) if population_eigenvalues is None else len(population_eigenvalues)
    t = np.ones(p) if population_eigenvalues is None else np.asarray(population_eigenvalues, float)
    root_t = np.sqrt(t)
    lambdas = np.asarray(lambdas, float)
    conditional = np.empty((reps, len(lambdas)))
    population = np.empty_like(conditional)
    test = np.empty_like(conditional)
    for k in range(reps):
        x = rng.standard_normal((n, p))*root_t
        s, basis, dual = spectrum(x)
        w = rng.standard_normal(p) * np.sqrt(alpha2/p)
        y = x @ w + rng.standard_normal(n)
        fitted = fit_from_spectrum(x, y, s, basis, dual, lambdas)
        x_test = rng.standard_normal((test_size, p))*root_t
        y_test = x_test @ w + rng.standard_normal(test_size)
        residual = y_test[:, None] - x_test @ fitted
        test[k] = np.mean(residual**2, axis=0)
        population[k] = 1 + np.sum(t[:, None]*(fitted-w[:, None])**2, axis=0)
        if population_eigenvalues is None:
            conditional[k] = conditional_risk(s, n, p, lambdas, alpha2)
        else:
            weights, missing = covariance_spectral_weights(x, s, basis, dual, t)
            conditional[k] = weighted_conditional_risk(s, weights, missing, n, p, lambdas, alpha2)
    cm, cs = summarize(conditional)
    pm, ps = summarize(population)
    tm, ts = summarize(test)
    dm, ds = summarize(test-conditional)
    theory = (theoretical_risk(p/n, lambdas, alpha2) if population_eigenvalues is None
              else general_theoretical_risk(p/n, lambdas, t, alpha2))
    rows = []
    for i, lam in enumerate(lambdas):
        rows.append(dict(model=model, n=n, p=p, gamma_target=float(gamma), gamma=p/n,
                         lambda_=float(lam), sqrt_lambda=float(np.sqrt(lam)), alpha2=alpha2,
                         population_mean_eigenvalue=float(t.mean()),
                         noise_variance=1., repetitions=reps, test_size=test_size,
                         theory=float(theory[i]), conditional_mean=float(cm[i]),
                         conditional_se=float(cs[i]), population_mean=float(pm[i]),
                         population_se=float(ps[i]), test_mean=float(tm[i]), test_se=float(ts[i]),
                         test_ci_low=float(tm[i]-1.96*ts[i]), test_ci_high=float(tm[i]+1.96*ts[i]),
                         paired_test_minus_conditional=float(dm[i]), paired_difference_se=float(ds[i]),
                         absolute_test_theory_gap=float(abs(tm[i]-theory[i]))))
    return rows


def run_checks():
    """Meaningful deterministic checks plus a fixed-X Monte Carlo risk check."""
    checks = {}
    for gamma, expected in ((.5, np.sqrt(2)), (1., (1+np.sqrt(5))/2), (2., (3+np.sqrt(17))/4)):
        value = float(theoretical_risk(gamma, gamma))
        np.testing.assert_allclose(value, expected, rtol=1e-12)
    checks['closed_form_optima'] = 'passed'
    np.testing.assert_allclose(theoretical_risk(.5, 1e-7), 2., rtol=1e-5)
    np.testing.assert_allclose(theoretical_risk(2., 1e-7), 2.5, rtol=1e-5)
    np.testing.assert_allclose(theoretical_risk(1., 1e7), 2., rtol=1e-6)
    assert theoretical_risk(1., 1e-7) > 1000
    checks['noise_and_regularization_limits'] = 'passed'
    rng = np.random.default_rng(8173)
    fit_errors, trace_errors = [], []
    for n, p in ((17, 9), (17, 17), (17, 29)):
        x = rng.standard_normal((n, p))
        y = rng.standard_normal(n)
        lambdas = np.array([.01, .3, 1., 4.])
        s, basis, dual = spectrum(x)
        fitted = fit_from_spectrum(x, y, s, basis, dual, lambdas)
        risk = conditional_risk(s, n, p, lambdas)
        gram = x.T @ x/n
        for j, lam in enumerate(lambdas):
            direct = np.linalg.solve(x.T @ x+n*lam*np.eye(p), x.T @ y)
            np.testing.assert_allclose(fitted[:, j], direct, atol=1e-10, rtol=1e-10)
            a = np.linalg.solve(gram+lam*np.eye(p), np.eye(p))
            trace_risk = 1 + lam**2/p*np.trace(a@a) + np.trace(a@gram@a)/n
            np.testing.assert_allclose(risk[j], trace_risk, atol=1e-10, rtol=1e-10)
            fit_errors.append(float(np.max(np.abs(fitted[:, j]-direct))))
            trace_errors.append(float(abs(risk[j]-trace_risk)))
        fine = np.unique(np.r_[np.geomspace(.001, 100, 400), p/n])
        optimum = fine[np.argmin(conditional_risk(s, n, p, fine))]
        np.testing.assert_allclose(optimum, p/n, rtol=1e-12)
    checks['primal_dual_fit_max_abs_difference'] = max(fit_errors)
    checks['spectral_trace_risk_max_abs_difference'] = max(trace_errors)
    checks['finite_sample_optimum_and_nullspace'] = 'passed'
    n, p, draws = 25, 40, 4000
    x = rng.standard_normal((n, p))
    s, basis, dual = spectrum(x)
    w = rng.standard_normal((p, draws))/np.sqrt(p)
    responses = x @ w + rng.standard_normal((n, draws))
    fixed_x = []
    for lam in [.1, 1., 4.]:
        fit = np.linalg.solve(x.T @ x+n*lam*np.eye(p), x.T @ responses)
        population = 1+np.sum((fit-w)**2, axis=0)
        target = float(conditional_risk(s, n, p, [lam])[0])
        mean, se = float(population.mean()), float(population.std(ddof=1)/np.sqrt(draws))
        z = (mean-target)/se
        assert abs(z) < 6, 'Fixed-X Monte Carlo discrepancy exceeds 6 standard errors'
        fixed_x.append(dict(lambda_=lam, exact_conditional_risk=target,
                            mean_fitted_population_risk=mean, se=se, z=z, draws=draws))
    checks['fixed_X_Monte_Carlo'] = fixed_x
    return checks


def write_csv(name, rows):
    with (ROOT/name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def conditional_convergence(rng, alpha2, reps):
    rows = []
    for n in (120, 300, 600):
        values = []
        for _ in range(reps):
            x = rng.standard_normal((n, n))
            s = np.maximum(np.linalg.eigvalsh(x.T @ x/n), 0)
            values.append(float(conditional_risk(s, n, n, [1/alpha2], alpha2)[0]))
        mean = float(np.mean(values))
        se = float(np.std(values, ddof=1)/np.sqrt(reps))
        theory = float(theoretical_risk(1., 1/alpha2, alpha2))
        rows.append(dict(n=n, p=n, gamma=1., lambda_=1/alpha2, repetitions=reps,
                         theory=theory, conditional_mean=mean, conditional_se=se,
                         absolute_gap=abs(mean-theory)))
        print(f'  convergence n=p={n}: mean={mean:.6f}, theory={theory:.6f}', flush=True)
    return rows


def coordinates(rows, xkey, ykey):
    return ' '.join(f"({r[xkey]:.8g},{r[ykey]:.8g})" for r in rows)


def error_coordinates(rows, xkey):
    return ' '.join(f"({r[xkey]:.8g},{r['test_mean']:.8g}) +- (0,{1.96*r['test_se']:.8g})" for r in rows)


def scientific_tex(value):
    mantissa, exponent = f'{value:.2e}'.split('e')
    return rf'{mantissa}\times10^{{{int(exponent)}}}'


def pgf_figure(rows, group_key, groups, xkey, xlabel, logarithmic=False):
    # Embedded coordinates keep the editor document standalone (no project files).
    lines = [r'\begin{figure}[htbp]', r'\centering', r'\begin{tikzpicture}',
             r'\begin{axis}[width=.91\linewidth,height=7cm,',
             r'xlabel={' + xlabel + r'},ylabel={Total prediction MSE},',
             r'grid=major,grid style={gray!20},',
             r'legend style={font=\scriptsize,at={(0.5,1.03)},anchor=south,legend columns=3},',
             r'tick label style={font=\small},label style={font=\small},',
             ('xmode=log,' if logarithmic else '% linear x axis'), r'ymin=1]']
    for group, color in zip(groups, ('blue!70!black', 'red!70!black', 'green!45!black')):
        subset = sorted([r for r in rows if abs(r[group_key]-group)<1e-10], key=lambda r:r[xkey])
        prefix = r'\gamma' if group_key=='gamma' else r'\lambda'
        # Draw a dense theoretical curve, keeping simulation points at their actual grid.
        low, high = subset[0][xkey], subset[-1][xkey]
        dense_x = np.geomspace(low, high, 300) if logarithmic else np.linspace(low, high, 300)
        dense_y = (theoretical_risk(group, dense_x, subset[0]['alpha2']) if logarithmic
                   else theoretical_risk(dense_x, group, subset[0]['alpha2']))
        dense = [{xkey:float(x), 'theory':float(y)} for x,y in zip(dense_x,dense_y)]
        lines += [r'\addplot[thick,'+color+r'] coordinates {'+coordinates(dense,xkey,'theory')+r'};',
                  r'\addlegendentry{$'+prefix+f'={group:g}'+r'$: theory}',
                  r'\addplot[only marks,mark=o,mark size=1.5pt,'+color+r',forget plot] coordinates {'+coordinates(subset,xkey,'conditional_mean')+r'};',
                  r'\addplot[only marks,mark=triangle*,mark size=1.5pt,'+color+r',forget plot,error bars/.cd,y dir=both,y explicit] coordinates {'+error_coordinates(subset,xkey)+r'};']
        if logarithmic:
            optimum = group/subset[0]['alpha2']
            optimum_risk = float(theoretical_risk(group,optimum,subset[0]['alpha2']))
            lines.append(r'\addplot[only marks,mark=star,mark size=3.5pt,'+color+
                         r',forget plot] coordinates {'+f'({optimum:.8g},{optimum_risk:.8g})'+r'};')
    lines += [r'\end{axis}',r'\end{tikzpicture}']
    label = r'固定 $\gamma$ 扫描 $\lambda$' if logarithmic else r'固定 $\lambda$ 扫描 $\gamma$'
    star_note = r'星号标出理论最优点 $\lambda^*=\gamma/\alpha^2$。' if logarithmic else ''
    figure_label = 'fig:lambda' if logarithmic else 'fig:gamma'
    lines += [r'\caption{'+label+r'。实线为渐近风险，空心圆为条件风险均值，实心三角为独立测试 MSE 均值；误差线为均值的近似点态 $95\%$ 区间。'+star_note+r'}',
              r'\label{'+figure_label+r'}',r'\end{figure}']
    return '\n'.join(lines)


def pgf_bias_figure(rows):
    """Mechanism figure: explain the U-shape by plotting both risk components."""
    lines = [r'\begin{figure}[htbp]\centering\begin{tikzpicture}',
             r'\begin{axis}[width=.91\linewidth,height=7cm,xmode=log,ymin=0,',
             r'xlabel={Regularization $\lambda$},ylabel={Risk contribution},grid=major,grid style={gray!20},',
             r'legend style={font=\scriptsize,at={(.5,1.03)},anchor=south,legend columns=2},',
             r'tick label style={font=\small},label style={font=\small}]']
    for key, label, style in [('bias','Bias','blue!70!black,thick'),
                               ('variance','Variance','red!70!black,thick,dashed'),
                               ('noise','Noise','black!50,densely dotted'),
                               ('theory','Total risk','black,very thick')]:
        lines += [r'\addplot['+style+r'] coordinates {'+coordinates(rows,'lambda_',key)+r'};',
                  r'\addlegendentry{'+label+r'}']
    lines += [r'\end{axis}\end{tikzpicture}',
              r'\caption{偏差、方差与总风险的理论分解（$\Sigma=I_p$、$\gamma=1$）。总风险为 $1+B+V$；曲线由公式计算，不是新增模拟结果。}',
              r'\label{fig:biasvariance}\end{figure}']
    return '\n'.join(lines)


def pgf_strong_figure(rows):
    """Qualitative counterpart of the paper's Fig. 3, with stated selected ratios."""
    lines = [r'\begin{figure}[htbp]\centering\begin{tikzpicture}',
             r'\begin{axis}[width=.91\linewidth,height=7cm,xmode=log,ymode=log,',
             r'xlabel={Signal strength $\alpha^2$},ylabel={Optimal total risk $R^*$},',
             r'grid=major,grid style={gray!20},',
             r'legend style={font=\scriptsize,at={(.5,1.03)},anchor=south,legend columns=5},',
             r'tick label style={font=\small},label style={font=\small}]']
    for gamma,color in zip(sorted(set(r['gamma'] for r in rows)),
                           ('blue!70!black','cyan!60!black','black','orange!90!black','red!70!black')):
        subset = sorted([r for r in rows if r['gamma']==gamma],key=lambda r:r['alpha2'])
        lines += [r'\addplot[thick,'+color+r'] coordinates {'+coordinates(subset,'alpha2','theory_optimal_risk')+r'};',
                  r'\addlegendentry{$\gamma='+f'{gamma:g}'+r'$}']
    lines += [r'\end{axis}\end{tikzpicture}',
              r'\caption{单位协方差下的强信号学习区间：每个点均使用 $\lambda^*=\gamma/\alpha^2$。本图选择 $\gamma\in\{0.5,0.9,1,1.1,2\}$，展示原论文图 3 的理论现象，并非其全部参数设置的逐点重绘。}',
              r'\label{fig:strongsignal}\end{figure}']
    return '\n'.join(lines)


def add_results_to_document(lambda_rows, gamma_rows, convergence, checks, config, bias_rows, strong_rows):
    if not TEX.exists():
        return False
    text = TEX.read_text(encoding='utf-8')
    begin, end = '% BEGIN A1 GENERATED RESULTS', '% END A1 GENERATED RESULTS'
    if begin not in text or end not in text:
        return False
    best = []
    actual_gammas = sorted(set(r['gamma'] for r in lambda_rows))
    for gamma in actual_gammas:
        group = [r for r in lambda_rows if abs(r['gamma']-gamma)<1e-10]
        exact = min(group, key=lambda r:abs(r['lambda_']-gamma/config['alpha2']))
        observed = min(group, key=lambda r:r['test_mean'])
        best.append((exact, observed))
    parts = [begin, r'\section{实验复现}\label{sec:results}',
             r'\subsection{实验设置与评价指标}',
             r'采用 Python 与 NumPy 实现单位协方差模型。每次重复独立生成 $X_{ij}\sim\mathcal N(0,1)$、$w\sim\mathcal N(0,(\alpha^2/p)I_p)$ 和 $\varepsilon\sim\mathcal N(0,I_n)$，构造 $Y=Xw+\varepsilon$ 并拟合岭回归；随后生成独立测试集，训练与测试共享同一个 $w$。',
             f"基础实验采用 $n={config['n']}$、$\\alpha^2={config['alpha2']:g}$，每组 ${config['repetitions']}$ 次独立重复，每次测试集 $T={config['test_size']}$；随机种子为 {config['seed']}。",
             r'正则扫描取 $\gamma\in\{0.5,1,2\}$、$\lambda\in[0.01,10]$，采用对数网格并加入理论最优点；维数扫描取 $\lambda\in\{0.1,1,4\}$、$\gamma\in[0.25,2.5]$。令 $p=\operatorname{round}(n\gamma)$，理论计算使用实际比值 $p/n$。评价指标为',
             r'\begin{equation}\label{eq:mse}',
             r'\operatorname{MSE}_{\mathrm{test}}=\frac{1}{T}\sum_{j=1}^{T}\left(y_j^{\mathrm{test}}-(x_j^{\mathrm{test}})^{\mathsf T}\whw_\lambda\right)^2.',
             r'\end{equation}',
             r'每组比较渐近风险 $R_\lambda$、由式~\eqref{eq:eigenrisk}~精确计算的条件风险 $r_\lambda(X)$，以及实际训练后的测试 MSE。重复均值的标准误为 $\operatorname{SE}=\operatorname{sd}(M_1,\ldots,M_K)/\sqrt K$，其中 $M_k$ 为第 $k$ 次重复的测试 MSE；图中误差线取均值加减 $1.96\operatorname{SE}$，表示近似点态 $95\%$ 区间。',
             r'\subsection{正则化与维数变化的风险表现}',
             r'表~\ref{tab:risk}~比较理论最优点处的三种风险。条件风险均值与渐近预测接近，测试均值在重复误差范围内表现一致。',
             r'\begin{table}[htbp]\centering\small',
             r'\caption{理论最优点处的预测风险比较；测试均值后给出标准误。}\label{tab:risk}',
             r'\begin{tabular}{ccccc}\toprule',
             r'$\gamma$ & $\lambda^*$ & 理论风险 & 条件风险均值 & 测试 MSE 均值 $\pm\operatorname{SE}$ \\ \midrule']
    for r,_ in best:
        parts.append(f"${r['gamma']:g}$ & ${r['lambda_']:g}$ & ${r['theory']:.4f}$ & ${r['conditional_mean']:.4f}$ & ${r['test_mean']:.4f}\\pm{r['test_se']:.4f}$ " + r'\\')
    parts += [r'\bottomrule\end{tabular}\end{table}',
              r'图~\ref{fig:lambda}~进一步展示完整的正则路径。风险随 $\lambda$ 先下降后上升，最低点位于 $\gamma/\alpha^2$ 附近；增大 $\gamma$ 后，最优点向更强正则方向移动。条件风险与理论曲线接近，有限测试 MSE 的最低点受到随机波动与网格分辨率影响。',
              pgf_figure(lambda_rows,'gamma',actual_gammas,'lambda_',r'Regularization $\lambda$',True),
              r'固定正则的维数扫描见图~\ref{fig:gamma}。较弱正则在插值阈值附近产生明显风险峰，峰值位置可以偏离 $\gamma=1$；增强正则后，噪声放大得到缓和，曲线更平滑。该比较保持样本量、总信号强度和噪声方差一致，反映固定收缩强度对不同维数的适应程度。',
              pgf_figure(gamma_rows,'lambda_',FIXED_LAMBDAS,'gamma',r'Aspect ratio $\gamma=p/n$'),
              r'图~\ref{fig:biasvariance}~用式~\eqref{eq:asymbv}~解释上述现象：随着 $\lambda$ 增加，收缩偏差上升，训练噪声方差下降，总风险在两者平衡处达到最低点。不可约噪声贡献始终为 $1$。该图由解析公式计算，用于说明风险变化的机制。',
              pgf_bias_figure(bias_rows),
              r'\subsection{强信号学习区间}',
              r'为复现论文第 2.1 节的学习区间现象，取 $\gamma\in\{0.5,0.9,1,1.1,2\}$，在 $\alpha^2\in[0.01,10^4]$ 上扫描式~\eqref{eq:optimalidentity}，每个点均使用 $\lambda^*=\gamma/\alpha^2$。图~\ref{fig:strongsignal}~在强信号区呈现常数量级、$\alpha$ 量级和 $\alpha^2$ 量级三类增长，双对数斜率分别趋于 $0$、$1/2$ 和 $1$，与式~\eqref{eq:strongunder}--\eqref{eq:strongover}~一致。该图展示单位协方差下的理论现象。',
              pgf_strong_figure(strong_rows),
              r'\subsection{有限样本核验与讨论}',
              r'在 $\gamma=1$、$\lambda=1/\alpha^2$ 处，取 $n=p\in\{120,300,600\}$ 并独立重复抽取训练特征。表~\ref{tab:convergence}~显示各样本量下的条件风险均值均接近理论极限；这里考察的是渐近近似，随机误差不要求逐次单调下降。',
              r'\begin{table}[htbp]\centering\small',
              r'\caption{不同样本规模下的条件风险与高维极限。}\label{tab:convergence}',
              r'\begin{tabular}{cccc}\toprule',
              r'$n=p$ & 理论风险 & 条件风险均值 & 绝对差 \\ \midrule']
    for r in convergence:
        parts.append(f"${r['n']}$ & ${r['theory']:.6f}$ & ${r['conditional_mean']:.6f}$ & ${r['absolute_gap']:.6f}$ " + r'\\')
    max_z = max(abs(r['z']) for r in checks['fixed_X_Monte_Carlo'])
    parts += [r'\bottomrule\end{tabular}\end{table}',
              f"直接线性方程求解与谱方法的最大系数差为 ${scientific_tex(checks['primal_dual_fit_max_abs_difference'])}$，谱风险与迹风险的最大差为 ${scientific_tex(checks['spectral_trace_risk_max_abs_difference'])}$。固定训练特征后进行了 $4000$ 次系数与训练噪声重复，平均模型总体风险与精确条件风险的最大标准化偏差为 ${max_z:.2f}$，通过程序核验。",
              r'边界核验还包括无正则极限、零预测器风险、有限样本最优参数及过参数区的零特征值贡献。上述基础实验考察单位协方差，原文两种协方差模型的复现见下文。',
              r'复现代码为 \filename{A1_ridge_reproduction.py}，参数、原始数值和运行说明保存在同目录，四张图的坐标嵌入本文源文件。',end]
    generated = '\n'.join(parts)
    left, rest = text.split(begin,1)
    _, right = rest.split(end,1)
    text = left+generated+right
    TEX.write_text(text,encoding='utf-8')
    return True


def run_covariance_checks():
    """Local checks of general theory, weighted traces and dual nullspaces."""
    results = {}
    lam = np.array([.01, .3, 1., 4.])
    errors, residuals = [], []
    for gamma in MAIN_GAMMAS:
        b, v, root, derivative, residual = general_theoretical_components(gamma, lam, np.ones(15))
        risk = 1+b+v
        reference = theoretical_risk(gamma, lam)
        np.testing.assert_allclose(risk, reference, rtol=1e-10, atol=1e-10)
        formula = (1+(lam/gamma-1)*(1-lam*derivative/root))/(lam*root)
        np.testing.assert_allclose(risk, formula, rtol=1e-10, atol=1e-10)
        errors.append(float(np.max(np.abs(risk-reference))))
        residuals.append(float(np.max(np.abs(residual))))
    results['general_to_identity_max_difference'] = max(errors)
    results['companion_equation_max_residual'] = max(residuals)
    t_tree = binary_tree_spectrum(4)
    assert len(t_tree) == 15
    np.testing.assert_allclose(t_tree.mean(), 1.)
    np.testing.assert_allclose(exponential_spectrum(1), [np.log(2)])
    results['author_population_spectra'] = 'passed'
    rng = np.random.default_rng(201817)
    trace_errors = []
    for n, p in [(17, 9), (17, 17), (11, 23)]:
        t = exponential_spectrum(p)
        x = rng.standard_normal((n, p))*np.sqrt(t)
        if p > n:
            x[-1] = x[0]  # also exercise a zero eigenvalue in the dual matrix
        s, basis, dual = spectrum(x)
        weights, missing = covariance_spectral_weights(x, s, basis, dual, t)
        spectral = weighted_conditional_risk(s, weights, missing, n, p, lam)
        gram = x.T @ x/n
        for j, value in enumerate(lam):
            a = np.linalg.solve(gram+value*np.eye(p), np.eye(p))
            trace = 1+value**2/p*np.sum(t*np.diag(a@a))+np.sum(t*np.diag(a@gram@a))/n
            np.testing.assert_allclose(spectral[j], trace, rtol=1e-9, atol=1e-9)
            trace_errors.append(float(abs(spectral[j]-trace)))
        grid = np.unique(np.r_[np.geomspace(.001, 100, 400), p/n])
        finite = weighted_conditional_risk(s, weights, missing, n, p, grid)
        np.testing.assert_allclose(grid[np.argmin(finite)], p/n, rtol=1e-12)
        general = general_theoretical_risk(p/n, grid, t)
        np.testing.assert_allclose(grid[np.argmin(general)], p/n, rtol=1e-12)
        np.testing.assert_allclose(general_theoretical_risk(p/n, [1e6], t),
                                   [1+t.mean()], rtol=1e-5)
    results['weighted_spectral_trace_max_difference'] = max(trace_errors)
    results['weighted_nullspace_and_optimal_lambda'] = 'passed'
    n, p, draws = 12, 20, 4000
    t = exponential_spectrum(p)
    x = rng.standard_normal((n, p))*np.sqrt(t)
    s, basis, dual = spectrum(x)
    weights, missing = covariance_spectral_weights(x, s, basis, dual, t)
    w = rng.standard_normal((p, draws))/np.sqrt(p)
    response = x @ w+rng.standard_normal((n, draws))
    fixed = []
    for value in [.1, 1., 4.]:
        fit = np.linalg.solve(x.T @ x+n*value*np.eye(p), x.T @ response)
        risk = 1+np.sum(t[:, None]*(fit-w)**2, axis=0)
        target = float(weighted_conditional_risk(s, weights, missing, n, p, [value])[0])
        se = float(risk.std(ddof=1)/np.sqrt(draws))
        z = float((risk.mean()-target)/se)
        if abs(z) >= 6:
            raise ArithmeticError('General fixed-X Monte Carlo risk check failed')
        fixed.append(dict(lambda_=value, exact_conditional_risk=target,
                          population_mean=float(risk.mean()), se=se, z=z, draws=draws))
    results['general_fixed_X_Monte_Carlo'] = fixed
    return results


def covariance_figure(rows, spectra, scan):
    """Figure-2 counterpart (six panels), or two-panel aspect-ratio extension."""
    is_lambda = scan == 'lambda'
    columns, row_count = 2, 3 if is_lambda else 1
    height = '4.2cm' if is_lambda else '6cm'
    xlabel = r'$\sqrt{\lambda}$' if is_lambda else r'$\gamma=p/n$'
    lines = [r'\begin{figure}[p]\centering\begin{tikzpicture}',
             r'\begin{groupplot}[group style={group size='+f'{columns} by {row_count}'+
             r',horizontal sep=1.1cm,vertical sep=1.2cm},',
             r'width=.43\linewidth,height='+height+r',grid=major,grid style={gray!20},',
             r'xlabel={'+xlabel+r'},ylabel={Total prediction MSE},ymin=1,',
             r'tick label style={font=\scriptsize},label style={font=\small},',
             r'title style={font=\small}]']
    panel_groups = MAIN_GAMMAS if is_lambda else (None,)
    for target in panel_groups:
        for model in ('BinaryTree', 'Exponential'):
            subset = [r for r in rows if r['model']==model and
                      (not is_lambda or abs(r['gamma_target']-target)<1e-10)]
            if is_lambda:
                actual = subset[0]['gamma']
                lines.append(r'\nextgroupplot[title={'+model+rf', $\gamma={actual:.3g}$'+r'}]')
                curves = [(subset, 'red!70!black')]
            else:
                lines.append(r'\nextgroupplot[title={'+model+r'}]')
                curves = [(sorted([r for r in subset if abs(r['lambda_']-value)<1e-10],
                                  key=lambda r:r['gamma']), color)
                          for value, color in zip(FIXED_LAMBDAS,
                                                ('blue!70!black','red!70!black','green!45!black'))]
            for group, color in curves:
                group = sorted(group, key=lambda r:r['lambda_'] if is_lambda else r['gamma'])
                if is_lambda:
                    dense_lam = np.linspace(np.sqrt(group[0]['lambda_']),
                                            np.sqrt(group[-1]['lambda_']), 180)**2
                    panel_spectrum = spectra[model] if model=='BinaryTree' else spectra[(model, target)]
                    risk = general_theoretical_risk(actual, dense_lam, panel_spectrum, group[0]['alpha2'])
                    dense = [dict(sqrt_lambda=float(np.sqrt(value)), theory=float(y))
                             for value, y in zip(dense_lam, risk)]
                    xkey = 'sqrt_lambda'
                else:
                    # Finite H_p changes with p in Exponential. Join theory evaluated
                    # on actual experimental configurations; do not reuse one H_p.
                    dense, xkey = group, 'gamma'
                conditional_color = 'black!60' if is_lambda else color
                lines += [r'\addplot[thick,'+color+r'] coordinates {'+coordinates(dense,xkey,'theory')+r'};',
                          r'\addplot[only marks,mark=o,mark size=1.1pt,'+conditional_color+r'] coordinates {'+coordinates(group,xkey,'conditional_mean')+r'};',
                          r'\addplot[only marks,mark=triangle*,mark size=1.1pt,'+color+
                          r',error bars/.cd,y dir=both,y explicit] coordinates {'+error_coordinates(group,xkey)+r'};']
                if is_lambda:
                    optimum = min(group, key=lambda r:abs(r['lambda_']-actual/r['alpha2']))
                    lines.append(r'\addplot[only marks,mark=star,mark size=2.5pt,blue!70!black] coordinates {'+
                                 f"({optimum['sqrt_lambda']:.8g},{optimum['theory']:.8g})"+r'};')
    lines.append(r'\end{groupplot}\end{tikzpicture}')
    if is_lambda:
        caption = (r'原文图 2 的协方差模型复现：各行对应目标 $\gamma=0.5,1,2$，标题记录实际 $p/n$；'
                   r'左列 BinaryTree，右列 Exponential，横轴统一为 $\sqrt{\lambda}$。'
                   r'红线为一般协方差理论风险，空心圆为精确条件风险均值，三角为独立测试 MSE 均值，'
                   r'误差线为近似点态 $95\%$ 区间，星号标出理论最优点。')
        label = 'fig:covlambda'
    else:
        caption = (r'两种协方差模型的维数样本比扫描。蓝、红、绿分别对应 $\lambda=0.1,1,4$；'
                   r'实线连接各实际配置下的理论风险，空心圆为条件风险均值，三角及误差线为测试均值及近似点态 $95\%$ 区间。'
                   r'BinaryTree 固定 $p$ 改变 $n$，Exponential 固定 $n$ 改变 $p$；此图为原文实验的扩展。')
        label = 'fig:covgamma'
    lines += [r'\caption{'+caption+r'}',r'\label{'+label+r'}\end{figure}']
    return '\n'.join(lines)


def covariance_method_text(config):
    """Same wording in the pre-run document and generated result section."""
    p = 2**config['tree_depth']-1
    return [r'\subsection{原文协方差模型的实验复现}\label{sec:covariance_experiments}',
            r'原文图 2 使用 BinaryTree 与 Exponential 两种总体协方差模型\cite{dw,authorcode}。本文按作者公开代码构造其特征值，采用谱等价表示 $\Sigma=\operatorname{diag}(t_1,\ldots,t_p)$ 和 $X=Z\Sigma^{1/2}$。由于高斯系数分布及岭惩罚具有旋转不变性，该表示保留预测风险的分布；两种模型的训练和测试特征均为高斯变量，模型名称描述的是协方差谱。',
            f"BinaryTree 取深度 $d={config['tree_depth']}$，因此 $p=2^d-1={p}$，按 $n=\\lfloor p/\\gamma_0\\rfloor$ 选择样本量；Exponential 固定 $n={config['exp_n']}$，按 $p=\\lfloor n\\gamma_0\\rfloor$ 选择维数，指数率为 ${config['exp_rate']:g}$。$\\gamma_0$ 是目标比值，所有理论和最优参数计算均使用实际 $\\gamma=p/n$。",
            f"两模型的正则扫描取 $\\gamma_0\\in\\{{0.5,1,2\\}}$，信号强度 $\\alpha^2={config['alpha2']:g}$、噪声方差 $1$，每组独立重复 ${config['paper_repetitions']}$ 次，每次测试集大小为 ${config['paper_test_size']}$；随机种子为 {config['seed']}。BinaryTree 的 $\\sqrt{{\\lambda}}$ 网格为 $[0.1,\\sqrt{{10}}]$，Exponential 为 $[0.1,2.5]$，各取 ${config['paper_lambda_points']}$ 点并加入实际最优点。",
            r'理论曲线以有限总体谱 $H_p=p^{-1}\sum_i\delta_{t_i}$ 代入式~\eqref{eq:generalrisk}，采用式~\eqref{eq:companion_solver}~的正根计算，是确定性等价近似；精确条件风险另由式~\eqref{eq:weightedrisk}~计算。维数扫描保持同一协方差构造规则，取 $\gamma_0\in[0.25,2.5]$、固定 $\lambda\in\{0.1,1,4\}$，用于补充原文固定比值的正则扫描。',
            r'与原文的对应关系需要说明：论文图注写 BinaryTree 的 $p=2^4$，公开代码采用 $p=2^d-1$；本文遵循代码并报告实际比值。公开模拟函数采用 $500$ 次训练重复、每次 $100$ 个测试点，正文与图注的配置记载并不完全一致。本文共享同次重复内不同正则参数的训练和测试数据，边际实验设定一致，随机波动及网格与原图不要求逐点相同。']


def update_covariance_document(lambda_rows, gamma_rows, spectra, config, checks):
    if not TEX.exists():
        return False
    text = TEX.read_text(encoding='utf-8')
    if COV_BEGIN not in text or COV_END not in text:
        raise RuntimeError('Missing covariance markers in the original TeX document')
    parts = [COV_BEGIN]+covariance_method_text(config)
    parts += [r'图~\ref{fig:covlambda}~直接对应原文的六个正则路径面板；图~\ref{fig:covgamma}~考察固定正则下的维数变化。两图统一比较总预测风险，其噪声底座为 $1$。',
              covariance_figure(lambda_rows, spectra, 'lambda'),
              covariance_figure(gamma_rows, spectra, 'gamma'),
              r'\begin{table}[htbp]\centering\small',
              r'\caption{两种协方差模型在理论最优点处的总预测风险。}\label{tab:covopt}',
              r'\begin{tabular}{lccccc}\toprule',
              r'模型 & 实际 $\gamma$ & $\lambda^*$ & 理论 & 条件均值 & 测试均值 $\pm\operatorname{SE}$\\\midrule']
    summary = []
    for model in ('BinaryTree', 'Exponential'):
        for target in MAIN_GAMMAS:
            group = [r for r in lambda_rows if r['model']==model and abs(r['gamma_target']-target)<1e-10]
            opt = min(group, key=lambda r:abs(r['lambda_']-r['gamma']/r['alpha2']))
            observed = min(group, key=lambda r:r['test_mean'])
            summary.append(dict(model=model, gamma_target=target, gamma=opt['gamma'],
                                lambda_star=opt['lambda_'], observed_grid_minimum=observed['lambda_'],
                                theory=opt['theory'], conditional_mean=opt['conditional_mean'],
                                test_mean=opt['test_mean'], test_se=opt['test_se']))
            parts.append(f"{model} & ${opt['gamma']:.3f}$ & ${opt['lambda_']:.3f}$ & ${opt['theory']:.4f}$ & ${opt['conditional_mean']:.4f}$ & ${opt['test_mean']:.4f}\\pm{opt['test_se']:.4f}$ "+r'\\')
    parts += [r'\bottomrule\end{tabular}\end{table}',
              r'表~\ref{tab:covopt}~给出理论最优点的数值对照。精确条件风险的最优参数仍为 $p/(n\alpha^2)$，协方差结构主要改变风险曲线的水平和形状；测试曲线最低点受有限重复和网格影响。两种谱模型之间的风险差异，应结合整个总体谱以及 $\alpha^2\tr(\Sigma)/p$ 的平均信号方差解释。',
              r'对 A1 的判断分为两部分：固定 $\gamma$ 时，以正则路径核对偏差与方差平衡以及理论最优点；固定 $\lambda$ 时，以维数扫描观察插值附近的噪声放大与正则作用。正则为正时风险有限，具体峰值与协方差谱有关，不能把单位协方差图中的形状直接套用到两种模型。']
    for model in ('BinaryTree', 'Exponential'):
        group = [r for r in lambda_rows+gamma_rows if r['model']==model]
        mean_gap = float(np.mean([r['absolute_test_theory_gap'] for r in group]))
        max_gap = max(r['absolute_test_theory_gap'] for r in group)
        parts.append(f"{model} 在本次两类扫描中，测试均值与理论风险的平均绝对差为 ${mean_gap:.4f}$，最大绝对差为 ${max_gap:.4f}$。这些差异同时包含有限样本近似误差和模拟误差；误差线仅量化重复均值的不确定性。")
    parts += [f"一般协方差加权谱公式与直接迹公式的最大差为 ${scientific_tex(checks['weighted_spectral_trace_max_difference'])}$；通用理论在单位协方差下与闭式公式的最大差为 ${scientific_tex(checks['general_to_identity_max_difference'])}$。相关核验及固定训练特征的模拟检查已随本次本地运行完成。", COV_END]
    left, rest = text.split(COV_BEGIN,1)
    _, right = rest.split(COV_END,1)
    TEX.write_text(left+'\n'.join(parts)+right, encoding='utf-8')
    write_csv('A1_covariance_optima.csv', summary)
    return True


def run_covariance_experiments(args, checks):
    started = time.perf_counter()
    spectra = {'BinaryTree':binary_tree_spectrum(args.tree_depth)}
    rng_l, rng_g = (np.random.default_rng(s) for s in
                    np.random.SeedSequence([args.seed, 2018]).spawn(2))
    lambda_rows, gamma_rows = [], []
    spectrum_rows = []
    for model in ('BinaryTree', 'Exponential'):
        seen = set()
        for target in MAIN_GAMMAS:
            if model == 'BinaryTree':
                t = spectra[model]
                n = max(1, int(np.floor(len(t)/target)))
            else:
                n = args.exp_n
                t = exponential_spectrum(max(1, int(np.floor(n*target))), args.exp_rate)
            actual = len(t)/n
            if model == 'Exponential':
                spectra[(model, target)] = t
            upper = np.sqrt(10) if model=='BinaryTree' else 2.5
            grid = np.unique(np.r_[np.linspace(.1, upper, args.paper_lambda_points)**2,
                                    actual/args.alpha2])
            lambda_rows.extend(simulate_group(rng_l, n, target, grid, args.paper_repetitions,
                                             args.paper_test_size, args.alpha2, t, model))
            for i,value in enumerate(t):
                spectrum_rows.append(dict(model=model, gamma_target=target, n=n, p=len(t),
                                          gamma=actual, index=i+1, eigenvalue=float(value)))
            print(f'{model} lambda scan: n={n}, p={len(t)}, gamma={actual:.6g}', flush=True)
        for target in GAMMA_GRID:
            if model == 'BinaryTree':
                t = spectra[model]
                n = max(1, int(np.floor(len(t)/target)))
            else:
                n = args.exp_n
                t = exponential_spectrum(max(1, int(np.floor(n*target))), args.exp_rate)
            if (n,len(t)) in seen:
                continue
            seen.add((n,len(t)))
            gamma_rows.extend(simulate_group(rng_g, n, target, FIXED_LAMBDAS, args.paper_repetitions,
                                            args.paper_test_size, args.alpha2, t, model))
            print(f'{model} gamma scan: n={n}, p={len(t)}, gamma={len(t)/n:.6g}', flush=True)
    # Each Exponential lambda panel has its own finite population spectrum.
    config = dict(vars(args), noise_variance=1., covariance_models=['BinaryTree','Exponential'],
                  ridge_penalty='n*lambda', theoretical_spectrum='finite population H_p',
                  author_code=AUTHOR_CODE, author_commit='d551acd2b3ce283e8ddf3885de4d9fc60dddd766',
                  numpy_version=np.__version__, gamma_grid=list(GAMMA_GRID),
                  fixed_lambdas=list(FIXED_LAMBDAS))
    write_csv('A1_covariance_lambda_scan.csv', lambda_rows)
    write_csv('A1_covariance_gamma_scan.csv', gamma_rows)
    write_csv('A1_population_spectra.csv', spectrum_rows)
    updated = update_covariance_document(lambda_rows, gamma_rows, spectra, config, checks)
    config.update(latex_results_updated=updated, elapsed_seconds=round(time.perf_counter()-started,2))
    (ROOT/'A1_协方差实验配置.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# A1 原文协方差模型复现', '',
             f"BinaryTree depth={args.tree_depth}; Exponential n={args.exp_n}, rate={args.exp_rate}; alpha²={args.alpha2}; repeats={args.paper_repetitions}; test size={args.paper_test_size}.",
             '', '使用作者代码的谱模型、实际 p/n 和一般协方差风险公式；图横轴为 sqrt(lambda)。',
             'BinaryTree p=2**depth-1，与论文图注的 p=2**4 有差异，文稿已明确说明。',
             '两个新图的坐标随运行写入原 TeX；本程序不调用 LaTeX 编译器。',
             '', '| Model | target gamma | actual gamma | mean test-theory absolute gap | max gap |',
             '|---|---|---|---|---|']
    for model in ('BinaryTree','Exponential'):
        for target in MAIN_GAMMAS:
            group = [r for r in lambda_rows if r['model']==model and r['gamma_target']==target]
            gaps = [r['absolute_test_theory_gap'] for r in group]
            lines.append(f"| {model} | {target:g} | {group[0]['gamma']:.6g} | {np.mean(gaps):.6g} | {max(gaps):.6g} |")
    lines += ['', f'LaTeX 结果区块已更新：{updated}。',
              '最优点数值详见 A1_covariance_optima.csv；理论近似误差与模拟标准误分开报告。']
    (ROOT/'A1_协方差复现结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n',type=int,default=120)
    parser.add_argument('--repetitions',type=int,default=100)
    parser.add_argument('--test-size',type=int,default=2000)
    parser.add_argument('--alpha2',type=float,default=1.)
    parser.add_argument('--seed',type=int,default=20261004)
    parser.add_argument('--experiments',choices=('all','identity','paper'),default='all')
    parser.add_argument('--tree-depth',type=int,default=4)
    parser.add_argument('--exp-n',type=int,default=20)
    parser.add_argument('--exp-rate',type=float,default=1.)
    parser.add_argument('--paper-repetitions',type=int,default=500)
    parser.add_argument('--paper-test-size',type=int,default=100)
    parser.add_argument('--paper-lambda-points',type=int,default=50)
    parser.add_argument('--check-only',action='store_true')
    args = parser.parse_args()
    if args.n < 4 or args.repetitions < 2 or args.test_size < 2 or args.alpha2 <= 0:
        parser.error('n>=4, repetitions>=2, test-size>=2, alpha2>0 are required')
    if args.tree_depth < 2 or args.exp_n < 4 or args.exp_rate <= 0 or args.paper_repetitions < 2 or args.paper_test_size < 2 or args.paper_lambda_points < 3:
        parser.error('tree-depth>=2, exp-n>=4, exp-rate>0, paper-repetitions>=2, paper-test-size>=2, paper-lambda-points>=3 are required')
    started = time.perf_counter()
    checks = run_checks()
    if args.experiments != 'paper':
        (ROOT/'A1_核验结果.json').write_text(json.dumps(checks,indent=2,ensure_ascii=False),encoding='utf-8')
    covariance_checks = None
    if args.experiments != 'identity':
        covariance_checks = run_covariance_checks()
        (ROOT/'A1_协方差核验结果.json').write_text(json.dumps(covariance_checks,indent=2,ensure_ascii=False),encoding='utf-8')
    print('All implementation checks passed.',flush=True)
    if args.check_only:
        return
    if args.experiments != 'identity':
        run_covariance_experiments(args, covariance_checks)
    if args.experiments == 'paper':
        return
    # Independent streams keep experiments reproducible when another grid is changed.
    seeds = np.random.SeedSequence(args.seed).spawn(3)
    rng_l, rng_g, rng_c = (np.random.default_rng(s) for s in seeds)
    lambda_rows=[]
    for gamma in MAIN_GAMMAS:
        p=max(1,round(gamma*args.n))
        grid=np.unique(np.r_[np.geomspace(.01,10,31),p/args.n/args.alpha2])
        lambda_rows.extend(simulate_group(rng_l,args.n,gamma,grid,args.repetitions,args.test_size,args.alpha2))
        print(f'Lambda scan complete: gamma={p/args.n:g}',flush=True)
    gamma_rows=[]
    for gamma in GAMMA_GRID:
        gamma_rows.extend(simulate_group(rng_g,args.n,gamma,FIXED_LAMBDAS,args.repetitions,args.test_size,args.alpha2))
        print(f'Gamma scan complete: target gamma={gamma:g}',flush=True)
    convergence=conditional_convergence(rng_c,args.alpha2,min(args.repetitions,100))
    strong=[]
    for gamma in [.5,.9,1.,1.1,2.]:
        for alpha2 in np.geomspace(.01,1e4,80):
            lam=gamma/alpha2
            strong.append(dict(gamma=gamma,alpha2=float(alpha2),lambda_star=float(lam),
                               theory_optimal_risk=float(theoretical_risk(gamma,lam,alpha2)),method='theory_only'))
    bias_rows=[]
    bias_grid=np.unique(np.r_[np.geomspace(.01,10,200),1/args.alpha2])
    bias,variance=theoretical_components(1.,bias_grid,args.alpha2)
    for lam,b,v in zip(bias_grid,bias,variance):
        bias_rows.append(dict(gamma=1.,alpha2=args.alpha2,lambda_=float(lam),
                              bias=float(b),variance=float(v),noise=1.,
                              theory=float(1+b+v),method='theory_only'))
    for name,rows in [('A1_lambda_scan.csv',lambda_rows),('A1_gamma_scan.csv',gamma_rows),
                      ('A1_sample_size_check.csv',convergence),('A1_strong_signal_theory.csv',strong),
                      ('A1_bias_variance_theory.csv',bias_rows)]:
        write_csv(name,rows)
    config=dict(vars(args),noise_variance=1.,covariance='identity',
                python_numpy_version=np.__version__,gamma_grid=list(GAMMA_GRID),
                fixed_lambdas=list(FIXED_LAMBDAS),ridge_penalty='n*lambda',
                convergence_repetitions=min(args.repetitions,100))
    updated=add_results_to_document(lambda_rows,gamma_rows,convergence,checks,config,bias_rows,strong)
    config['latex_results_updated']=updated
    config['elapsed_seconds']=round(time.perf_counter()-started,2)
    (ROOT/'A1_实验配置.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf-8')
    summary=['# A1 数值复现结果','',f"基础配置：n={args.n}，α²={args.alpha2:g}，Σ=I，噪声方差=1，{args.repetitions} 次重复，测试集 T={args.test_size}，种子 {args.seed}。",'',
             '## 理论最优点的风险比较','', '| gamma | lambda* | theory | conditional mean | test mean | test SE | paired gap SE |',
             '|---|---|---|---|---|---|---|']
    for gamma in MAIN_GAMMAS:
        group=[r for r in lambda_rows if abs(r['gamma']-round(gamma*args.n)/args.n)<1e-10]
        opt=min(group,key=lambda r:abs(r['lambda_']-r['gamma']/args.alpha2))
        best=min(group,key=lambda r:r['test_mean'])
        summary.append(f"| {opt['gamma']:g} | {opt['lambda_']:g} | {opt['theory']:.4f} | {opt['conditional_mean']:.4f} | {opt['test_mean']:.4f} | {opt['test_se']:.4f} | {opt['paired_difference_se']:.4f} |")
        print(f"Optimal gamma={opt['gamma']:g}: theory={opt['theory']:.4f}, conditional={opt['conditional_mean']:.4f}, test={opt['test_mean']:.4f} +/- {opt['test_se']:.4f}; observed grid minimum lambda={best['lambda_']:.4f}",flush=True)
    summary += ['', '## 如何解释', '',
                '理论最优参数为 gamma/alpha²。条件风险精确最优点与其一致；有限测试集与有限重复下的最小点可能偏离。测试误差和条件风险的配对差异及标准误保存在 CSV 中。',
                '固定正 lambda 能缓和 gamma=1 附近的噪声放大，不应把有限信号、正正则曲线理解为发散。',
                '本摘要记录单位协方差基础实验。BinaryTree 与 Exponential 的独立结果保存在 A1_协方差复现结果.md 和 A1_covariance_*.csv 中（运行 paper 或 all 模式后生成）。',
                '95% 区间为重复均值的近似点态区间，不是整条曲线的同时置信带。', '',
                '文稿生成四张论证图：lambda 扫描、gamma 扫描、偏差方差分解、强信号学习区间；后两张为理论图。',
                f"耗时：{config['elapsed_seconds']:.2f} 秒。LaTeX 结果区块已更新：{updated}。",
                '图表坐标嵌入同目录文字综述 .tex，随整篇文稿进行 LaTeX 编译；无需外部绘图库。']
    (ROOT/'A1_复现结果.md').write_text('\n'.join(summary)+'\n',encoding='utf-8')
    print('Saved CSV data, configuration, checks, summary, and paper figure coordinates.',flush=True)


if __name__ == '__main__':
    main()
