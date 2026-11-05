"""Exact finite-design risks and data-only weight criteria for ridge averaging.

Convention: minimize ||y-X beta||^2/n + lambda ||beta||^2.  ``prior_risk``
integrates an independent mean-zero beta with known diagonal covariance Psi and
training noise, conditional on X.  It does NOT evaluate data-selected weights.
``realized_risk`` evaluates a fitted coefficient vector against a simulation's
true beta and population covariance, integrating only a new test observation.

No experiment is run and no file is written on import.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def prepare_design(X, population_variances, prior_variances, noise_variance=1.0):
    """Diagonalize the complete p-dimensional Gram matrix once.

    Population covariance and prior covariance are diagonal in the supplied
    feature basis; their sample-eigenbasis representations need not commute.
    Keeping all p directions is essential for p>n: the bias contains the sample
    nullspace, including off-diagonal prior/population interactions.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("X must be a finite matrix")
    n, p = X.shape
    mu = np.asarray(population_variances, dtype=float)
    psi = np.asarray(prior_variances, dtype=float)
    if mu.shape != (p,) or psi.shape != (p,):
        raise ValueError("Population/prior variance vectors must have length p")
    if np.any(mu <= 0) or np.any(psi < 0) or noise_variance <= 0:
        raise ValueError("Population/noise variances must be positive; prior nonnegative")
    if not np.isfinite(mu).all() or not np.isfinite(psi).all():
        raise ValueError("Variances must be finite")
    s, V = np.linalg.eigh(X.T @ X / n)
    tolerance = np.finfo(float).eps * max(n, p) * max(float(s.max()), 1.0)
    if float(s.min()) < -10 * tolerance:
        raise ArithmeticError("Gram matrix has an appreciably negative eigenvalue")
    # Discard only numerical zeros, not genuine positive small eigenvalues.
    s = np.where(s > tolerance, s, 0.0)
    A = V.T @ (mu[:, None] * V)
    P = V.T @ (psi[:, None] * V)
    schur = A * P
    schur = (schur + schur.T) / 2
    return {
        "X": X, "n": n, "p": p, "s": s, "V": V,
        "XV": X @ V, "population_variances": mu,
        "prior_variances": psi, "noise_variance": float(noise_variance),
        "population_in_sample_basis": A,
        "prior_in_sample_basis": P,
        "bias_schur": schur, "sample_covariance_weights": np.diag(A).copy(),
        "rank_tolerance": tolerance,
    }


def prior_risk_matrix(design, lambdas):
    """Return excess-risk matrix C for fixed/X-only weights summing to one.

    In the sample eigenbasis, B=(V' Sigma V) elementwise*(V' Psi V) is PSD.
    This keeps the noncommuting Psi between resolvents exactly rather than
    incorrectly applying a one-parameter trace difference identity.
    """
    lam = np.atleast_1d(np.asarray(lambdas, dtype=float))
    if np.any(lam <= 0) or not np.isfinite(lam).all():
        raise ValueError("All candidate penalties must be finite and positive")
    inv = 1.0 / (design["s"][:, None] + lam[None, :])
    shrink = inv * lam
    bias = shrink.T @ design["bias_schur"] @ shrink
    noise_weights = design["sample_covariance_weights"] * design["s"]
    variance = (design["noise_variance"] / design["n"]) * (
        inv.T @ (noise_weights[:, None] * inv))
    C = bias + variance
    return (C + C.T) / 2


def single_prior_risk(design, lambda_):
    """Exact total risk averaged over prior/noise conditional on this X."""
    lam = float(lambda_)
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("lambda_ must be finite and positive")
    inv = 1.0 / (design["s"] + lam)
    bias_filter = lam * inv
    bias = bias_filter @ design["bias_schur"] @ bias_filter
    variance = (design["noise_variance"] / design["n"]) * np.dot(
        design["sample_covariance_weights"] * design["s"], inv * inv)
    return float(design["noise_variance"] + bias + variance)


def candidate_path(design, y, lambdas):
    """Fit candidates, exact prior risk, and three training-only criteria.

    corrected_matrix is the Gram matrix of component-wise df-adjusted
    residuals, not the ordinary GCV of the aggregate. loo_matrix uses exact
    individual leverages, holding n*lambda fixed in each leave-one-out fit.
    Both matrices estimate total prediction cross-risks, so the test-noise
    constant is already included when weights sum to one.
    """
    lam = np.atleast_1d(np.asarray(lambdas, dtype=float))
    y = np.asarray(y, dtype=float)
    if y.shape != (design["n"],) or not np.isfinite(y).all():
        raise ValueError("y must be a finite length-n vector")
    if np.any(lam <= 0) or not np.isfinite(lam).all():
        raise ValueError("All penalties must be finite and positive")
    s = design["s"]
    n = design["n"]
    inv = 1.0 / (s[:, None] + lam)
    rhs = design["XV"].T @ y / n
    rhs = np.where(s > 0, rhs, 0.0)
    spectral_coefficients = rhs[:, None] * inv
    coefficients = design["V"] @ spectral_coefficients
    residuals = y[:, None] - design["XV"] @ spectral_coefficients
    df = np.sum(s[:, None] * inv, axis=0)
    leverages = (design["XV"] ** 2 / n) @ inv
    d = 1.0 - df / n
    individual_d = 1.0 - leverages
    if np.any(d <= 0) or np.any(individual_d <= 0):
        raise ArithmeticError("Nonpositive residual degrees of freedom")
    corrected = residuals / d[None, :]
    loo = residuals / individual_d
    return {
        "lambdas": lam, "coefficients": coefficients,
        "residuals": residuals, "df": df, "leverages": leverages,
        "aggregate_denominator": d,
        "prior_risk_matrix": prior_risk_matrix(design, lam),
        "corrected_matrix": corrected.T @ corrected / n,
        "loo_matrix": loo.T @ loo / n,
        "training_residual_matrix": residuals.T @ residuals / n,
    }


def realized_risk(coefficients, beta, population_variances, noise_variance=1.0):
    """Exact total new-observation risk of fitted coefficient vector(s)."""
    coef = np.asarray(coefficients, dtype=float)
    beta = np.asarray(beta, dtype=float)
    mu = np.asarray(population_variances, dtype=float)
    if coef.ndim == 1:
        return float(noise_variance + np.sum(mu * (coef - beta) ** 2))
    if coef.ndim != 2 or coef.shape[0] != len(beta):
        raise ValueError("coefficients must have shape (p,) or (p,K)")
    return noise_variance + np.sum(mu[:, None] * (coef - beta[:, None]) ** 2, axis=0)


def _linear_minimum(gradient, signed_l1):
    """A linear oracle for sum(w)=1, ||w||_1<=L (L=1: simplex)."""
    out = np.zeros_like(gradient)
    lo, hi = int(np.argmin(gradient)), int(np.argmax(gradient))
    L = 1.0 if signed_l1 is None else float(signed_l1)
    out[lo] += (L + 1.0) / 2.0
    out[hi] -= (L - 1.0) / 2.0
    return out


def weight_diagnostics(matrix, weights, signed_l1=None):
    """Global Frank-Wolfe/KKT gap on the actual constrained weight set."""
    C = (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T) / 2
    w = np.asarray(weights, dtype=float)
    gradient = 2 * C @ w
    v = _linear_minimum(gradient, signed_l1)
    gap = max(0.0, float(gradient @ (w - v)))
    normalization = max(1.0, float(np.max(np.abs(gradient))))
    L = 1.0 if signed_l1 is None else float(signed_l1)
    return {
        "objective": float(w @ C @ w), "kkt_gap": gap,
        "normalized_kkt_gap": gap / normalization,
        "sum_error": abs(float(w.sum()) - 1.0),
        "l1_norm": float(np.sum(np.abs(w))),
        "constraint_violation": max(abs(float(w.sum()) - 1.0),
                                    float(np.sum(np.abs(w))) - L, 0.0),
    }


def solve_weights(matrix, signed_l1=None):
    """Globally convex QP; return (weights, objective/KKT diagnostics).

    ``signed_l1=None`` imposes nonnegative weights summing to one.  A numerical
    L>=1 allows signed weights with ||w||_1<=L and sum(w)=1. SLSQP uses analytic
    derivatives and a scale-normalized objective. A short exact-line-search
    Frank-Wolfe polish provides a directly checkable global optimality gap.
    """
    C = np.asarray(matrix, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1] or not np.isfinite(C).all():
        raise ValueError("matrix must be finite and square")
    C = (C + C.T) / 2.0
    K = len(C)
    if K < 1:
        raise ValueError("At least one candidate is required")
    L = 1.0 if signed_l1 is None else float(signed_l1)
    if not np.isfinite(L) or L < 1.0:
        raise ValueError("signed_l1 must be at least one")
    scale = max(float(np.max(np.abs(C))), 1e-12)
    A = C / scale
    w0 = np.zeros(K)
    w0[int(np.argmin(np.diag(C)))] = 1.0
    if K == 1:
        diagnostics = weight_diagnostics(C, w0, signed_l1)
        diagnostics.update(success=True, optimizer_success=True, iterations=0,
                           message="single candidate", polish_iterations=0)
        return w0, diagnostics
    if L == 1.0:
        result = minimize(
            lambda w: float(w @ A @ w), w0, jac=lambda w: 2 * A @ w,
            method="SLSQP", bounds=[(0.0, 1.0)] * K,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1,
                          "jac": lambda w: np.ones(K)}],
            options={"ftol": 2e-13, "maxiter": 600, "disp": False})
        w = np.maximum(result.x, 0.0)
        w /= w.sum()
    else:
        z0 = np.concatenate([w0, np.zeros(K)])
        one_minus = np.concatenate([np.ones(K), -np.ones(K)])
        def objective(z):
            w = z[:K] - z[K:]
            return float(w @ A @ w)
        def derivative(z):
            g = 2 * A @ (z[:K] - z[K:])
            return np.concatenate([g, -g])
        result = minimize(
            objective, z0, jac=derivative, method="SLSQP",
            bounds=[(0.0, (L + 1.0) / 2)] * K + [(0.0, (L - 1.0) / 2)] * K,
            constraints=[
                {"type": "eq", "fun": lambda z: one_minus @ z - 1,
                 "jac": lambda z: one_minus},
                {"type": "ineq", "fun": lambda z: L - z.sum(),
                 "jac": lambda z: -np.ones(2 * K)}],
            options={"ftol": 2e-13, "maxiter": 700, "disp": False})
        w = result.x[:K] - result.x[K:]
        # SLSQP's equality error is normally <1e-14. Keep exact feasibility
        # without increasing the L1 bound by combining with an interior point.
        if abs(w.sum() - 1) > 1e-10 or np.abs(w).sum() > L + 1e-9:
            w = w0.copy()
        else:
            w[int(np.argmax(w))] += 1.0 - w.sum()
    if w @ C @ w > w0 @ C @ w0 + 1e-11 * scale:
        w = w0.copy()
    polish_iterations = 0
    for polish_iterations in range(201):
        gradient = 2 * A @ w
        vertex = _linear_minimum(gradient, signed_l1)
        direction = vertex - w
        gap = -float(gradient @ direction)
        if gap <= 2e-9:
            break
        curvature = float(direction @ A @ direction)
        if curvature <= 0:
            step = 1.0
        else:
            step = min(1.0, max(0.0, gap / (2 * curvature)))
        w = w + step * direction
    diagnostics = weight_diagnostics(C, w, signed_l1)
    refined = False
    refinement_iterations = 0
    # A reported SLSQP success can still have an appreciable global gap when
    # the lambda path is highly collinear. Refine only failed certificates so
    # that already accepted fits remain bit-for-bit unchanged on replay.
    if (diagnostics["constraint_violation"] >= 1e-7
            or diagnostics["normalized_kkt_gap"] >= 2e-5):
        refined = True
        refine_scale = max(1.0, abs(float(w @ C @ w)))
        Ac = C / refine_scale
        if L == 1.0:
            refined_result = minimize(
                lambda v: float(v @ Ac @ v), w,
                jac=lambda v: 2 * Ac @ v, method="SLSQP",
                bounds=[(0.0, 1.0)] * K,
                constraints=[{"type": "eq", "fun": lambda v: v.sum() - 1,
                              "jac": lambda v: np.ones(K)}],
                options={"ftol": 1e-15, "maxiter": 3000, "disp": False})
            candidate = np.maximum(refined_result.x, 0.0)
            candidate /= candidate.sum()
        else:
            zstart = np.concatenate([np.maximum(w, 0), np.maximum(-w, 0)])
            sign = np.concatenate([np.ones(K), -np.ones(K)])
            def refine_objective(z):
                v = z[:K] - z[K:]
                return float(v @ Ac @ v)
            def refine_derivative(z):
                g = 2 * Ac @ (z[:K] - z[K:])
                return np.concatenate([g, -g])
            refined_result = minimize(
                refine_objective, zstart, jac=refine_derivative, method="SLSQP",
                bounds=[(0.0, (L + 1.0) / 2)] * K + [(0.0, (L - 1.0) / 2)] * K,
                constraints=[
                    {"type": "eq", "fun": lambda z: sign @ z - 1,
                     "jac": lambda z: sign},
                    {"type": "ineq", "fun": lambda z: L - z.sum(),
                     "jac": lambda z: -np.ones(2 * K)}],
                options={"ftol": 1e-15, "maxiter": 3000, "disp": False})
            candidate = refined_result.x[:K] - refined_result.x[K:]
        candidate_diagnostics = weight_diagnostics(C, candidate, signed_l1)
        if (candidate_diagnostics["constraint_violation"] < 1e-7
                and candidate_diagnostics["objective"] <= diagnostics["objective"] + 1e-11 * scale):
            w = candidate
            diagnostics = candidate_diagnostics
            refinement_iterations = int(refined_result.nit)
    diagnostics = weight_diagnostics(C, w, signed_l1)
    diagnostics.update(
        success=(diagnostics["constraint_violation"] < 1e-7
                 and diagnostics["normalized_kkt_gap"] < 2e-5),
        optimizer_success=bool(result.success), iterations=int(result.nit),
        message=str(result.message), polish_iterations=int(polish_iterations),
        refined=refined, refinement_iterations=refinement_iterations)
    return w, diagnostics


def aggregate_gcv_weights(corrected_weights, denominator):
    """Map the equivalent simplex-QP solution to ordinary aggregate-GCV weights.

    If u minimizes u' corrected_matrix u on the simplex and d_j=1-df_j/n,
    then pi_j proportional to u_j/d_j minimizes aggregate ordinary GCV.
    The actual corrected-risk estimator uses u itself, not this transformed pi.
    """
    u = np.asarray(corrected_weights, dtype=float)
    d = np.asarray(denominator, dtype=float)
    if np.any(d <= 0) or u.shape != d.shape or np.any(u < -1e-12):
        raise ValueError("Positive denominators and simplex weights required")
    pi = np.maximum(u, 0.0) / d
    return pi / pi.sum()


def solve_aggregate_gcv(path):
    """Solve ordinary aggregate GCV via its exact fractional-to-QP mapping."""
    u, diagnostics = solve_weights(path["corrected_matrix"])
    pi = aggregate_gcv_weights(u, path["aggregate_denominator"])
    diagnostics = dict(diagnostics)
    diagnostics["equivalent_corrected_weights"] = u
    diagnostics["aggregate_objective"] = float(
        pi @ path["training_residual_matrix"] @ pi
        / (path["aggregate_denominator"] @ pi) ** 2)
    return pi, diagnostics


def mp_unit_counterexample(lambdas=(0.1, 1.0), weights=(0.5, 0.5)):
    """Exact MP(gamma=1), Sigma=I, alpha2=noise2=1 limit calculations."""
    lam = np.asarray(lambdas, dtype=float)
    w = np.asarray(weights, dtype=float)
    if len(lam) != len(w) or not np.isclose(w.sum(), 1) or np.any(lam <= 0):
        raise ValueError("Positive lambdas and sum-one weights required")
    t = (np.sqrt(1 + 4 / lam) - 1) / 2
    tp = -1 / (lam ** 2 * np.sqrt(1 + 4 / lam))
    K = len(lam)
    B, V = np.empty((K, K)), np.empty((K, K))
    for j in range(K):
        for k in range(K):
            if np.isclose(lam[j], lam[k], rtol=1e-13, atol=0):
                B[j, k] = -tp[j]
                V[j, k] = t[j] + lam[j] * tp[j]
            else:
                B[j, k] = (t[j] - t[k]) / (lam[k] - lam[j])
                V[j, k] = (lam[k] * t[k] - lam[j] * t[j]) / (lam[k] - lam[j])
    return {
        "risk": float(1 + w @ (np.outer(lam, lam) * B + V) @ w),
        "aggregate_gcv": float(w @ (np.outer(lam, lam) * (B + V)) @ w
                               / (w @ (lam * t)) ** 2),
        "corrected_cross_risk": float(w @ ((B + V) / np.outer(t, t)) @ w),
    }
