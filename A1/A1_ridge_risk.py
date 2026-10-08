"""Reproduce Dobriban--Wager (2018), Section 2, for identity covariance.

Run with Python and NumPy. Outputs an SVG figure and CSV data beside this file.
The ridge convention is (X.T @ X + n * lam * I)^(-1) X.T @ y.
"""

from pathlib import Path
import csv
import math
import numpy as np


ROOT = Path(__file__).resolve().parent
ALPHA2 = 1.0
N = 120
REPS = 12
GAMMAS = (0.5, 1.0, 2.0)
LAMBDAS = np.geomspace(0.03, 10.0, 180)
COLORS = ("#277da1", "#d1495b", "#648c45")


def theoretical_risk(gamma, lam, alpha2=ALPHA2):
    """Theorem 2.1 for H = point mass at 1 and noise variance 1."""
    lam = np.asarray(lam, dtype=float)
    b = 1.0 - gamma + lam
    m = 2.0 / (b + np.sqrt(b * b + 4.0 * gamma * lam))
    m_prime = (gamma * m * m + m) / (2.0 * gamma * lam * m + b)
    return 1.0 + gamma * m + lam * (lam * alpha2 - gamma) * m_prime


def finite_sample_conditional_risk(singular_eigenvalues, p, n, lam):
    """E[(y0 - x0.T w_hat)^2 | X], averaging over w, noise, and test data."""
    lam = np.asarray(lam, dtype=float)
    s = singular_eigenvalues[:, None]
    bias = (ALPHA2 / p) * np.sum((lam / (s + lam)) ** 2, axis=0)
    if p > n:
        bias += ALPHA2 * (p - n) / p
    variance = np.sum(s / (s + lam) ** 2, axis=0) / n
    return 1.0 + bias + variance


def simulate():
    rng = np.random.default_rng(20261002)
    points = np.geomspace(0.04, 9.0, 13)
    result = {}
    for gamma in GAMMAS:
        p = round(gamma * N)
        risks = []
        for _ in range(REPS):
            x = rng.standard_normal((N, p))
            # Nonzero eigenvalues of X.T X / n equal those of X X.T / n.
            if p <= N:
                s = np.linalg.eigvalsh(x.T @ x / N)
            else:
                s = np.linalg.eigvalsh(x @ x.T / N)
            risks.append(finite_sample_conditional_risk(s, p, N, points))
        result[gamma] = (points, np.mean(risks, axis=0))
    return result


def svg_figure(simulated):
    width, height = 1060, 490
    left = (80, 65, 485, 375)
    right = (615, 65, 485, 375)
    ymin, ymax = 1.0, 3.7
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font:15px Arial,sans-serif;fill:#253348} '
        '.title{font-size:19px;font-weight:bold}.small{font-size:13px}</style>',
        '<text class="title" x="80" y="32">Ridge prediction risk: theory and finite-sample calculation</text>',
    ]

    def panel(box, xlabel):
        x, y, w, h = box
        for tick in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5):
            yy = y + h * (ymax - tick) / (ymax - ymin)
            parts.append(f'<line x1="{x}" x2="{x+w}" y1="{yy:.2f}" y2="{yy:.2f}" stroke="#e3e8ee"/>')
            parts.append(f'<text class="small" x="{x-11}" y="{yy+5:.2f}" text-anchor="end">{tick:g}</text>')
        parts.append(f'<path d="M{x},{y} V{y+h} H{x+w}" fill="none" stroke="#556274" stroke-width="1.5"/>')
        parts.append(f'<text x="{x+w/2}" y="{height-18}" text-anchor="middle">{xlabel}</text>')

    panel(left, "Regularization λ (log scale)")
    panel(right, "Dimension ratio γ = p/n")
    parts.append('<text x="18" y="250" transform="rotate(-90 18 250)" text-anchor="middle">Prediction MSE</text>')

    def py(value, box):
        return box[1] + box[3] * (ymax - value) / (ymax - ymin)

    def px_log(value):
        return left[0] + left[2] * (math.log(value) - math.log(0.03)) / (math.log(10) - math.log(0.03))

    def px_gamma(value):
        return right[0] + right[2] * (value - 0.25) / (2.5 - 0.25)

    for tick in (0.03, 0.1, 0.3, 1, 3, 10):
        xx = px_log(tick)
        parts.append(f'<line x1="{xx:.2f}" x2="{xx:.2f}" y1="{left[1]+left[3]}" y2="{left[1]+left[3]+5}" stroke="#556274"/>')
        parts.append(f'<text class="small" x="{xx:.2f}" y="{left[1]+left[3]+22}" text-anchor="middle">{tick:g}</text>')

    for gamma, color in zip(GAMMAS, COLORS):
        risk = theoretical_risk(gamma, LAMBDAS)
        path = " ".join(f'{"M" if i == 0 else "L"}{px_log(lam):.2f},{py(val,left):.2f}' for i, (lam, val) in enumerate(zip(LAMBDAS, risk)))
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        xp, yp = simulated[gamma]
        for lam, val in zip(xp, yp):
            parts.append(f'<circle cx="{px_log(lam):.2f}" cy="{py(val,left):.2f}" r="3.2" fill="white" stroke="{color}" stroke-width="1.7"/>')
        optimal = gamma / ALPHA2
        parts.append(f'<circle cx="{px_log(optimal):.2f}" cy="{py(theoretical_risk(gamma,optimal),left):.2f}" r="5.2" fill="{color}"/>')

    gs = np.linspace(0.25, 2.5, 220)
    for tick in (0.5, 1, 1.5, 2, 2.5):
        xx = px_gamma(tick)
        parts.append(f'<line x1="{xx:.2f}" x2="{xx:.2f}" y1="{right[1]+right[3]}" y2="{right[1]+right[3]+5}" stroke="#556274"/>')
        parts.append(f'<text class="small" x="{xx:.2f}" y="{right[1]+right[3]+22}" text-anchor="middle">{tick:g}</text>')
    for lam, color in zip((0.1, 1.0, 4.0), COLORS):
        risk = theoretical_risk(gs, lam)
        path = " ".join(f'{"M" if i == 0 else "L"}{px_gamma(g):.2f},{py(val,right):.2f}' for i, (g, val) in enumerate(zip(gs, risk)))
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.5"/>')

    for i, (gamma, color) in enumerate(zip(GAMMAS, COLORS)):
        x, y = 245 + i * 107, 55
        parts.append(f'<line x1="{x}" x2="{x+20}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="2.5"/>')
        parts.append(f'<text class="small" x="{x+25}" y="{y+5}">γ={gamma:g}</text>')
    for i, (lam, color) in enumerate(zip((0.1, 1.0, 4.0), COLORS)):
        x, y = 765 + i * 96, 55
        parts.append(f'<line x1="{x}" x2="{x+20}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="2.5"/>')
        parts.append(f'<text class="small" x="{x+24}" y="{y+5}">λ={lam:g}</text>')
    parts.append('<text class="small" x="95" y="89">Lines: asymptotic theory; open circles: mean finite-sample risk; solid dots: λ*=γ</text>')
    parts.append('<text class="small" x="630" y="89">Asymptotic theory for fixed λ</text>')
    parts.append('</svg>')
    (ROOT / "A1_ridge_risk.svg").write_text("\n".join(parts), encoding="utf-8")


def write_data(simulated):
    with (ROOT / "A1_ridge_risk.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("gamma", "lambda", "theory_risk", "finite_sample_mean_risk", "n", "repetitions"))
        for gamma in GAMMAS:
            lambdas, empirical = simulated[gamma]
            for lam, value in zip(lambdas, empirical):
                writer.writerow((gamma, lam, float(theoretical_risk(gamma, lam)), value, N, REPS))


if __name__ == "__main__":
    simulations = simulate()
    write_data(simulations)
    svg_figure(simulations)
    for gamma in GAMMAS:
        lam = gamma / ALPHA2
        print(f"gamma={gamma:g}, lambda*={lam:g}, theory minimum={theoretical_risk(gamma,lam):.4f}")
    print("Wrote A1_ridge_risk.svg and A1_ridge_risk.csv")
