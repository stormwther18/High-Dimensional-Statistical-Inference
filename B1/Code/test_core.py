"""Independent algebra/optimization checks. Run: python test_core.py."""
import unittest
import numpy as np
from ridge_core import (prepare_design, candidate_path, prior_risk_matrix,
                        single_prior_risk, solve_weights, solve_aggregate_gcv,
                        realized_risk, mp_unit_counterexample)


class RidgeCoreChecks(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261104)

    def test_anisotropic_matrix_and_nullspace_against_direct_inverse(self):
        for n, p in [(17, 11), (11, 17)]:
            mu = np.geomspace(0.15, 3, p)
            psi = np.geomspace(0.3, 0.02, p)
            X = self.rng.normal(size=(n, p)) * np.sqrt(mu)
            design = prepare_design(X, mu, psi, 0.7)
            lam = np.array([0.003, 0.2, 2.0])
            C = prior_risk_matrix(design, lam)
            S = X.T @ X / n
            Q = [np.linalg.inv(S + x * np.eye(p)) for x in lam]
            direct = np.empty_like(C)
            for j in range(len(lam)):
                for k in range(len(lam)):
                    direct[j, k] = (
                        lam[j] * lam[k] * np.trace(np.diag(mu) @ Q[j] @ np.diag(psi) @ Q[k])
                        + 0.7 / n * np.trace(np.diag(mu) @ S @ Q[j] @ Q[k]))
            np.testing.assert_allclose(C, direct, rtol=2e-10, atol=2e-10)
            for j, x in enumerate(lam):
                self.assertAlmostEqual(single_prior_risk(design, x), 0.7 + C[j, j], places=10)
            self.assertGreaterEqual(np.linalg.eigvalsh(C).min(), -1e-10)
            if p > n:
                self.assertEqual(np.count_nonzero(design["s"] == 0), p-n)

    def test_path_and_exact_delete_one_residual(self):
        n, p = 13, 9
        mu = np.geomspace(0.3, 2, p)
        X = self.rng.normal(size=(n, p)) * np.sqrt(mu)
        y = self.rng.normal(size=n)
        lam = np.array([0.04, 0.7])
        design = prepare_design(X, mu, np.ones(p)/p)
        path = candidate_path(design, y, lam)
        explicit = np.empty((n, len(lam)))
        for j, penalty in enumerate(lam):
            direct_beta = np.linalg.solve(X.T @ X + n*penalty*np.eye(p), X.T @ y)
            np.testing.assert_allclose(path["coefficients"][:, j], direct_beta, atol=1e-11)
            for i in range(n):
                keep = np.arange(n) != i
                # Total penalty n*lambda is held fixed after deletion.
                fit = np.linalg.solve(X[keep].T @ X[keep] + n*penalty*np.eye(p),
                                      X[keep].T @ y[keep])
                explicit[i, j] = y[i] - X[i] @ fit
        analytic = path["residuals"] / (1-path["leverages"])
        np.testing.assert_allclose(analytic, explicit, atol=1e-11)
        np.testing.assert_allclose(path["loo_matrix"], explicit.T @ explicit/n, atol=1e-11)
        np.testing.assert_allclose(path["leverages"].sum(axis=0), path["df"], atol=1e-11)

    def test_bayes_ridge_cannot_be_improved_by_fixed_mixture(self):
        n, p, alpha2, sigma2 = 19, 25, 1.4, 0.8
        mu = np.geomspace(0.1, 4, p)
        X = self.rng.normal(size=(n,p))*np.sqrt(mu)
        design = prepare_design(X, mu, np.full(p,alpha2/p), sigma2)
        star = p/n*sigma2/alpha2
        lam = np.array([0.02, 0.1, star, 3.0, 10.0])
        C = prior_risk_matrix(design, lam)
        ridge = single_prior_risk(design, star)
        for L in [None, 3.0]:
            w, diagnostics = solve_weights(C, L)
            self.assertTrue(diagnostics["success"], diagnostics)
            self.assertGreaterEqual(sigma2 + w @ C @ w, ridge-2e-9)
            self.assertLessEqual(sigma2 + w @ C @ w, ridge+2e-7)

    def test_solver_and_fractional_gcv_mapping(self):
        C = np.array([[2.0, 0.2, 0.4], [0.2, 1.0, 0.3], [0.4,0.3,1.7]])
        w, diagnostic = solve_weights(C)
        inverse = np.linalg.solve(C,np.ones(3))
        inverse /= inverse.sum()
        np.testing.assert_allclose(w, inverse, atol=2e-6)
        self.assertLess(diagnostic["normalized_kkt_gap"], 1e-6)
        for L in [1.0, 2.0, 3.0]:
            w, d = solve_weights(C, L)
            self.assertLessEqual(np.abs(w).sum(), L+1e-9)
            self.assertAlmostEqual(w.sum(), 1, places=10)
            self.assertLess(d["normalized_kkt_gap"], 1e-6)
        n,p=17,21
        X=self.rng.normal(size=(n,p))
        design=prepare_design(X,np.ones(p),np.ones(p)/p)
        path=candidate_path(design,self.rng.normal(size=n),[.03,.2,1.5])
        pi,d=solve_aggregate_gcv(path)
        self.assertAlmostEqual(d["objective"],d["aggregate_objective"],places=9)
        best=d["aggregate_objective"]
        for candidate in self.rng.dirichlet(np.ones(3),size=100):
            val=(candidate@path["training_residual_matrix"]@candidate
                 /(path["aggregate_denominator"]@candidate)**2)
            self.assertLessEqual(best,val+1e-8)

    def test_realized_risk_and_mp_counterexample(self):
        coef=np.array([[1.,2.],[3.,4.]])
        beta=np.array([.5,1.])
        mu=np.array([2.,.2])
        np.testing.assert_allclose(realized_risk(coef,beta,mu),[2.3,7.3])
        for lam,truth,gcv in [((.1,1.),1.7875250569921637,1.680756692306628),
                             ((.01,10.),2.443469136398286,1.7699549947094972)]:
            out=mp_unit_counterexample(lam)
            self.assertAlmostEqual(out["risk"],truth,places=11)
            self.assertAlmostEqual(out["aggregate_gcv"],gcv,places=11)
            self.assertAlmostEqual(out["corrected_cross_risk"],truth,places=11)
            self.assertGreater(out["risk"],out["aggregate_gcv"])

    def test_collinear_signed_qp_global_certificate(self):
        # Regression case from the completed experiment: SLSQP's local exit
        # flag was successful, yet the original global gap was 4.04e-5.
        rng = np.random.default_rng(3459721514)
        n = p = 100
        index = np.arange(1, p+1, dtype=float)
        mu = index**(-1.5)
        mu /= mu.mean()
        psi = index**0.3
        psi /= mu @ psi
        X = rng.normal(size=(n,p)) * np.sqrt(mu)
        design = prepare_design(X, mu, psi)
        lambdas = np.unique(np.r_[np.logspace(-4,2,21), 1.])
        C = prior_risk_matrix(design, lambdas)
        w, diag = solve_weights(C, signed_l1=3.)
        self.assertTrue(diag["success"], diag)
        self.assertLess(diag["normalized_kkt_gap"], 2e-5)
        self.assertLessEqual(w @ C @ w, 0.3220337)


if __name__ == "__main__":
    unittest.main(verbosity=2)
