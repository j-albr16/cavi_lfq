import math
from unittest import TestCase

import jax.numpy as jnp
import numpy as np

from cavi.variational_updates import carbon_count, heavy_isotope, m_k, nig, peak_width, pi, q_c
from cavi.priors import CHARGES, DELTA, HyperParameters, VariationalParams


class TestVariationalUpdates(TestCase):

    def setUp(self):
        K, G, N, J = 3, 4, 20, 3
        rng = np.random.default_rng(0)
        c_grid = jnp.array([10, 11, 12, 13], dtype=jnp.int32)
        p = HyperParameters(
            alpha=jnp.full((K + 1,), 1.5),
            m_0=jnp.array([100.0, 200.0, 300.0]),
            nu_0=jnp.full((K,), 4.0),
            a_0=jnp.full((K,), 3.0),
            b_0=jnp.full((K,), 2.0),
            m_hat=jnp.array([500.0, 501.0, 502.0]),
            tau=jnp.full((K,), 0.1),
            rho=jnp.array(rng.dirichlet(np.ones(4), K), dtype=jnp.float32),
            n=jnp.full((K, G), 1.0 / G),
            c_grid=c_grid,
            a_p=jnp.array(2.0),
            b_p=jnp.array(200.0),
            a_s=jnp.array(3.0),
            b_s=jnp.array(1.0),
        )
        self.v = VariationalParams(
            alpha=p.alpha,
            a_p=p.a_p,
            b_p=p.b_p,
            a_s=p.a_s,
            b_s=p.b_s,
            q_n=p.n,
            q_c=p.rho,
            mu=p.m_0,
            nu=p.nu_0,
            a_k=p.a_0,
            b_k=p.b_0,
            tau=jnp.array([0.3, 0.2, 0.1]),
            m=jnp.array(rng.normal(500.0, 1.0, (K, 4)), dtype=jnp.float32),
        )
        self.K, self.N_pix, self.J = K, N, J
        self.p = p
        self.y = jnp.array(rng.normal(500.5, 0.5, N), dtype=jnp.float32)
        self.t = jnp.array(rng.normal(150.0, 20.0, N), dtype=jnp.float32)
        self.w = jnp.array(rng.uniform(0.5, 1.0, N), dtype=jnp.float32)
        self.r = jnp.array(rng.uniform(0.0, 0.1, (N, K, J)), dtype=jnp.float32)
        self.R = self.r.sum(axis=2)  # N K
        self.N_k = (self.w[:, None] * self.R).sum(axis=0)  # K

    def test_pi(self):
        N = jnp.arange(self.K + 1, dtype=jnp.float32)
        np.testing.assert_allclose(pi(self.p, N), self.p.alpha + N)

    def test_nig(self):
        nu, mu, a, b = nig(self.p, self.R, self.N_k, self.w, self.t)
        w, R, t = map(np.asarray, (self.w, self.R, self.t))
        for k in range(self.K):
            nu_0, m_0 = float(self.p.nu_0[k]), float(self.p.m_0[k])
            N_k = (w * R[:, k]).sum()
            mu_ref = (nu_0 * m_0 + (w * R[:, k] * t).sum()) / (nu_0 + N_k)
            b_ref = float(self.p.b_0[k]) + 0.5 * (
                nu_0 * (m_0 - mu_ref) ** 2 + (w * R[:, k] * (t - mu_ref) ** 2).sum()
            )
            np.testing.assert_allclose(nu[k], nu_0 + N_k, rtol=1e-5)
            np.testing.assert_allclose(mu[k], mu_ref, rtol=1e-5)
            np.testing.assert_allclose(a[k], float(self.p.a_0[k]) + N_k / 2, rtol=1e-5)
            np.testing.assert_allclose(b[k], b_ref, rtol=1e-4)

    def test_m_k(self):
        lam = float(self.v.a_s / self.v.b_s)
        w, r, y = map(np.asarray, (self.w, self.r, self.y))
        tau2, m = m_k(self.v, self.p, self.y, self.w, self.r, self.N_k)
        self.assertEqual(tau2.shape, (self.K,))
        self.assertEqual(m.shape, (self.K, 4))
        for k in range(self.K):
            tau_0 = float(self.p.tau[k])
            for c in range(4):
                u = y[:, None] - np.arange(self.J)[None, :] * DELTA / float(CHARGES[c])
                N_k = (w[:, None] * r[:, k, :]).sum()
                tau2_ref = 1 / (lam * N_k + 1 / tau_0**2)
                m_ref = tau2_ref * (
                    lam * (w[:, None] * r[:, k, :] * u).sum()
                    + float(self.p.m_hat[k]) / tau_0**2
                )
                np.testing.assert_allclose(tau2[k], tau2_ref, rtol=1e-4)
                np.testing.assert_allclose(m[k, c], m_ref, rtol=1e-5)

    def test_q_c_is_distribution(self):
        q = q_c(self.v, self.p, self.y, self.w, self.r, self.v.m)
        self.assertEqual(q.shape, (self.K, 4))
        self.assertTrue(bool(jnp.all(q >= 0)))
        np.testing.assert_allclose(q.sum(axis=1), 1.0, atol=1e-5)

    def test_q_c_matches_numerical_marginal(self):
        """q(c) must equal rho_c * integral over m of the joint, by brute force."""
        lam = float(self.v.a_s / self.v.b_s)
        w, r, y = map(np.asarray, (self.w, self.r, self.y))
        # the update is evaluated at the new m_tilde, as in the CAVI order
        _, m_new = m_k(self.v, self.p, self.y, self.w, self.r, self.N_k)
        q = q_c(self.v, self.p, self.y, self.w, self.r, m_new)
        self.assertEqual(q.shape, (self.K, 4))

        for k in range(self.K):
            tau_0 = float(self.p.tau[k])
            m_hat = float(self.p.m_hat[k])
            grid = np.linspace(m_hat - 5, m_hat + 5, 200001)
            log_marg = []
            for c in range(4):
                u = y[:, None] - np.arange(self.J)[None, :] * DELTA / float(CHARGES[c])
                res = (u[None] - grid[:, None, None]) ** 2  # G N J
                quad = lam * (w[None, :, None] * r[None, :, k, :] * res).sum(axis=(1, 2))
                log_joint = -0.5 * quad - 0.5 * (grid - m_hat) ** 2 / tau_0**2
                mx = log_joint.max()
                integral = np.exp(log_joint - mx).sum() * (grid[1] - grid[0])
                log_marg.append(np.log(float(self.p.rho[k, c])) + mx + np.log(integral))
            log_marg = np.array(log_marg)
            ref = np.exp(log_marg - log_marg.max())
            ref /= ref.sum()
            np.testing.assert_allclose(q[k], ref, atol=2e-3)

    def test_peak_width(self):
        qc = self.p.rho
        a_s, b_s = peak_width(
            self.v, self.p, self.N_k, self.w, self.r, qc, self.y
        )
        w, r, y, m, tau2, qc_ = map(
            np.asarray, (self.w, self.r, self.y, self.v.m, self.v.tau, qc)
        )
        acc = 0.0
        for n in range(self.N_pix):
            for k in range(self.K):
                for j in range(self.J):
                    e = tau2[k] + sum(
                        qc_[k, c] * (m[k, c] + j * DELTA / float(CHARGES[c]) - y[n]) ** 2
                        for c in range(4)
                    )
                    acc += w[n] * r[n, k, j] * e
        np.testing.assert_allclose(a_s, float(self.p.a_s) + 0.5 * float(self.N_k.sum()), rtol=1e-5)
        np.testing.assert_allclose(b_s, float(self.p.b_s) + 0.5 * acc, rtol=1e-4)

    def test_carbon_count(self):
        q = carbon_count(self.v, self.p, self.N_k, self.w, self.r)
        G = self.p.c_grid.shape[0]
        self.assertEqual(q.shape, (self.K, G))
        self.assertTrue(bool(jnp.all(q >= 0)))
        np.testing.assert_allclose(q.sum(axis=1), 1.0, atol=1e-5)

        def log_binom(n, j):
            return math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)

        from scipy.special import digamma

        e_log_1mp = digamma(float(self.v.b_p)) - digamma(float(self.v.a_p + self.v.b_p))
        w, r = np.asarray(self.w), np.asarray(self.r)
        for k in range(self.K):
            ell = np.zeros(G)
            for g in range(G):
                C = int(self.p.c_grid[g])
                bin_sum = sum(
                    w[n] * r[n, k, j] * log_binom(C, j)
                    for n in range(self.N_pix)
                    for j in range(self.J)
                )
                ell[g] = (
                    np.log(float(self.p.n[k, g]))
                    + bin_sum
                    + C * float(self.N_k[k]) * e_log_1mp
                )
            ref = np.exp(ell - ell.max())
            ref /= ref.sum()
            np.testing.assert_allclose(q[k], ref, rtol=1e-4, atol=1e-6)

    def test_carbon_count_follows_prior(self):
        """With no assignments (R = 0) q(n) reduces to the prior."""
        r0 = jnp.zeros_like(self.r)
        prior = jnp.array(np.random.default_rng(3).dirichlet(np.ones(4), self.K), dtype=jnp.float32)
        p = self.p._replace(n=prior)
        q = carbon_count(self.v, p, jnp.zeros(self.K), self.w, r0)
        np.testing.assert_allclose(q, prior, atol=1e-5)

    def test_heavy_isotope(self):
        q_n = jnp.array(
            np.random.default_rng(5).dirichlet(np.ones(4), self.K), dtype=jnp.float32
        )
        a_p, b_p = heavy_isotope(self.p, self.N_k, self.w, self.r, q_n)
        self.assertEqual(a_p.shape, ())
        self.assertEqual(b_p.shape, ())

        w, r = np.asarray(self.w), np.asarray(self.r)
        S = sum(
            w[n] * r[n, k, j] * j
            for n in range(self.N_pix)
            for k in range(self.K)
            for j in range(self.J)
        )
        e_n = np.asarray(q_n) @ np.asarray(self.p.c_grid)
        total_trials = sum(float(self.N_k[k]) * e_n[k] for k in range(self.K))
        np.testing.assert_allclose(a_p, float(self.p.a_p) + S, rtol=1e-5)
        np.testing.assert_allclose(b_p, float(self.p.b_p) + total_trials - S, rtol=1e-5)

    def test_heavy_isotope_no_assignments_is_prior(self):
        q_n = self.p.n
        a_p, b_p = heavy_isotope(
            self.p, jnp.zeros(self.K), self.w, jnp.zeros_like(self.r), q_n
        )
        np.testing.assert_allclose(a_p, self.p.a_p)
        np.testing.assert_allclose(b_p, self.p.b_p)

    def test_heavy_isotope_counts_add_up(self):
        """a_tilde + b_tilde - a - b equals the expected number of carbon atoms."""
        a_p, b_p = heavy_isotope(self.p, self.N_k, self.w, self.r, self.p.n)
        e_n = np.asarray(self.p.n) @ np.asarray(self.p.c_grid)
        expected = float(jnp.sum(self.N_k * e_n))
        np.testing.assert_allclose(
            a_p + b_p - self.p.a_p - self.p.b_p, expected, rtol=1e-5
        )
        self.assertGreater(float(b_p), 0.0)
