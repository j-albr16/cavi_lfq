from unittest import TestCase

import jax.numpy as jnp
import numpy as np
from scipy import integrate, stats
from scipy.special import logsumexp

from cavi.variational_updates import (
    carbon_count,
    heavy_isotope,
    m_k,
    nig,
    peak_width,
    pi,
    q_c,
)
from cavi.z_j import get_log_rho_all, get_r
from cavi.elbo import (
    elbo,
    kl_beta,
    kl_carbon_count,
    kl_categorical,
    kl_dirichlet,
    kl_heavy_isotope,
    kl_inverse_gamma,
    kl_mass_charge,
    kl_nig,
    kl_normal,
    kl_peak_width,
    kl_pi,
    kl_total,
)
from cavi.priors import HyperParameters, VariationalParams


def numeric_kl_1d(logpdf_q, logpdf_p, lo, hi):
    """KL = int q (log q - log p) by quadrature."""
    f = lambda x: np.exp(logpdf_q(x)) * (logpdf_q(x) - logpdf_p(x))
    return integrate.quad(f, lo, hi, limit=500, epsabs=1e-10, epsrel=1e-10)[0]


class TestKLPrimitives(TestCase):

    def test_categorical(self):
        q = np.array([0.2, 0.5, 0.3, 0.0])
        p = np.array([0.25, 0.25, 0.25, 0.25])
        ref = sum(qi * np.log(qi / pi_) for qi, pi_ in zip(q, p) if qi > 0)
        np.testing.assert_allclose(kl_categorical(jnp.array(q), jnp.array(p)), ref, rtol=1e-5)

    def test_categorical_batched_and_properties(self):
        rng = np.random.default_rng(0)
        q = rng.dirichlet(np.ones(4), 5)
        p = rng.dirichlet(np.ones(4), 5)
        kl = kl_categorical(jnp.array(q), jnp.array(p))
        self.assertEqual(kl.shape, (5,))
        self.assertTrue(bool(jnp.all(kl >= -1e-7)))
        np.testing.assert_allclose(kl_categorical(jnp.array(q), jnp.array(q)), 0.0, atol=1e-6)

    def test_normal(self):
        m_q, v_q, m_p, v_p = 0.4, 0.8, -0.3, 1.7
        ref = numeric_kl_1d(
            lambda x: stats.norm.logpdf(x, m_q, np.sqrt(v_q)),
            lambda x: stats.norm.logpdf(x, m_p, np.sqrt(v_p)),
            -20, 20,
        )
        np.testing.assert_allclose(kl_normal(m_q, v_q, m_p, v_p), ref, rtol=1e-5)

    def test_beta(self):
        a_q, b_q, a_p, b_p = 3.0, 7.0, 2.0, 5.5
        ref = numeric_kl_1d(
            lambda x: stats.beta.logpdf(x, a_q, b_q),
            lambda x: stats.beta.logpdf(x, a_p, b_p),
            1e-12, 1 - 1e-12,
        )
        np.testing.assert_allclose(kl_beta(a_q, b_q, a_p, b_p), ref, rtol=1e-4)

    def test_inverse_gamma(self):
        a_q, b_q, a_p, b_p = 4.0, 2.5, 3.0, 1.0
        ref = numeric_kl_1d(
            lambda x: stats.invgamma.logpdf(x, a_q, scale=b_q),
            lambda x: stats.invgamma.logpdf(x, a_p, scale=b_p),
            1e-9, 500,
        )
        np.testing.assert_allclose(kl_inverse_gamma(a_q, b_q, a_p, b_p), ref, rtol=1e-4)

    def test_dirichlet_matches_monte_carlo(self):
        a_q = np.array([2.0, 3.5, 1.2, 6.0])
        a_p = np.array([1.5, 1.5, 1.5, 1.5])
        x = stats.dirichlet.rvs(a_q, size=400_000, random_state=1)
        x = np.clip(x, 1e-300, None)
        x /= x.sum(axis=1, keepdims=True)
        ref = np.mean(stats.dirichlet.logpdf(x.T, a_q) - stats.dirichlet.logpdf(x.T, a_p))
        np.testing.assert_allclose(kl_dirichlet(jnp.array(a_q), jnp.array(a_p)), ref, atol=2e-2)

    def test_dirichlet_two_components_is_beta(self):
        a_q, a_p = np.array([3.0, 7.0]), np.array([2.0, 5.5])
        np.testing.assert_allclose(
            kl_dirichlet(jnp.array(a_q), jnp.array(a_p)),
            kl_beta(a_q[0], a_q[1], a_p[0], a_p[1]),
            rtol=1e-5,
        )

    def test_zero_when_equal(self):
        np.testing.assert_allclose(kl_dirichlet(jnp.array([1.0, 2.0, 3.0]), jnp.array([1.0, 2.0, 3.0])), 0.0, atol=1e-5)
        np.testing.assert_allclose(kl_beta(2.0, 3.0, 2.0, 3.0), 0.0, atol=1e-5)
        np.testing.assert_allclose(kl_inverse_gamma(2.0, 3.0, 2.0, 3.0), 0.0, atol=1e-5)
        np.testing.assert_allclose(kl_normal(1.0, 2.0, 1.0, 2.0), 0.0, atol=1e-6)


class ModelMixin:
    """Small model with K=2 features shared by the wrapper and ELBO tests."""

    def build(self):
        K, G, self.N_pix, self.J = 2, 4, 60, 3
        self.K = K
        rng = np.random.default_rng(0)
        self.h = HyperParameters(
            alpha=jnp.array([2.0, 1.5, 1.2]),
            m_0=jnp.array([50.0, 120.0]),
            nu_0=jnp.array([1.0, 2.0]),
            a_0=jnp.array([3.0, 2.5]),
            b_0=jnp.array([3.0, 2.0]),
            m_hat=jnp.array([500.0, 501.0]),
            tau=jnp.array([0.05, 0.08]),
            rho=jnp.array([[0.1, 0.6, 0.2, 0.1], [0.25, 0.25, 0.25, 0.25]], dtype=jnp.float32),
            n=jnp.array(rng.dirichlet(np.ones(G), K), dtype=jnp.float32),
            c_grid=jnp.array([10, 11, 12, 13], dtype=jnp.int32),
            a_p=jnp.array(2.0),
            b_p=jnp.array(200.0),
            a_s=jnp.array(3.0),
            b_s=jnp.array(0.02),
        )
        self.v0 = VariationalParams(  # q initialized at the prior
            alpha=self.h.alpha,
            a_p=self.h.a_p,
            b_p=self.h.b_p,
            a_s=self.h.a_s,
            b_s=self.h.b_s,
            q_n=self.h.n,
            q_c=self.h.rho,
            mu=self.h.m_0,
            nu=self.h.nu_0,
            a_k=self.h.a_0,
            b_k=self.h.b_0,
            tau=self.h.tau**2,
            m=jnp.broadcast_to(self.h.m_hat[:, None], (K, 4)),
        )
        # data: two combs (charge 2) plus uniform background
        ys, ts = [], []
        for k in range(K):
            for _ in range(15):
                j = rng.integers(0, self.J)
                ys.append(float(self.h.m_hat[k]) + j / 2 + rng.normal(0, 0.1))
                ts.append(float(self.h.m_0[k]) + rng.normal(0, 1.0))
        for _ in range(self.N_pix - len(ys)):
            ys.append(rng.uniform(450, 550))
            ts.append(rng.uniform(0, 200))
        self.y = jnp.array(ys, dtype=jnp.float32)
        self.t = jnp.array(ts, dtype=jnp.float32)
        self.w = jnp.array(rng.uniform(1.0, 2.0, self.N_pix), dtype=jnp.float32)
        self.T, self.Y = 200, 100


class TestKLPerDistribution(ModelMixin, TestCase):

    def setUp(self):
        self.build()
        rng = np.random.default_rng(7)
        K = self.K
        # a variational distribution that differs from the prior everywhere
        self.v = self.v0._replace(
            alpha=jnp.array([5.0, 12.0, 8.0]),
            a_p=jnp.array(30.0),
            b_p=jnp.array(2900.0),
            a_s=jnp.array(40.0),
            b_s=jnp.array(0.5),
            q_n=jnp.array(rng.dirichlet(np.ones(4), K), dtype=jnp.float32),
            q_c=jnp.array(rng.dirichlet(np.ones(4), K), dtype=jnp.float32),
            mu=jnp.array([50.4, 119.5]),
            nu=jnp.array([6.0, 9.0]),
            a_k=jnp.array([10.0, 8.0]),
            b_k=jnp.array([9.0, 7.0]),
            tau=jnp.array([0.001, 0.002]),
            m=jnp.array(rng.normal(0.0, 0.03, (K, 4)) + np.array([[500.0], [501.0]]), dtype=jnp.float32),
        )

    def test_kl_pi(self):
        np.testing.assert_allclose(
            kl_pi(self.v, self.h), kl_dirichlet(self.v.alpha, self.h.alpha), rtol=1e-6
        )
        self.assertGreater(float(kl_pi(self.v, self.h)), 0.0)

    def test_kl_nig_matches_2d_quadrature(self):
        kl = kl_nig(self.v, self.h)
        self.assertEqual(kl.shape, (self.K,))
        for k in range(self.K):
            m, nu, a, b = (float(x[k]) for x in (self.v.mu, self.v.nu, self.v.a_k, self.v.b_k))
            m0, nu0, a0, b0 = (float(x[k]) for x in (self.h.m_0, self.h.nu_0, self.h.a_0, self.h.b_0))

            def log_q(mu, s2):
                return stats.norm.logpdf(mu, m, np.sqrt(s2 / nu)) + stats.invgamma.logpdf(s2, a, scale=b)

            def log_p(mu, s2):
                return stats.norm.logpdf(mu, m0, np.sqrt(s2 / nu0)) + stats.invgamma.logpdf(s2, a0, scale=b0)

            f = lambda mu, s2: np.exp(log_q(mu, s2)) * (log_q(mu, s2) - log_p(mu, s2))
            ref = integrate.dblquad(
                f,
                1e-4,
                80.0,
                lambda s2: m - 12 * np.sqrt(s2 / nu),
                lambda s2: m + 12 * np.sqrt(s2 / nu),
                epsabs=1e-9,
                epsrel=1e-9,
            )[0]
            np.testing.assert_allclose(kl[k], ref, rtol=2e-4)

    def test_kl_mass_charge(self):
        kl = kl_mass_charge(self.v, self.h)
        self.assertEqual(kl.shape, (self.K,))
        for k in range(self.K):
            q, rho = np.asarray(self.v.q_c[k]), np.asarray(self.h.rho[k])
            ref = float(np.sum(q * np.log(q / rho)))
            for c in range(4):
                ref += q[c] * numeric_kl_1d(
                    lambda x: stats.norm.logpdf(x, float(self.v.m[k, c]), np.sqrt(float(self.v.tau[k]))),
                    lambda x: stats.norm.logpdf(x, float(self.h.m_hat[k]), float(self.h.tau[k])),
                    float(self.h.m_hat[k]) - 2,
                    float(self.h.m_hat[k]) + 2,
                )
            np.testing.assert_allclose(kl[k], ref, rtol=1e-3)

    def test_kl_carbon_count(self):
        kl = kl_carbon_count(self.v, self.h)
        q, p = np.asarray(self.v.q_n), np.asarray(self.h.n)
        np.testing.assert_allclose(kl, (q * np.log(q / p)).sum(axis=1), rtol=1e-5)

    def test_kl_heavy_isotope(self):
        ref = numeric_kl_1d(
            lambda x: stats.beta.logpdf(x, 30.0, 2900.0),
            lambda x: stats.beta.logpdf(x, 2.0, 200.0),
            1e-12, 0.3,
        )
        np.testing.assert_allclose(kl_heavy_isotope(self.v, self.h), ref, rtol=1e-3)

    def test_kl_peak_width(self):
        ref = numeric_kl_1d(
            lambda x: stats.invgamma.logpdf(x, 40.0, scale=0.5),
            lambda x: stats.invgamma.logpdf(x, 3.0, scale=0.02),
            1e-6, 5.0,
        )
        np.testing.assert_allclose(kl_peak_width(self.v, self.h), ref, rtol=1e-3)

    def test_kl_total_is_sum_of_parts(self):
        total = (
            kl_pi(self.v, self.h)
            + kl_nig(self.v, self.h).sum()
            + kl_mass_charge(self.v, self.h).sum()
            + kl_carbon_count(self.v, self.h).sum()
            + kl_heavy_isotope(self.v, self.h)
            + kl_peak_width(self.v, self.h)
        )
        np.testing.assert_allclose(kl_total(self.v, self.h), total, rtol=1e-6)
        self.assertGreater(float(kl_total(self.v, self.h)), 0.0)

    def test_kl_zero_at_prior(self):
        np.testing.assert_allclose(kl_total(self.v0, self.h), 0.0, atol=1e-4)


class TestELBO(ModelMixin, TestCase):

    def setUp(self):
        self.build()

    def _evaluate(self, v, r, r_0):
        log_rho, log_rho_0 = get_log_rho_all(v, self.h, self.y, self.t, self.J, self.T, self.Y)
        return float(elbo(v, self.h, self.w, r, r_0, log_rho, log_rho_0))

    def test_matches_loop_reference(self):
        v = self.v0
        log_rho, log_rho_0 = get_log_rho_all(v, self.h, self.y, self.t, self.J, self.T, self.Y)
        # arbitrary (not optimal) normalized responsibilities
        rng = np.random.default_rng(2)
        raw = rng.uniform(0.01, 1.0, (self.N_pix, self.K * self.J + 1))
        raw /= raw.sum(axis=1, keepdims=True)
        r = jnp.array(raw[:, 1:].reshape(self.N_pix, self.K, self.J), dtype=jnp.float32)
        r_0 = jnp.array(raw[:, 0], dtype=jnp.float32)

        got = elbo(v, self.h, self.w, r, r_0, log_rho, log_rho_0)

        w, r_, r0_, lr, lr0 = map(np.asarray, (self.w, r, r_0, log_rho, log_rho_0))
        data = 0.0
        for n in range(self.N_pix):
            s = r0_[n] * (lr0 - np.log(r0_[n]))
            for k in range(self.K):
                for j in range(self.J):
                    s += r_[n, k, j] * (lr[n, k, j] - np.log(r_[n, k, j]))
            data += w[n] * s
        np.testing.assert_allclose(got, data - float(kl_total(v, self.h)), rtol=1e-4)

    def test_optimal_r_gives_logsumexp(self):
        """For the optimal r the pixel term collapses to w_n log(rho_0 + sum rho)."""
        v = self.v0
        log_rho, log_rho_0 = get_log_rho_all(v, self.h, self.y, self.t, self.J, self.T, self.Y)
        r, r_0 = get_r(v, self.h, self.y, self.t, self.J, self.T, self.Y)
        got = elbo(v, self.h, self.w, r, r_0, log_rho, log_rho_0)
        lse = np.logaddexp(
            float(log_rho_0),
            logsumexp(np.asarray(log_rho), axis=(1, 2)),
        )
        ref = float(np.sum(np.asarray(self.w) * lse)) - float(kl_total(v, self.h))
        np.testing.assert_allclose(got, ref, rtol=1e-4)

    def test_optimal_r_beats_perturbed_r(self):
        v = self.v0
        r, r_0 = get_r(v, self.h, self.y, self.t, self.J, self.T, self.Y)
        best = self._evaluate(v, r, r_0)
        rng = np.random.default_rng(4)
        for _ in range(5):
            # near one-hot r is insensitive to multiplicative noise, so mix with a random distribution
            other = rng.dirichlet(np.ones(self.K * self.J + 1), self.N_pix)
            full = np.concatenate([np.asarray(r_0)[:, None], np.asarray(r).reshape(self.N_pix, -1)], axis=1)
            full = 0.7 * full + 0.3 * other
            r_p = jnp.array(full[:, 1:].reshape(self.N_pix, self.K, self.J), dtype=jnp.float32)
            r0_p = jnp.array(full[:, 0], dtype=jnp.float32)
            self.assertLess(self._evaluate(v, r_p, r0_p), best)

    def test_cavi_updates_never_decrease_elbo(self):
        """Every coordinate update must increase (or keep) the ELBO."""
        h, y, t, w, J = self.h, self.y, self.t, self.w, self.J
        v = self.v0
        trace = []

        def record(name, r, r_0):
            trace.append((name, self._evaluate(v, r, r_0)))

        for _ in range(6):
            r, r_0 = get_r(v, h, y, t, J, self.T, self.Y)
            record("q(z,j)", r, r_0)
            R = r.sum(axis=2)
            N_k = jnp.sum(w[:, None] * R, axis=0)
            N_0 = jnp.sum(w * r_0)

            v = v._replace(alpha=pi(h, jnp.concatenate([N_0[None], N_k])))
            record("q(pi)", r, r_0)

            nu, mu, a, b = nig(h, R, N_k, w, t)
            v = v._replace(nu=nu, mu=mu, a_k=a, b_k=b)
            record("q(mu,sigma2)", r, r_0)

            tau2, m = m_k(v, h, y, w, r, N_k)
            v = v._replace(tau=tau2, m=m)
            v = v._replace(q_c=q_c(v, h, y, w, r, m))
            record("q(m,c)", r, r_0)

            a_s, b_s = peak_width(v, h, N_k, w, r, v.q_c, y)
            v = v._replace(a_s=a_s, b_s=b_s)
            record("q(s2)", r, r_0)

            v = v._replace(q_n=carbon_count(v, h, N_k, w, r))
            record("q(n)", r, r_0)

            a_p, b_p = heavy_isotope(h, N_k, w, r, v.q_n)
            v = v._replace(a_p=a_p, b_p=b_p)
            record("q(p)", r, r_0)

        values = np.array([e for _, e in trace])
        self.assertTrue(bool(np.all(np.isfinite(values))), trace)
        tol = 1e-4 * np.abs(values).max()
        for i in range(1, len(trace)):
            self.assertGreaterEqual(
                values[i] - values[i - 1],
                -tol,
                f"ELBO decreased after {trace[i][0]}: {values[i - 1]} -> {values[i]}",
            )
        self.assertGreater(values[-1], values[0])
