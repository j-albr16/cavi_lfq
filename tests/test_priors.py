from unittest import TestCase

import jax
import jax.numpy as jnp

from cavi.priors import HyperParameters, Theta, sample_normal_inverse_gamma, sample_theta, sample_z_and_j


class TestPriors(TestCase):

    def test_sample_normal_ig(self):
        key = jax.random.PRNGKey(0)
        K = 5
        m_0 = jnp.linspace(-1.0, 1.0, K)
        nu_0 = jnp.full((K,), 4.0)
        a_0 = jnp.full((K,), 3.0)
        b_0 = jnp.full((K,), 2.0)

        mu, sigma2 = sample_normal_inverse_gamma(key, m_0, nu_0, a_0, b_0)

        self.assertEqual(mu.shape, (K,))
        self.assertEqual(sigma2.shape, (K,))
        # sigma^2 ~ IG must be strictly positive
        self.assertTrue(bool(jnp.all(sigma2 > 0)))

        # different keys must give different draws (catches key-reuse bugs)
        mu2, sigma2_2 = sample_normal_inverse_gamma(jax.random.PRNGKey(1), m_0, nu_0, a_0, b_0)
        self.assertFalse(bool(jnp.allclose(mu, mu2)))
        self.assertFalse(bool(jnp.allclose(sigma2, sigma2_2)))

        # the K noise draws that make up mu must be independent of each other,
        # not one scalar broadcast across all K components
        self.assertGreater(jnp.std(mu - m_0), 0.0)

        # Monte Carlo check against the closed-form NIG marginals from lfq.typ:
        # sigma^2 ~ IG(a_0, b_0) has mean b_0 / (a_0 - 1); mu is marginally
        # unbiased for m_0 regardless of sigma^2.
        keys = jax.random.split(key, 20_000)
        mus, sigma2s = jax.vmap(
            sample_normal_inverse_gamma, in_axes=(0, None, None, None, None)
        )(keys, m_0, nu_0, a_0, b_0)

        theoretical_sigma2_mean = b_0 / (a_0 - 1)
        self.assertTrue(
            jnp.allclose(sigma2s.mean(axis=0), theoretical_sigma2_mean, rtol=0.1)
        )
        self.assertTrue(jnp.allclose(mus.mean(axis=0), m_0, atol=0.1))

    def test_sample_theta(self):
        K, G = 3, 4
        c_grid = jnp.array([10, 11, 12, 13], dtype=jnp.int32)

        params = HyperParameters(
            alpha=jnp.full((K + 1,), 1.5),
            m_0=jnp.array([100.0, 200.0, 300.0]),
            nu_0=jnp.full((K,), 4.0),
            a_0=jnp.full((K,), 3.0),
            b_0=jnp.full((K,), 2.0),
            m_hat=jnp.array([500.0, 501.0, 502.0]),
            tau=jnp.full((K,), 0.01),
            rho=jnp.full((K, 4), 0.25),
            n=jnp.full((K, G), 1.0 / G),
            c_grid=c_grid,
            a_p=jnp.array(2.0),
            b_p=jnp.array(200.0),
            a_s=jnp.array(3.0),
            b_s=jnp.array(1.0),
        )

        key = jax.random.PRNGKey(0)
        theta = sample_theta(key, params)

        self.assertEqual(theta.pi.shape, (K + 1,))
        self.assertEqual(theta.mu.shape, (K,))
        self.assertEqual(theta.sigma2.shape, (K,))
        self.assertEqual(theta.m.shape, (K,))
        self.assertEqual(theta.c.shape, (K,))
        self.assertEqual(theta.n.shape, (K,))
        self.assertEqual(theta.p.shape, ())
        self.assertEqual(theta.s2.shape, ())

        # pi lives on the simplex (Dirichlet)
        self.assertTrue(bool(jnp.all(theta.pi >= 0)))
        self.assertTrue(bool(jnp.allclose(theta.pi.sum(), 1.0, atol=1e-5)))

        self.assertTrue(bool(jnp.all(theta.sigma2 > 0)))
        self.assertTrue(bool(theta.s2 > 0))
        self.assertTrue(bool((theta.p >= 0) and (theta.p <= 1)))

        # charge must be an actual charge value in {1,2,3,4}, not a raw category index
        self.assertTrue(bool(jnp.all(jnp.isin(theta.c, jnp.array([1, 2, 3, 4])))))

        # carbon count must be a value from c_grid, not a raw grid index
        self.assertTrue(bool(jnp.all(jnp.isin(theta.n, c_grid))))

        # different keys must give different draws (catches the hardcoded-key bug)
        theta2 = sample_theta(jax.random.PRNGKey(1), params)
        self.assertFalse(bool(jnp.allclose(theta.mu, theta2.mu)))
        self.assertFalse(bool(jnp.allclose(theta.pi, theta2.pi)))

        # the different latents of one draw must not be coupled through a shared,
        # un-split key: pi and m come from unrelated distributions (Dirichlet vs.
        # Gaussian), so pi's entropy shouldn't determine m's offset from m_hat.
        self.assertFalse(bool(jnp.allclose(theta.pi[: params.m_hat.shape[0]], theta.m)))

        # Monte Carlo checks that the priors are wired to the right fields:
        # pi's mean should match Dir(alpha)'s mean, and charge should be uniform
        # since rho is uniform over the 4 charges.
        keys = jax.random.split(key, 5_000)
        thetas = jax.vmap(sample_theta, in_axes=(0, None))(keys, params)

        theoretical_pi_mean = params.alpha / params.alpha.sum()
        self.assertTrue(
            jnp.allclose(thetas.pi.mean(axis=0), theoretical_pi_mean, atol=0.05)
        )

        charge_freqs = jnp.array([(thetas.c == c).mean() for c in (1, 2, 3, 4)])
        self.assertTrue(jnp.allclose(charge_freqs, 0.25, atol=0.05))

    def test_sample_z_and_j(self):
        K = 3
        pi = jnp.array([0.4, 0.3, 0.2, 0.1])  # pi[0] = background weight
        n_k = jnp.array([10, 50, 100], dtype=jnp.int32)  # distinct per feature
        p = jnp.array(0.3)

        theta = Theta(
            pi=pi,
            mu=jnp.zeros(K),
            sigma2=jnp.ones(K),
            m=jnp.zeros(K),
            c=jnp.ones(K, dtype=jnp.int32),
            n=n_k,
            p=p,
            s2=jnp.array(1.0),
        )

        N = 20_000
        key = jax.random.PRNGKey(0)
        z, j = sample_z_and_j(key, theta, N)

        self.assertEqual(z.shape, (N,))
        self.assertEqual(j.shape, (N,))
        self.assertTrue(jnp.issubdtype(z.dtype, jnp.integer))
        self.assertTrue(jnp.issubdtype(j.dtype, jnp.integer))

        # z indexes background (0) plus the K features
        self.assertTrue(bool(jnp.all((z >= 0) & (z <= K))))

        # background pixels carry no isotope pattern
        self.assertTrue(bool(jnp.all(j[z == 0] == 0)))

        # j_n | z_n = k is Binomial(n_k, p): bounded by the feature's carbon count
        for k in range(1, K + 1):
            mask = z == k
            self.assertTrue(bool(jnp.all(j[mask] >= 0)))
            self.assertTrue(bool(jnp.all(j[mask] <= n_k[k - 1])))

        # empirical feature-assignment frequencies must match pi (Monte Carlo)
        z_freqs = jnp.array([(z == k).mean() for k in range(K + 1)])
        self.assertTrue(jnp.allclose(z_freqs, pi, atol=0.02))

        # conditional mean of j given z = k must match the Binomial(n_k, p) mean,
        # i.e. j must actually be drawn using the *assigned* feature's n_k,
        # not an unconditional / mismatched one
        for k in range(1, K + 1):
            mask = z == k
            empirical_mean = j[mask].mean()
            theoretical_mean = n_k[k - 1] * p
            self.assertTrue(jnp.allclose(empirical_mean, theoretical_mean, atol=1.0))

        # different keys must give different draws
        z2, j2 = sample_z_and_j(jax.random.PRNGKey(1), theta, N)
        self.assertFalse(bool(jnp.allclose(z, z2)))
        self.assertFalse(bool(jnp.allclose(j, j2)))
