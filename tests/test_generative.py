from unittest import TestCase

import jax
import jax.numpy as jnp

from cavi.priors import Theta
from cavi.generative import generative_model


def make_theta():
    return Theta(
        pi=jnp.array([0.4, 0.3, 0.2, 0.1]),
        mu=jnp.array([10.0, 20.0, 30.0]),
        sigma2=jnp.array([1.0, 4.0, 9.0]),
        m=jnp.array([500.0, 600.0, 700.0]),
        c=jnp.array([1, 2, 3], dtype=jnp.int32),
        n=jnp.array([10, 20, 30], dtype=jnp.int32),
        p=jnp.array(0.3),
        s2=jnp.array(0.01),
    )


class TestGenerativeModel(TestCase):

    def test_background_is_uniform_over_the_window(self):
        theta = make_theta()
        N = 20_000
        z = jnp.zeros((N,), dtype=jnp.int32)
        j = jnp.zeros((N,), dtype=jnp.int32)
        t_range = jnp.array([0.0, 100.0])
        y_range = jnp.array([400.0, 900.0])

        t, y = generative_model(
            jax.random.PRNGKey(0), theta, z, j, t_range, y_range, delta=1.00336
        )

        self.assertTrue(bool(jnp.all((t >= t_range[0]) & (t <= t_range[1]))))
        self.assertTrue(bool(jnp.all((y >= y_range[0]) & (y <= y_range[1]))))

        # Uniform(a, b) has mean (a + b) / 2; Monte Carlo check
        self.assertTrue(jnp.allclose(t.mean(), t_range.mean(), atol=1.0))
        self.assertTrue(jnp.allclose(y.mean(), y_range.mean(), atol=5.0))

    def test_feature_pixels_follow_the_feature_gaussians(self):
        theta = make_theta()
        N = 20_000
        k = 2  # 1-indexed feature id -> theta.mu[k - 1], theta.c[k - 1], ...
        isotope = 3
        delta = 1.00336

        z = jnp.full((N,), k, dtype=jnp.int32)
        j = jnp.full((N,), isotope, dtype=jnp.int32)
        t_range = jnp.array([0.0, 100.0])
        y_range = jnp.array([400.0, 900.0])

        t, y = generative_model(
            jax.random.PRNGKey(0), theta, z, j, t_range, y_range, delta
        )

        # t_n | z_n = k ~ N(mu_k, sigma_k^2)
        expected_t_mean = theta.mu[k - 1]
        expected_t_std = jnp.sqrt(theta.sigma2[k - 1])
        self.assertTrue(jnp.allclose(t.mean(), expected_t_mean, atol=0.1))
        self.assertTrue(jnp.allclose(t.std(), expected_t_std, rtol=0.05))

        # y_n | z_n = k, j_n ~ N(m_k + j_n * delta / c_k, s^2)
        expected_y_mean = theta.m[k - 1] + isotope * delta / theta.c[k - 1]
        expected_y_std = jnp.sqrt(theta.s2)
        self.assertTrue(jnp.allclose(y.mean(), expected_y_mean, atol=0.02))
        self.assertTrue(jnp.allclose(y.std(), expected_y_std, rtol=0.05))

    def test_background_and_features_are_not_mixed_up(self):
        theta = make_theta()
        delta = 1.00336
        z = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
        j = jnp.array([0, 1, 2, 0], dtype=jnp.int32)
        t_range = jnp.array([0.0, 100.0])
        y_range = jnp.array([400.0, 900.0])

        t, y = generative_model(
            jax.random.PRNGKey(0), theta, z, j, t_range, y_range, delta
        )

        self.assertEqual(t.shape, (4,))
        self.assertEqual(y.shape, (4,))

        # background pixel lands in the window, not near any feature's mu/m
        self.assertTrue(bool((t[0] >= t_range[0]) and (t[0] <= t_range[1])))
        self.assertTrue(bool((y[0] >= y_range[0]) and (y[0] <= y_range[1])))

        # feature pixels land close (a few sigma) to their own feature's mu,
        # catching any off-by-one in the background/feature index mapping
        for i, k in enumerate([1, 2, 3]):
            tol = 5.0 * float(jnp.sqrt(theta.sigma2[k - 1]))
            self.assertTrue(jnp.allclose(t[i + 1], theta.mu[k - 1], atol=tol))

    def test_different_keys_give_different_draws(self):
        theta = make_theta()
        delta = 1.00336
        z = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
        j = jnp.array([0, 1, 2, 0], dtype=jnp.int32)
        t_range = jnp.array([0.0, 100.0])
        y_range = jnp.array([400.0, 900.0])

        t1, y1 = generative_model(jax.random.PRNGKey(0), theta, z, j, t_range, y_range, delta)
        t2, y2 = generative_model(jax.random.PRNGKey(1), theta, z, j, t_range, y_range, delta)

        self.assertFalse(bool(jnp.allclose(t1, t2)))
        self.assertFalse(bool(jnp.allclose(y1, y2)))
