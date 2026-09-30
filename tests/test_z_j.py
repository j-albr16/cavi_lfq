from unittest import TestCase

import jax.numpy as jnp

from cavi.priors import HyperParameters, VariationalParams
from cavi.z_j import get_r


class TestZJ(TestCase):

    def setUp(self):
        K, G, N = 3, 4, 100
        c_grid = jnp.array([10, 11, 12, 13], dtype=jnp.int32)
        p = HyperParameters(
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
            b_k=p.a_0,
            tau=p.tau,
            m=jnp.broadcast_to(p.m_hat[:, None], (K, 4)),
        )
        self.K, self.G, self.N = K, G, N
        self.y, self.t = jnp.full((N,), 500.5), jnp.full((N,), 100.)
        self.T = 100
        self.Y = 300
        self.J = 3
        self.p = p

    def test_get_r(self):
        r = get_r(
            self.v,
            self.p,
            self.y,
            self.t,
            self.J,
            self.T,
            self.Y,
        )

        r, r_0 = r
        self.assertEqual(r.shape, (self.N, self.K, self.J))
        self.assertEqual(r_0.shape, (self.N,))
        self.assertFalse(bool(jnp.any(jnp.isnan(r))))
        self.assertFalse(bool(jnp.any(jnp.isnan(r_0))))
        total = r.sum(axis=(1, 2)) + r_0
        self.assertTrue(bool(jnp.allclose(total, 1.0, atol=1e-5)))




