import json
import tempfile
from pathlib import Path
from unittest import TestCase, skipUnless

import jax.numpy as jnp
import numpy as np

from cavi.generative import default_hyperparameters, generate_spectrum

try:
    from cavi.main import init_variational_params, run
except ImportError:  # main.py needs pyopenms to load the spectrum
    run = None


@skipUnless(run is not None, "pyopenms not installed")
class TestFit(TestCase):
    """End to end: simulate a map, run CAVI, compare the posterior with the truth."""

    NUM_ITER = 15

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        mzml, truth = generate_spectrum(
            out_dir=Path(cls.tmp.name),
            seed=1,
            n_ions=200_000,
            n_scans=600,
            mz_bin=0.01,
            background_alpha=20.0,
        )
        cls.truth = json.loads(truth.read_text())
        cls.p = default_hyperparameters(background_alpha=20.0)
        cls.v, cls.elbos, cls.history = run(str(mzml), cls.p, num_j=3, num_iter=cls.NUM_ITER)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_shapes(self):
        K = self.p.m_hat.shape[0]
        self.assertEqual(self.elbos.shape, (self.NUM_ITER,))
        self.assertEqual(self.history.mu.shape, (self.NUM_ITER, K))
        self.assertEqual(self.history.q_c.shape, (self.NUM_ITER, K, 4))
        np.testing.assert_allclose(self.history.mu[-1], self.v.mu)

    def test_initialization_is_the_prior(self):
        v0 = init_variational_params(self.p)
        np.testing.assert_allclose(v0.q_c, self.p.rho)
        np.testing.assert_allclose(v0.tau, self.p.tau**2)
        np.testing.assert_allclose(v0.m, np.repeat(np.asarray(self.p.m_hat)[:, None], 4, axis=1))

    def test_elbo_increases_and_converges(self):
        e = np.asarray(self.elbos)
        self.assertTrue(np.all(np.isfinite(e)))
        self.assertGreater(e[-1], e[0])
        # float32 noise on a value of ~2e6 allows tiny dips once converged
        self.assertTrue(np.all(np.diff(e) > -1e-4 * np.abs(e).max()), e)
        self.assertLess(abs(e[-1] - e[-2]), 1e-4 * abs(e[-1]))

    def test_recovers_retention_time_mass_and_charge(self):
        for f in self.truth["features"]:
            k = f["feature"] - 1
            c = int(jnp.argmax(self.v.q_c[k]))
            self.assertAlmostEqual(float(self.v.mu[k]), f["rt_apex_min"], delta=0.1)
            self.assertEqual(c + 1, f["charge"])
            self.assertAlmostEqual(float(self.v.m[k, c]), f["mono_mz"], delta=0.01)

    def test_recovers_feature_sizes(self):
        # alpha_k = prior + expected number of ions of component k
        n_ions = np.array([self.truth["n_background_ions"]] + [f["n_ions"] for f in self.truth["features"]])
        expected = np.asarray(self.p.alpha) + n_ions
        np.testing.assert_allclose(np.asarray(self.v.alpha), expected, rtol=0.05)

    def test_recovers_peak_width(self):
        a_s, b_s = float(self.v.a_s), float(self.v.b_s)
        sigma = np.sqrt(b_s / (a_s - 1))  # mean of the inverse gamma
        self.assertAlmostEqual(sigma, self.truth["mz_sigma"], delta=0.01)
