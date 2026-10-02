import json
import tempfile
from pathlib import Path
from unittest import TestCase, skipUnless

import jax.numpy as jnp
import numpy as np
import pyopenms as oms

from cavi.generative import generate_spectrum
from cavi.main import (get_gamma, hyperparameters_from_features,
                       init_variational_params, load_spectrum, run)
from cavi.priors import HyperParameters, VariationalParams

INTENSITY_PER_ION = 100.0
N_ION = 50_000
N_SCANS = 100
T_RANGE = (0.0, 60.0)
Y_RANGE = (400.0, 1200.0)


class TestMain(TestCase):

    @classmethod
    def setUpClass(cls):
        # a small synthetic map in a temp dir, so the tests do not depend on data/ or the cwd
        cls.tmp = tempfile.TemporaryDirectory()
        cls.spectrum_path, truth_path = generate_spectrum(
            out_dir=Path(cls.tmp.name),
            seed=2,
            n_ions=N_ION,
            t_range=T_RANGE,
            y_range=Y_RANGE,
            n_scans=N_SCANS,
            mz_bin=0.01,
            intensity_per_ion=INTENSITY_PER_ION,
            background_alpha=20.0,
        )
        cls.spectrum_path = str(cls.spectrum_path)
        cls.truth = json.loads(truth_path.read_text())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        K, G = 3, 4
        c_grid = jnp.array([10, 11, 12, 13], dtype=jnp.int32)

        self.p = HyperParameters(
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
        self.K, self.G = K, G

    # --- loading -----------------------------------------------------------------

    def test_get_gamma_is_inverse_of_smallest_intensity(self):
        np.testing.assert_allclose(get_gamma(jnp.array([300.0, 100.0, 200.0])), 0.01)

    def test_load_spectrum_shapes_and_units(self):
        t, y, w = load_spectrum(self.spectrum_path)
        self.assertEqual(t.shape, y.shape)
        self.assertEqual(t.shape, w.shape)
        # retention time is converted from OpenMS seconds to minutes
        self.assertGreaterEqual(float(t.min()), T_RANGE[0])
        self.assertLessEqual(float(t.max()), T_RANGE[1])
        self.assertGreaterEqual(float(y.min()), Y_RANGE[0])
        self.assertLessEqual(float(y.max()), Y_RANGE[1])
        # scans are stored in order
        self.assertTrue(bool(jnp.all(jnp.diff(t) >= 0)))
        # with 50k ions spread over the window, every scan holds at least one pixel
        scan_times = np.unique(np.asarray(t))
        self.assertEqual(len(scan_times), N_SCANS)
        np.testing.assert_allclose(
            scan_times, (np.arange(N_SCANS) + 0.5) * (T_RANGE[1] - T_RANGE[0]) / N_SCANS, atol=1e-3
        )

    def test_load_spectrum_weights_are_ion_counts(self):
        _, _, w = load_spectrum(self.spectrum_path)
        w = np.asarray(w)
        # intensity = ions * INTENSITY_PER_ION and gamma = 1 / min(intensity), so w = ion counts
        self.assertAlmostEqual(w.min(), 1.0, places=5)
        np.testing.assert_allclose(w, np.round(w), atol=1e-3)
        # every simulated ion inside the window lands in exactly one pixel
        np.testing.assert_allclose(w.sum(), self.truth["n_ions"], rtol=1e-4)

    # --- initialization ----------------------------------------------------------

    def test_init_variational_params(self):
        v = init_variational_params(self.p)
        self.assertIsInstance(v, VariationalParams)
        self.assertEqual(v.q_c.shape, (self.K, 4))
        self.assertEqual(v.m.shape, (self.K, 4))
        self.assertEqual(v.q_n.shape, (self.K, self.G))
        np.testing.assert_allclose(v.tau, self.p.tau**2)
        np.testing.assert_allclose(v.m, jnp.repeat(self.p.m_hat[:, None], 4, axis=1))

    # --- run ---------------------------------------------------------------------

    def test_run(self):
        num_iter = 8
        v, elbos, history = run(self.spectrum_path, self.p, num_j=3, num_iter=num_iter)

        self.assertIsInstance(v, VariationalParams)
        self.assertEqual(elbos.shape, (num_iter,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(elbos))))
        for field in VariationalParams._fields:
            self.assertTrue(bool(jnp.all(jnp.isfinite(getattr(v, field)))), field)
            self.assertEqual(
                getattr(history, field).shape,
                (num_iter, *getattr(v, field).shape),
                field,
            )
        # the last entry of the history is the returned posterior
        np.testing.assert_allclose(history.mu[-1], v.mu)
        np.testing.assert_allclose(history.alpha[-1], v.alpha)

    def test_run_posterior_is_normalized_and_valid(self):
        v, _, _ = run(self.spectrum_path, self.p, num_j=3, num_iter=5)
        np.testing.assert_allclose(v.q_c.sum(axis=1), 1.0, atol=1e-5)
        np.testing.assert_allclose(v.q_n.sum(axis=1), 1.0, atol=1e-5)
        for positive in (v.nu, v.a_k, v.b_k, v.tau, v.a_s, v.b_s, v.a_p, v.b_p, v.alpha):
            self.assertTrue(bool(jnp.all(positive > 0)))

    def test_run_elbo_does_not_decrease(self):
        _, elbos, _ = run(self.spectrum_path, self.p, num_j=3, num_iter=10)
        e = np.asarray(elbos)
        self.assertTrue(np.all(np.diff(e) > -1e-4 * np.abs(e).max()), e)

    def test_run_is_deterministic(self):
        _, e1, _ = run(self.spectrum_path, self.p, num_j=3, num_iter=3)
        _, e2, _ = run(self.spectrum_path, self.p, num_j=3, num_iter=3)
        np.testing.assert_array_equal(e1, e2)

    def test_background_dominates_for_a_prior_that_misses_the_features(self):
        # this prior puts its features at m/z 500-502, where the simulation has none:
        # nearly all ion mass must end up in the background component alpha_0
        v, _, _ = run(self.spectrum_path, self.p, num_j=3, num_iter=10)
        total = float(v.alpha.sum() - self.p.alpha.sum())
        self.assertGreater(float(v.alpha[0] - self.p.alpha[0]) / total, 0.95)


def _load_features(path):
    fm = oms.FeatureMap()
    oms.FeatureXMLFile().load(str(path), fm)
    return list(fm)


REAL_MZML = Path(__file__).resolve().parent.parent / "data" / "FeatureFinderCentroided_1_input.mzML"
REAL_FEATURES = Path(__file__).resolve().parent.parent / "data" / "FeatureFinderCentroided_1_1_output.featureXML"


@skipUnless(REAL_MZML.exists() and REAL_FEATURES.exists(), "OpenMS example data not in data/")
class TestRealData(TestCase):
    """OpenMS example map (centroided peptides, 8 candidate features of charge 2)."""

    @classmethod
    def setUpClass(cls):
        cls.p = hyperparameters_from_features(str(REAL_FEATURES))
        cls.ref = [
            (f.getRT() / 60, f.getMZ(), f.getCharge()) for f in _load_features(REAL_FEATURES)
        ]
        cls.v, cls.elbos, _ = run(str(REAL_MZML), cls.p, num_j=5, num_iter=30)

    def test_priors_from_features(self):
        K = len(self.ref)
        p = self.p
        self.assertEqual(K, 8)
        self.assertEqual(p.alpha.shape, (K + 1,))
        np.testing.assert_allclose(p.m_hat, [m for _, m, _ in self.ref], rtol=1e-6)
        np.testing.assert_allclose(p.m_0, [t for t, _, _ in self.ref], rtol=1e-5)
        np.testing.assert_allclose(p.rho.sum(axis=1), 1.0, atol=1e-6)
        np.testing.assert_allclose(p.n.sum(axis=1), 1.0, atol=1e-5)
        self.assertTrue(bool(jnp.all(jnp.argmax(p.rho, axis=1) == 1)))  # charge 2
        self.assertEqual(p.c_grid.dtype, jnp.int32)
        # averagine: a ~1300 Da peptide has ~57 carbons
        mean_n = (p.n * p.c_grid[None, :]).sum(axis=1)
        self.assertTrue(bool(jnp.all((mean_n > 40) & (mean_n < 75))), mean_n)

    def test_elbo_does_not_decrease(self):
        e = np.asarray(self.elbos)
        self.assertTrue(np.all(np.isfinite(e)))
        self.assertTrue(np.all(np.diff(e) > -1e-4 * np.abs(e).max()), e)

    def test_agrees_with_featurefinder(self):
        # the strong features agree to ~1 ppm; the weak ones (low S/N) drift, so the bound loosens
        order = np.argsort([-f.getIntensity() for f in _load_features(REAL_FEATURES)])
        for rank, k in enumerate(order):
            rt, mz, charge = self.ref[k]
            c = int(jnp.argmax(self.v.q_c[k]))
            self.assertEqual(c + 1, charge)
            strong = rank < 3
            self.assertAlmostEqual(float(self.v.mu[k]), rt, delta=0.1 if strong else 0.5)  # minutes
            self.assertAlmostEqual(float(self.v.m[k, c]), mz, delta=0.002 if strong else 0.01)  # Da

    def test_isotope_peaks_are_explained(self):
        # p must stay at the order of the natural 13C abundance (0.0107): a collapse to ~0 means
        # the isotope peaks j >= 1 were dropped, which happens with a vague charge prior
        p_mean = float(self.v.a_p / (self.v.a_p + self.v.b_p))
        self.assertGreater(p_mean, 0.005)
        self.assertLess(p_mean, 0.03)
