from pathlib import Path
from unittest import TestCase, skipUnless

import jax.numpy as jnp
import numpy as np

from cavi.main import fit, init_variational_params, load_spectrum, random_hyperparameters

DATA = Path(__file__).resolve().parent.parent / "data"
REAL_MZML = DATA / "FeatureFinderCentroided_1_input.mzML"
REAL_FEATURES = DATA / "FeatureFinderCentroided_1_1_output.featureXML"


def toy_pixels(n=500, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.uniform(10.0, 20.0, n).astype(np.float32)
    y = rng.uniform(500.0, 700.0, n).astype(np.float32)
    w = rng.uniform(1.0, 5.0, n).astype(np.float32)
    return jnp.array(t), jnp.array(y), jnp.array(w)


class TestRandomHyperparameters(TestCase):

    def test_shapes_and_normalisation(self):
        t, y, w = toy_pixels()
        p = random_hyperparameters(t, y, w, num_features=30)
        self.assertEqual(p.alpha.shape, (31,))
        self.assertEqual(p.m_hat.shape, (30,))
        self.assertEqual(p.rho.shape, (30, 4))
        np.testing.assert_allclose(p.rho.sum(axis=1), 1.0, atol=1e-6)
        np.testing.assert_allclose(p.n.sum(axis=1), 1.0, atol=1e-5)
        np.testing.assert_allclose(p.rho[0], p.rho[7])  # the same weak charge prior for all
        self.assertTrue(bool(jnp.all(p.alpha == 1.0)))

    def test_pixel_init_sits_on_pixels(self):
        t, y, w = toy_pixels()
        p = random_hyperparameters(t, y, w, num_features=50, init="pixels")
        pixels = set(zip(np.asarray(t).tolist(), np.asarray(y).tolist()))
        for rt, mz in zip(np.asarray(p.m_0).tolist(), np.asarray(p.m_hat).tolist()):
            self.assertIn((rt, mz), pixels)

    def test_pixel_init_follows_the_intensity(self):
        t = jnp.array([10.0, 11.0], dtype=jnp.float32)
        y = jnp.array([500.0, 600.0], dtype=jnp.float32)
        w = jnp.array([1.0, 99.0], dtype=jnp.float32)  # the second pixel is 99 times brighter
        p = random_hyperparameters(t, y, w, num_features=2000, init="pixels")
        share = float(jnp.mean(p.m_hat > 550.0))
        self.assertGreater(share, 0.97)

    def test_uniform_init_covers_the_window(self):
        t, y, w = toy_pixels()
        p = random_hyperparameters(t, y, w, num_features=2000, init="uniform")
        self.assertTrue(bool(jnp.all((p.m_0 >= float(t.min())) & (p.m_0 <= float(t.max())))))
        self.assertTrue(bool(jnp.all((p.m_hat >= float(y.min())) & (p.m_hat <= float(y.max())))))
        # spread over the whole window, not concentrated
        self.assertAlmostEqual(float(jnp.mean(p.m_hat)), float(y.min() + y.max()) / 2, delta=8.0)

    def test_mz_range_is_respected(self):
        t, y, w = toy_pixels()
        for init in ("pixels", "uniform"):
            p = random_hyperparameters(t, y, w, num_features=300, init=init, mz_range=(550.0, 600.0))
            self.assertTrue(bool(jnp.all((p.m_hat >= 550.0) & (p.m_hat <= 600.0))), init)

    def test_charges_follow_the_distribution(self):
        t, y, w = toy_pixels()
        p = random_hyperparameters(t, y, w, num_features=4000, charge_probs=(0.0, 1.0, 0.0, 0.0))
        # the carbon-count prior is centred on the averagine value for charge 2 only
        carbons = np.asarray((p.n * p.c_grid[None, :]).sum(axis=1))
        expected = 0.0444 * (np.asarray(p.m_hat) - 1.007276) * 2
        np.testing.assert_allclose(carbons, expected, atol=2.0)

    def test_seed_and_errors(self):
        t, y, w = toy_pixels()
        a = random_hyperparameters(t, y, w, num_features=20, seed=3)
        b = random_hyperparameters(t, y, w, num_features=20, seed=3)
        c = random_hyperparameters(t, y, w, num_features=20, seed=4)
        np.testing.assert_array_equal(a.m_hat, b.m_hat)
        self.assertFalse(bool(jnp.all(a.m_hat == c.m_hat)))
        with self.assertRaises(ValueError):
            random_hyperparameters(t, y, w, init="grid")
        with self.assertRaises(ValueError):
            random_hyperparameters(t, y, w, mz_range=(5000.0, 6000.0))


@skipUnless(REAL_MZML.exists() and REAL_FEATURES.exists(), "OpenMS example data not in data/")
class TestAdaptsOnRealData(TestCase):
    """Random candidates must move onto the peaks; the OpenMS features are only the answer key."""

    @classmethod
    def setUpClass(cls):
        import pyopenms as oms

        cls.t, cls.y, cls.w, cls.scale = load_spectrum(str(REAL_MZML), return_scale=True)
        fm = oms.FeatureMap()
        oms.FeatureXMLFile().load(str(REAL_FEATURES), fm)
        cls.features = [
            {"rt_apex_min": f.getRT() / 60, "mono_mz": f.getMZ(), "charge": f.getCharge(),
             "intensity": f.getIntensity()}
            for f in fm
        ]
        cls.ref = [(f["rt_apex_min"], f["mono_mz"]) for f in cls.features]

    def fit(self, init, seed, K=30):
        p = random_hyperparameters(self.t, self.y, self.w, num_features=K, seed=seed, init=init)
        v, elbos, _ = fit(init_variational_params(p), p, self.t, self.y, self.w, 5, 40)
        return p, v, np.asarray(elbos)

    def recovered(self, p, v):
        n_k = np.asarray(v.alpha[1:] - p.alpha[1:])
        alive = np.flatnonzero(n_k > 30)
        found = 0
        for rt, mz in self.ref:
            for k in alive:
                c = int(jnp.argmax(v.q_c[k]))
                if abs(float(v.m[k, c]) - mz) < 0.03 and abs(float(v.mu[k]) - rt) < 0.5:
                    found += 1
                    break
        return found, len(alive)

    def test_pixel_init_finds_most_reference_features(self):
        # with only 30 candidates how many of the 8 features are found depends on the seed
        # (3 to 7 over seeds 0..5); these two seeds are fixed, so the result is deterministic
        for seed, at_least in ((1, 6), (2, 4)):
            p, v, elbos = self.fit("pixels", seed)
            found, _ = self.recovered(p, v)
            self.assertGreaterEqual(found, at_least, f"seed {seed}")
            self.assertTrue(np.all(np.diff(elbos) > -1e-4 * np.abs(elbos).max()))

    def test_more_candidates_find_more_features(self):
        found = {}
        for K in (10, 100):
            p, v, _ = self.fit("pixels", 0, K=K)
            found[K] = self.recovered(p, v)[0]
        self.assertGreater(found[100], found[10])
        self.assertEqual(found[100], len(self.ref))

    def test_peak_width_is_learned(self):
        # the prior guess is 0.05 Da, the peaks of this map are about 0.008 Da wide
        _, v, _ = self.fit("pixels", 1)
        s = float(jnp.sqrt(v.b_s / (v.a_s - 1)))
        self.assertAlmostEqual(s, 0.008, delta=0.004)

    def test_intensities_agree_with_the_reference_up_to_a_factor(self):
        from cavi.evaluate import compare_intensities, feature_intensities

        p, v, _ = self.fit("pixels", 1)
        res = feature_intensities(v, p, self.features, 5, self.scale)
        ref = np.array([f["intensity"] for f in self.features])
        c = compare_intensities(res["intensity"], ref)
        self.assertGreaterEqual(c["n_compared"], 6)
        self.assertGreater(c["pearson_log"], 0.75)  # 0.84 to 0.97 over seeds 0..5
        # the reference is a model estimate and exceeds the summed pixel intensities of the whole
        # file, so the ratio is below 1 (0.3 to 0.5 here) but must not be tiny
        self.assertGreater(c["median_ratio"], 0.2)
        self.assertLess(c["median_ratio"], 1.0)
        # everything that was assigned stays inside the file
        total = res["intensity"].sum() + res["other"] + res["background"]
        np.testing.assert_allclose(total, float(self.w.sum()) * self.scale, rtol=0.02)

    def test_uniform_placement_mostly_fails(self):
        # documents the limitation: candidates that overlap no pixel never get responsibility
        p, v, _ = self.fit("uniform", 0, K=60)
        found, alive = self.recovered(p, v)
        self.assertLess(found, 3)
