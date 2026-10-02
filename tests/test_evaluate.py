from unittest import TestCase

import jax.numpy as jnp
import numpy as np

from cavi.evaluate import assign_components, compare_intensities, feature_intensities
from cavi.generative import DELTA_13C
from cavi.priors import HyperParameters, VariationalParams

NUM_J = 4


def posterior(components: list[tuple[float, float, int, float]], background: float = 100.0):
    """Posterior with one component per (rt, mass, charge index, expected ions); alpha prior = 1."""
    K, G = len(components), 3
    rt, mass, ci, ions = (np.array(x) for x in zip(*components))
    q_c = np.zeros((K, 4))
    q_c[np.arange(K), ci.astype(int)] = 1.0
    m = np.repeat(mass[:, None], 4, axis=1)
    prior = jnp.ones(K + 1)
    p = HyperParameters(
        alpha=prior, m_0=jnp.zeros(K), nu_0=jnp.ones(K), a_0=jnp.ones(K), b_0=jnp.ones(K),
        m_hat=jnp.zeros(K), tau=jnp.ones(K), rho=jnp.full((K, 4), 0.25), n=jnp.full((K, G), 1 / G),
        c_grid=jnp.arange(G), a_p=jnp.array(1.0), b_p=jnp.array(1.0), a_s=jnp.array(1.0), b_s=jnp.array(1.0),
    )
    v = VariationalParams(
        alpha=jnp.concatenate([jnp.array([1.0 + background]), 1.0 + jnp.asarray(ions)]),
        a_p=p.a_p, b_p=p.b_p, a_s=p.a_s, b_s=p.b_s, q_n=p.n, q_c=jnp.asarray(q_c),
        mu=jnp.asarray(rt), nu=jnp.ones(K), a_k=jnp.ones(K), b_k=jnp.ones(K),
        tau=jnp.ones(K), m=jnp.asarray(m),
    )
    return v, p


FEATURES = [
    {"rt_apex_min": 10.0, "mono_mz": 600.0, "charge": 2, "intensity": 1000.0},
    {"rt_apex_min": 20.0, "mono_mz": 800.0, "charge": 3, "intensity": 500.0},
]


class TestAssignComponents(TestCase):

    def test_monoisotopic_tooth_and_unrelated_component(self):
        v, p = posterior([
            (10.0, 600.0, 1, 100.0),  # on feature 0
            (10.1, 600.0 + DELTA_13C / 2, 1, 40.0),  # tooth j = 1 of feature 0 (charge 2)
            (20.0, 800.0 + 2 * DELTA_13C / 3, 2, 30.0),  # tooth j = 2 of feature 1 (charge 3)
            (15.0, 700.0, 1, 50.0),  # nowhere near a feature
        ])
        np.testing.assert_array_equal(assign_components(v, p, FEATURES, NUM_J), [0, 0, 1, -1])

    def test_tolerances(self):
        v, p = posterior([
            (10.0, 600.04, 1, 10.0),  # 0.04 Da off, beyond 0.03
            (10.0, 600.02, 1, 10.0),  # within
            (11.5, 600.0, 1, 10.0),  # RT 1.5 min off, beyond 1.0
            (10.9, 600.0, 1, 10.0),  # within
        ])
        np.testing.assert_array_equal(assign_components(v, p, FEATURES, NUM_J), [-1, 0, -1, 0])

    def test_teeth_beyond_num_j_and_wrong_charge_spacing(self):
        v, p = posterior([
            (10.0, 600.0 + 4 * DELTA_13C / 2, 1, 10.0),  # j = 4 is outside num_j = 4 (j < 4)
            (10.0, 600.0 + DELTA_13C / 3, 1, 10.0),  # spacing of charge 3, feature 0 has charge 2
        ])
        np.testing.assert_array_equal(assign_components(v, p, FEATURES, NUM_J), [-1, -1])

    def test_tiny_components_are_ignored(self):
        v, p = posterior([(10.0, 600.0, 1, 0.5), (10.0, 600.0, 1, 10.0)])
        np.testing.assert_array_equal(assign_components(v, p, FEATURES, NUM_J, min_ions=1.0), [-1, 0])

    def test_closest_feature_wins(self):
        features = [
            {"rt_apex_min": 10.0, "mono_mz": 600.0, "charge": 2},
            {"rt_apex_min": 10.0, "mono_mz": 600.0 - DELTA_13C / 2 + 0.01, "charge": 2},
        ]
        # sits at the monoisotopic mass of feature 0 and 0.01 Da from the j = 1 tooth of feature 1
        v, p = posterior([(10.0, 600.0 + 0.0005, 1, 10.0)])
        self.assertEqual(int(assign_components(v, p, features, NUM_J)[0]), 0)


class TestFeatureIntensities(TestCase):

    def test_sums_per_feature_in_file_units(self):
        v, p = posterior([
            (10.0, 600.0, 1, 100.0),
            (10.1, 600.0 + DELTA_13C / 2, 1, 40.0),
            (20.0, 800.0, 2, 30.0),
            (15.0, 700.0, 1, 50.0),
        ], background=1000.0)
        r = feature_intensities(v, p, FEATURES, NUM_J, scale=11.64)
        np.testing.assert_allclose(r["intensity"], [140.0 * 11.64, 30.0 * 11.64])
        np.testing.assert_array_equal(r["n_components"], [2, 1])
        np.testing.assert_allclose(r["other"], 50.0 * 11.64)
        np.testing.assert_allclose(r["background"], 1000.0 * 11.64)

    def test_feature_without_component_gets_zero(self):
        v, p = posterior([(10.0, 600.0, 1, 100.0)])
        r = feature_intensities(v, p, FEATURES, NUM_J)
        np.testing.assert_allclose(r["intensity"], [100.0, 0.0])
        np.testing.assert_array_equal(r["n_components"], [1, 0])

    def test_everything_adds_up(self):
        v, p = posterior([(10.0, 600.0, 1, 100.0), (20.0, 800.0, 2, 30.0), (15.0, 700.0, 1, 50.0)], background=70.0)
        r = feature_intensities(v, p, FEATURES, NUM_J, scale=2.0)
        total = r["intensity"].sum() + r["other"] + r["background"]
        np.testing.assert_allclose(total, 2.0 * (100 + 30 + 50 + 70))


class TestCompareIntensities(TestCase):

    def test_perfect_proportionality(self):
        ref = np.array([1000.0, 500.0, 100.0, 10.0])
        c = compare_intensities(0.4 * ref, ref)
        np.testing.assert_allclose(c["ratio"], 0.4)
        self.assertAlmostEqual(c["median_ratio"], 0.4)
        self.assertAlmostEqual(c["pearson_log"], 1.0)
        self.assertAlmostEqual(c["spearman"], 1.0)
        self.assertAlmostEqual(c["ratio_spread"], 1.0)

    def test_zero_intensities_are_not_compared(self):
        ref = np.array([1000.0, 500.0, 100.0, 10.0])
        c = compare_intensities(np.array([400.0, 0.0, 40.0, 4.0]), ref)
        self.assertEqual(c["n_compared"], 3)
        self.assertTrue(np.isnan(c["ratio"][1]))
        self.assertAlmostEqual(c["median_ratio"], 0.4)

    def test_reversed_order_has_negative_rank_correlation(self):
        ref = np.array([1000.0, 500.0, 100.0, 10.0])
        c = compare_intensities(ref[::-1].copy(), ref)
        self.assertAlmostEqual(c["spearman"], -1.0)
        self.assertLess(c["pearson_log"], 0.0)

    def test_noisy_ratio_spread(self):
        ref = np.array([1000.0, 500.0, 100.0, 10.0])
        c = compare_intensities(ref * np.array([0.2, 0.8, 0.2, 0.8]), ref)
        self.assertGreater(c["ratio_spread"], 1.5)

    def test_too_few_pairs_give_no_correlation(self):
        c = compare_intensities(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        self.assertNotIn("pearson_log", c)
