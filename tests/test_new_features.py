import json
import tempfile
from pathlib import Path
from unittest import TestCase

import jax.numpy as jnp
import numpy as np

from cavi.generative import default_hyperparameters, generate_spectrum
from cavi.main import (
    append_candidates,
    fit,
    fit_with_new_features,
    hyperparameters_from_candidates,
    init_variational_params,
    load_spectrum,
    propose_candidates,
)

NUM_J = 3


def prior_from_truth(truth: list[dict], keep: list[int]):
    """A prior whose candidates sit at the true RT apex and mass of the features in `keep`."""
    p = default_hyperparameters(background_alpha=20.0)
    idx = jnp.array(keep)
    return p._replace(
        alpha=p.alpha[jnp.concatenate([jnp.array([0]), idx + 1])],
        m_0=jnp.array([truth[k]["rt_apex_min"] for k in keep], dtype=jnp.float32),
        m_hat=jnp.array([truth[k]["mono_mz"] for k in keep], dtype=jnp.float32),
        nu_0=p.nu_0[idx], a_0=p.a_0[idx], b_0=p.b_0[idx], tau=p.tau[idx], rho=p.rho[idx], n=p.n[idx],
    )


class TestCandidates(TestCase):

    def test_from_candidates(self):
        p = hyperparameters_from_candidates([10.0, 20.0], [650.5, 800.25], [2, 3])
        self.assertEqual(p.alpha.shape, (3,))
        self.assertEqual(p.rho.shape, (2, 4))
        np.testing.assert_allclose(p.rho.sum(axis=1), 1.0, atol=1e-6)
        self.assertEqual(int(jnp.argmax(p.rho[1])), 2)  # charge 3
        np.testing.assert_allclose(p.n.sum(axis=1), 1.0, atol=1e-5)
        np.testing.assert_allclose(p.tau, p.m_hat * 10e-6, rtol=1e-5)  # 10 ppm

    def test_tau_override_is_absolute(self):
        p = hyperparameters_from_candidates([10.0], [650.5], [2], tau=0.03)
        np.testing.assert_allclose(p.tau, 0.03)

    def test_append_keeps_old_and_adds_new(self):
        p = hyperparameters_from_candidates([10.0, 20.0], [650.5, 800.25], [2, 3])
        q = append_candidates(p, [30.0], [900.5], [4], tau=0.03)
        self.assertEqual(q.alpha.shape, (4,))
        self.assertEqual(q.m_hat.shape, (3,))
        self.assertEqual(q.rho.shape, (3, 4))
        self.assertEqual(q.n.shape, (3, p.n.shape[1]))
        np.testing.assert_array_equal(q.m_hat[:2], p.m_hat)
        np.testing.assert_array_equal(q.rho[:2], p.rho)
        np.testing.assert_array_equal(q.c_grid, p.c_grid)
        self.assertEqual(float(q.alpha[0]), float(p.alpha[0]))  # the background entry stays first
        self.assertEqual(int(jnp.argmax(q.rho[2])), 3)  # charge 4
        np.testing.assert_allclose(q.tau[2], 0.03)
        # global parameters untouched
        self.assertEqual(float(q.a_s), float(p.a_s))
        self.assertEqual(float(q.b_p), float(p.b_p))


class TestFindingNewFeatures(TestCase):
    """Simulated maps with known truth: features that are missing from the prior must be found."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.maps = {}
        for seed in (1, 2):
            mzml, truth = generate_spectrum(
                out_dir=Path(cls.tmp.name), name=f"s{seed}", seed=seed, n_ions=200_000,
                n_scans=600, mz_bin=0.01, background_alpha=20.0,
            )
            t, y, w = load_spectrum(str(mzml))
            cls.maps[seed] = (t, y, w, json.loads(truth.read_text())["features"])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_nothing_is_proposed_when_all_features_are_known(self):
        for seed, (t, y, w, truth) in self.maps.items():
            p = prior_from_truth(truth, [0, 1, 2, 3])
            v, _, _ = fit(init_variational_params(p), p, t, y, w, NUM_J, 15)
            new = propose_candidates(v, p, t, y, w, NUM_J, n_new=4)
            self.assertEqual(len(new["rt"]), 0, f"seed {seed}: {new}")

    def test_missing_features_are_found(self):
        t, y, w, truth = self.maps[1]
        p = prior_from_truth(truth, [0, 1])  # features 3 and 4 are not in the prior
        v, _, _ = fit(init_variational_params(p), p, t, y, w, NUM_J, 15)
        new = propose_candidates(v, p, t, y, w, NUM_J, n_new=4)

        self.assertEqual(len(new["rt"]), 2, new)
        for f in truth[2:]:
            d = np.abs(new["mz"] - f["mono_mz"])
            i = int(np.argmin(d))
            self.assertLess(d[i], 0.1)  # m/z within the bin
            self.assertLess(abs(new["rt"][i] - f["rt_apex_min"]), 2.0)  # RT within the elution width
            self.assertEqual(int(new["charge"][i]), f["charge"])

    def test_cap_on_the_number_of_candidates(self):
        t, y, w, truth = self.maps[1]
        p = prior_from_truth(truth, [0])
        v, _, _ = fit(init_variational_params(p), p, t, y, w, NUM_J, 15)
        new = propose_candidates(v, p, t, y, w, NUM_J, n_new=1)
        self.assertEqual(len(new["rt"]), 1)

    def test_refit_with_the_new_features_recovers_the_truth(self):
        t, y, w, truth = self.maps[1]
        p = prior_from_truth(truth, [0, 1])
        p2, added = fit_with_new_features(t, y, w, p, NUM_J, 15, rounds=2, n_new=4)
        self.assertEqual(p2.m_hat.shape[0], 4)
        self.assertEqual(len(added["rt"]), 2)

        v, elbos, _ = fit(init_variational_params(p2), p2, t, y, w, NUM_J, 25)
        n_k = np.asarray(v.alpha[1:] - p2.alpha[1:])
        for k, f in enumerate(truth):
            j = int(np.argmin(np.abs(np.asarray(v.mu) - f["rt_apex_min"])))  # new ones are appended
            self.assertAlmostEqual(float(v.mu[j]), f["rt_apex_min"], delta=0.1)
            self.assertAlmostEqual(float(v.m[j, int(jnp.argmax(v.q_c[j]))]), f["mono_mz"], delta=0.01)
            self.assertEqual(int(jnp.argmax(v.q_c[j])) + 1, f["charge"])
            np.testing.assert_allclose(n_k[j], f["n_ions"], rtol=0.03)

    def test_extending_the_candidates_never_lowers_the_elbo_much(self):
        # more components can only explain more of the data: the ELBO must not get worse
        t, y, w, truth = self.maps[1]
        p = prior_from_truth(truth, [0, 1])
        p2, _ = fit_with_new_features(t, y, w, p, NUM_J, 15, rounds=1, n_new=4)
        e_old = float(fit(init_variational_params(p), p, t, y, w, NUM_J, 25)[1][-1])
        e_new = float(fit(init_variational_params(p2), p2, t, y, w, NUM_J, 25)[1][-1])
        self.assertGreater(e_new, e_old)
