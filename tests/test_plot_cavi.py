import importlib.util
import tempfile
from pathlib import Path
from unittest import TestCase

import numpy as np

from cavi.generative import default_hyperparameters
from cavi.main import init_variational_params

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("plot_cavi", ROOT / "plots" / "plot_cavi.py")
plot_cavi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plot_cavi)


class TestGridEdges(TestCase):

    def setUp(self):
        self.values = 0.05 + 0.1 * np.arange(100)  # cell centres, step 0.1

    def test_edges_sit_on_the_grid_and_cover_the_window(self):
        edges = plot_cavi.grid_edges(self.values, 2.03, 5.47, group=1)
        self.assertTrue(np.all(np.diff(edges) > 0))
        np.testing.assert_allclose(np.diff(edges), 0.1)
        # grid cell boundaries are at multiples of 0.1
        np.testing.assert_allclose(edges / 0.1, np.round(edges / 0.1), atol=1e-6)
        self.assertLessEqual(edges[0], 2.03 + 1e-9)
        self.assertGreaterEqual(edges[-1], 5.47 - 1e-9)

    def test_group_merges_cells(self):
        edges = plot_cavi.grid_edges(self.values, 0.0, 10.0, group=4)
        np.testing.assert_allclose(np.diff(edges)[:-1], 0.4)

    def test_window_outside_the_data_is_never_empty(self):
        # this used to give an empty edge array and crash np.histogram2d
        for lo, hi in [(-50.0, -40.0), (500.0, 600.0), (3.0, 3.0)]:
            edges = plot_cavi.grid_edges(self.values, lo, hi, group=1)
            self.assertGreaterEqual(len(edges), 2)
            self.assertTrue(np.all(np.diff(edges) > 0))


class TestExpectedMap(TestCase):

    def setUp(self):
        self.p = default_hyperparameters(background_alpha=20.0)
        self.v = init_variational_params(self.p)
        self.v = type(self.v)(*(np.asarray(x) for x in self.v))
        self.c_grid = np.asarray(self.p.c_grid)
        self.t_size, self.y_size = 60.0, 800.0
        self.win = plot_cavi.Window(np.linspace(0.0, 60.0, 61), np.linspace(400.0, 1200.0, 321))
        self.n_total = 1000.0

    def expected(self, win=None, S=16, num_j=12, seed=0):
        return plot_cavi.expected_map(
            self.v, self.c_grid, win or self.win, self.n_total, self.t_size, self.y_size,
            num_j, np.random.default_rng(seed), S,
        )

    def test_shapes_and_signs(self):
        f = self.expected()
        self.assertEqual(f.mean.shape, (60, 320))
        self.assertEqual(f.std.shape, (60, 320))
        self.assertEqual(f.proj_t_mean.shape, (60,))
        self.assertEqual(f.proj_y_mean.shape, (320,))
        self.assertTrue(np.all(f.mean >= 0))
        self.assertTrue(np.all(f.std >= 0))

    def test_mass_is_conserved(self):
        """Over a window holding all the support the expected counts add up to the total ions."""
        f = self.expected()
        np.testing.assert_allclose(f.mean.sum(), self.n_total, rtol=0.02)
        # the projections are the marginals of the same map
        np.testing.assert_allclose(f.proj_t_mean.sum(), f.mean.sum(), rtol=1e-6)
        np.testing.assert_allclose(f.proj_y_mean.sum(), f.mean.sum(), rtol=1e-6)
        np.testing.assert_allclose(f.proj_t_mean, f.mean.sum(axis=1), rtol=1e-6)

    def test_background_alone_is_uniform(self):
        v = self.v._replace(alpha=np.array([1e9, 1.0, 1.0, 1.0, 1.0]))  # pi_0 ~ 1
        f = plot_cavi.expected_map(
            v, self.c_grid, self.win, self.n_total, self.t_size, self.y_size, 4,
            np.random.default_rng(0), 8,
        )
        np.testing.assert_allclose(f.mean, self.n_total / f.mean.size, rtol=1e-3)

    def test_uncertainty_follows_the_posterior(self):
        # a sharp posterior gives (almost) no spread, the prior gives clearly more
        sharp = self.v._replace(
            alpha=self.v.alpha * 1e6,
            nu=self.v.nu * 1e6,
            a_k=self.v.a_k * 1e6,
            b_k=self.v.b_k * 1e6,
            tau=self.v.tau * 1e-12,
            a_s=self.v.a_s * 1e6,
            b_s=self.v.b_s * 1e6,
            a_p=self.v.a_p * 1e6,
            b_p=self.v.b_p * 1e6,
        )
        broad = self.expected()
        narrow = plot_cavi.expected_map(
            sharp, self.c_grid, self.win, self.n_total, self.t_size, self.y_size, 12,
            np.random.default_rng(0), 16,
        )
        self.assertLess(narrow.std.max(), 0.05 * broad.std.max())

    def test_same_seed_same_result(self):
        a, b = self.expected(seed=3), self.expected(seed=3)
        np.testing.assert_array_equal(a.mean, b.mean)
        np.testing.assert_array_equal(a.std, b.std)


class TestVideos(TestCase):
    """Smoke tests: two frames of a small window are rendered to a non-empty file."""

    def setUp(self):
        p = default_hyperparameters(background_alpha=20.0)
        v = init_variational_params(p)
        v = type(v)(*(np.asarray(x) for x in v))
        self.win = plot_cavi.Window(np.linspace(7.0, 12.0, 21), np.linspace(599.5, 601.5, 41))
        frames = [
            plot_cavi.expected_map(
                s, np.asarray(p.c_grid), self.win, 1000.0, 60.0, 800.0, 3, np.random.default_rng(0), 4
            )
            for s in (v, v._replace(tau=v.tau * 0.01))
        ]
        self.frames = frames
        rng = np.random.default_rng(1)
        self.actual = rng.poisson(3.0, frames[0].mean.shape).astype(float)
        self.labels = ["prior", "sweep 1"]

    def test_make_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = plot_cavi.make_video(
                Path(tmp) / "x", self.win, self.actual, self.frames, self.labels, "test", fps=2
            )
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 1000)

    def test_make_video_3d(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = plot_cavi.make_video_3d(
                Path(tmp) / "x3d", self.win, self.actual, self.frames, self.labels, "test", fps=2
            )
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 1000)


class TestIntensityPlot(TestCase):

    def features(self):
        return [{"rt_apex_min": 10.0 * (i + 1), "mono_mz": 600.0 + 50 * i, "charge": 2, "intensity": v}
                for i, v in enumerate([5000.0, 2000.0, 800.0, 300.0])]

    def result(self, fitted):
        fitted = np.asarray(fitted, float)
        return {"intensity": fitted, "n_components": (fitted > 0).astype(int) * 2, "other": 120.0,
                "background": 4000.0, "owner": np.zeros(3, int)}

    def test_plot_is_written_and_stats_are_returned(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, stats = plot_cavi.plot_intensity(
                self.result([2100.0, 900.0, 310.0, 130.0]), self.features(), 9000.0, Path(tmp)
            )
            self.assertEqual(path.name, "intensity.png")
            self.assertGreater(path.stat().st_size, 5000)
        self.assertAlmostEqual(stats["median_ratio"], 0.42, delta=0.03)
        self.assertGreater(stats["pearson_log"], 0.99)

    def test_a_feature_without_a_component_does_not_break_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, stats = plot_cavi.plot_intensity(
                self.result([2100.0, 900.0, 310.0, 0.0]), self.features(), 9000.0, Path(tmp)
            )
            self.assertTrue(path.exists())
        self.assertEqual(stats["n_compared"], 3)

    def test_truth_json_gives_the_intensity_of_a_feature(self):
        import json

        truth = {"features": [{"feature": 1, "rt_apex_min": 10.0, "mono_mz": 600.0, "charge": 2,
                               "total_intensity": 1234.0}]}
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "demo.truth.json").write_text(json.dumps(truth))
            ref = plot_cavi.find_reference(Path(tmp) / "demo.mzML", None)
        self.assertEqual(ref["features"][0]["intensity"], 1234.0)

    def test_no_reference_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(plot_cavi.find_reference(Path(tmp) / "nothing.mzML", None))


class TestZoom(TestCase):

    def test_limits_ignore_the_first_sweeps(self):
        v = np.array([5.0, 1.0, 0.021, 0.020, 0.0205, 0.0201])
        lo, hi = plot_cavi.zoom_limits([v], burn=2)
        self.assertGreater(lo, 0.0)
        self.assertLess(hi, 0.05)  # the 5.0 and 1.0 do not stretch the axis
        self.assertLess(lo, 0.020)  # but everything from sweep 2 on fits, with some margin
        self.assertGreater(hi, 0.021)

    def test_limits_cover_all_later_arrays_and_include(self):
        lo, hi = plot_cavi.zoom_limits([np.array([9.0, 9.0, 1.0, 2.0]), np.array([9.0, 9.0, -1.0, 0.5])], 2, include=0.0)
        self.assertLess(lo, -1.0)
        self.assertGreater(hi, 2.0)
        lo, hi = plot_cavi.zoom_limits([np.array([3.0, 3.0, 5.0, 6.0])], 2, include=0.0)
        self.assertLessEqual(lo, 0.0)

    def test_constant_values_give_a_valid_range(self):
        lo, hi = plot_cavi.zoom_limits([np.full(6, 2e-6)], 2)
        self.assertLess(lo, 2e-6)
        self.assertGreater(hi, 2e-6)
        self.assertGreater(hi - lo, 0.0)

    def test_note_only_when_early_sweeps_are_cut(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        plot_cavi.zoom_axis(ax, [np.array([5.0, 1.0, 0.02, 0.021])], burn=2)
        self.assertEqual(len(ax.texts), 1)
        fig2, ax2 = plt.subplots()
        plot_cavi.zoom_axis(ax2, [np.array([0.02, 0.0205, 0.02, 0.021])], burn=2)
        self.assertEqual(len(ax2.texts), 0)  # nothing was cut, no note
        plt.close("all")

    def test_metrics_figure_uses_the_zoomed_axes(self):
        import jax.numpy as jnp

        from cavi.main import fit, random_hyperparameters
        from cavi.priors import VariationalParams

        rng = np.random.default_rng(0)
        t = jnp.array(rng.uniform(10, 12, 600), dtype=jnp.float32)
        y = jnp.array(rng.uniform(500, 502, 600), dtype=jnp.float32)
        w = jnp.array(rng.uniform(1, 3, 600), dtype=jnp.float32)
        p = random_hyperparameters(t, y, w, num_features=6, mz_range=(400.0, 700.0))
        v, _, hist = fit(init_variational_params(p), p, t, y, w, 3, 8)
        import jax

        hist = jax.tree.map(np.asarray, hist)
        v0 = jax.tree.map(np.asarray, init_variational_params(p))
        states = [v0] + [VariationalParams(*(f[i] for f in hist)) for i in range(8)]
        with tempfile.TemporaryDirectory() as tmp:
            path = plot_cavi.plot_metrics(states, p, None, Path(tmp), top=4)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 5000)
