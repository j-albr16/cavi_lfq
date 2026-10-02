import base64
import json
import tempfile
from pathlib import Path
from unittest import TestCase, skipUnless

import numpy as np

from cavi.generative import (
    bin_ions,
    default_hyperparameters,
    generate_spectrum,
    simulate_ions,
    write_mzml,
)

import jax.random as jr

try:
    import pyopenms
except ImportError:  # optional: only used for the OpenMS round trip
    pyopenms = None

NS = {"m": "http://psi.hupo.org/ms/mzml"}


def read_mzml(path):
    """Minimal independent mzML reader: returns (rts, [(mz, intensity)], profile_flag)."""
    import xml.etree.ElementTree as ET

    root = ET.parse(path).getroot()
    spectra, rts = [], []
    for sp in root.iterfind(".//m:spectrum", NS):
        rts.append(
            float(sp.find(".//m:cvParam[@accession='MS:1000016']", NS).attrib["value"])
        )
        arrays = {}
        for arr in sp.iterfind(".//m:binaryDataArray", NS):
            accs = {c.attrib["accession"] for c in arr.iterfind("m:cvParam", NS)}
            dtype = "<f8" if "MS:1000523" in accs else "<f4"
            raw = base64.b64decode(arr.find("m:binary", NS).text or "")
            name = "mz" if "MS:1000514" in accs else "int"
            arrays[name] = np.frombuffer(raw, dtype=dtype)
            assert int(arr.attrib["encodedLength"]) == len(arr.find("m:binary", NS).text or "")
        assert int(sp.attrib["defaultArrayLength"]) == len(arrays["mz"]) == len(arrays["int"])
        spectra.append((arrays["mz"], arrays["int"]))
    profile = root.find(".//m:fileContent/m:cvParam[@accession='MS:1000128']", NS) is not None
    return np.array(rts), spectra, profile


class TestSpectrum(TestCase):

    T_RANGE = (0.0, 60.0)
    Y_RANGE = (400.0, 1200.0)

    def setUp(self):
        self.h = default_hyperparameters(background_alpha=20.0)
        self.sim = simulate_ions(jr.PRNGKey(1), self.h, 20_000, self.T_RANGE, self.Y_RANGE)

    def test_simulation_shapes_and_labels(self):
        sim = self.sim
        self.assertEqual(sim.t.shape, (20_000,))
        self.assertEqual(sim.y.shape, (20_000,))
        self.assertTrue(set(np.unique(sim.z)) <= set(range(5)))
        self.assertTrue(np.all(sim.j[sim.z == 0] == 0))
        # background ions are uniform in the window
        bg = sim.z == 0
        self.assertTrue(np.all((sim.t[bg] >= self.T_RANGE[0]) & (sim.t[bg] <= self.T_RANGE[1])))
        self.assertTrue(np.all((sim.y[bg] >= self.Y_RANGE[0]) & (sim.y[bg] <= self.Y_RANGE[1])))

    def test_bin_ions_matches_histogram2d(self):
        n_scans, mz_bin = 60, 0.5
        rt, spectra = bin_ions(self.sim.t, self.sim.y, self.T_RANGE, self.Y_RANGE, n_scans, mz_bin)
        n_mz = int(np.ceil((self.Y_RANGE[1] - self.Y_RANGE[0]) / mz_bin))
        ref, _, _ = np.histogram2d(
            self.sim.t,
            self.sim.y,
            bins=[n_scans, n_mz],
            range=[self.T_RANGE, (self.Y_RANGE[0], self.Y_RANGE[0] + n_mz * mz_bin)],
        )
        np.testing.assert_allclose(rt, (np.arange(n_scans) + 0.5) * 1.0)
        for s, (mz, counts) in enumerate(spectra):
            self.assertTrue(np.all(np.diff(mz) > 0), "m/z must be strictly increasing within a scan")
            idx = np.floor((mz - self.Y_RANGE[0]) / mz_bin).astype(int)
            np.testing.assert_array_equal(ref[s, idx], counts)
            self.assertEqual(ref[s].sum(), counts.sum())  # no ion lost, empty bins omitted
            self.assertTrue(np.all(counts > 0))

    def test_mzml_round_trip(self):
        rt, spectra = bin_ions(self.sim.t, self.sim.y, self.T_RANGE, self.Y_RANGE, 30, 0.05)
        spectra = [(mz, counts * 100.0) for mz, counts in spectra]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.mzML"
            write_mzml(path, rt, spectra, self.Y_RANGE)
            rt_back, spectra_back, profile = read_mzml(path)
        self.assertFalse(profile)
        np.testing.assert_allclose(rt_back, rt, atol=1e-5)
        self.assertEqual(len(spectra_back), 30)
        for (mz, inten), (mz_b, inten_b) in zip(spectra, spectra_back):
            np.testing.assert_array_equal(mz_b, mz)  # float64 is lossless
            np.testing.assert_allclose(inten_b, inten, rtol=1e-6)

    def test_profile_flag(self):
        rt, spectra = bin_ions(self.sim.t, self.sim.y, self.T_RANGE, self.Y_RANGE, 5, 0.1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.mzML"
            write_mzml(path, rt, spectra, self.Y_RANGE, profile=True)
            self.assertTrue(read_mzml(path)[2])

    def test_empty_scan_is_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.mzML"
            write_mzml(path, np.array([0.5]), [(np.array([]), np.array([]))], self.Y_RANGE)
            _, spectra, _ = read_mzml(path)
        self.assertEqual(len(spectra[0][0]), 0)

    def test_generate_spectrum_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            mzml, truth = generate_spectrum(
                out_dir=Path(tmp) / "data",
                name="demo",
                seed=3,
                n_ions=30_000,
                n_scans=60,
                mz_bin=0.05,
                intensity_per_ion=50.0,
                background_alpha=20.0,
            )
            self.assertEqual(mzml.name, "demo.mzML")
            self.assertTrue(mzml.exists() and truth.exists())
            rt, spectra, _ = read_mzml(mzml)
            info = json.loads(truth.read_text())

        total_intensity = sum(float(i.sum()) for _, i in spectra)
        # every ion inside the window contributes exactly intensity_per_ion
        self.assertLessEqual(total_intensity, info["n_ions"] * 50.0 + 1e-3)
        self.assertGreater(total_intensity, 0.9 * info["n_ions"] * 50.0)
        self.assertEqual(len(info["features"]), 4)
        self.assertEqual(
            info["n_background_ions"] + sum(f["n_ions"] for f in info["features"]),
            info["n_ions"],
        )
        for f in info["features"]:
            self.assertGreater(f["charge"], 0)
            self.assertAlmostEqual(f["isotope_mz"][0], f["mono_mz"])

    def test_generation_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            kwargs = dict(n_ions=5_000, n_scans=20, mz_bin=0.1, background_alpha=20.0)
            a, _ = generate_spectrum(out_dir=Path(tmp) / "a", seed=5, **kwargs)
            b, _ = generate_spectrum(out_dir=Path(tmp) / "b", seed=5, **kwargs)
            self.assertEqual(a.read_bytes(), b.read_bytes())

    @skipUnless(pyopenms is not None, "pyopenms not installed")
    def test_loadable_with_openms(self):
        with tempfile.TemporaryDirectory() as tmp:
            mzml, _ = generate_spectrum(
                out_dir=tmp, n_ions=10_000, n_scans=20, mz_bin=0.05, background_alpha=20.0
            )
            exp = pyopenms.MSExperiment()
            pyopenms.MzMLFile().load(str(mzml), exp)
        self.assertEqual(exp.getNrSpectra(), 20)
        self.assertEqual(exp.getSpectrum(0).getMSLevel(), 1)
