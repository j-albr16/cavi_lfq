import argparse
import base64
import json
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Float, Int, ArrayLike, Array, PRNGKeyArray, jaxtyped
from beartype import beartype as typechecker

from cavi.priors import HyperParameters, Theta, sample_theta, sample_z_and_j

DELTA_13C = 1.00336  # mass difference 13C - 12C in Da


@jaxtyped(typechecker=typechecker)
def generative_model(
    key: PRNGKeyArray,
    theta: Theta,
    z: Int[Array, "N"],
    j: Int[Array, "N"],
    t_range: Float[ArrayLike, "2"],
    y_range: Float[ArrayLike, "2"],
    delta: float,
) -> tuple[Float[Array, "N"], Float[Array, "N"]]:
    t_range = jnp.asarray(t_range)
    y_range = jnp.asarray(y_range)

    t_key, y_key, bg_t_key, bg_y_key = jr.split(key, 4)

    is_background = z == 0
    k_idx = jnp.clip(z - 1, 0, theta.mu.shape[0] - 1)

    # t_n | z_n = k >= 1 ~ N(mu_k, sigma_k^2)
    mu_k = theta.mu[k_idx]
    sigma_k = jnp.sqrt(theta.sigma2[k_idx])
    t_feature = mu_k + sigma_k * jr.normal(t_key, shape=z.shape)

    # y_n | z_n = k >= 1, j_n ~ N(m_k + j_n * delta / c_k, s^2)
    y_mean = theta.m[k_idx] + (j * delta) / theta.c[k_idx]
    s = jnp.sqrt(theta.s2)
    y_feature = y_mean + s * jr.normal(y_key, shape=z.shape)

    # background (z_n = 0): (t_n, y_n) uniform over the window
    t_bg = jr.uniform(bg_t_key, shape=z.shape, minval=t_range[0], maxval=t_range[1])
    y_bg = jr.uniform(bg_y_key, shape=z.shape, minval=y_range[0], maxval=y_range[1])

    t = jnp.where(is_background, t_bg, t_feature)
    y = jnp.where(is_background, y_bg, y_feature)

    return t, y


# ---------------------------------------------------------------------------
# Synthetic LC-MS map that can be loaded in OpenMS (mzML) to validate the algorithm
# ---------------------------------------------------------------------------


def default_hyperparameters(background_alpha: float = 2000.0) -> HyperParameters:
    """Prior with K=4 well separated features; a large alpha_0 makes most ions background."""
    K, G = 4, 5
    c_grid = jnp.array([30, 40, 50, 60, 70], dtype=jnp.int32)

    # carbon-count prior per feature, peaked near a different grid slot each
    n_probs = jnp.eye(K, G) * 0.6 + jnp.full((K, G), 0.4 / G)
    n_probs = n_probs / n_probs.sum(axis=-1, keepdims=True)

    alpha = jnp.full((K + 1,), 2.0).at[0].set(background_alpha)
    return HyperParameters(
        alpha=alpha,
        m_0=jnp.array([10.0, 25.0, 40.0, 55.0]),
        nu_0=jnp.full((K,), 5.0),
        a_0=jnp.full((K,), 4.0),
        b_0=jnp.full((K,), 3.0),
        m_hat=jnp.array([600.0, 750.0, 900.0, 1050.0]),
        tau=jnp.full((K,), 0.05),
        rho=jnp.tile(jnp.array([0.1, 0.6, 0.2, 0.1]), (K, 1)),
        n=n_probs,
        c_grid=c_grid,
        a_p=jnp.array(1.07),  # natural 13C abundance ~1.07%
        b_p=jnp.array(98.93),
        a_s=jnp.array(5.0),
        b_s=jnp.array(0.01),
    )


class SimulatedIons(NamedTuple):
    t: np.ndarray  # retention time of every ion (min)
    y: np.ndarray  # m/z of every ion
    z: np.ndarray  # 0 = background, k >= 1 = feature k
    j: np.ndarray  # isotope index
    theta: Theta


def simulate_ions(
    key: PRNGKeyArray,
    h_params: HyperParameters,
    n_ions: int,
    t_range: tuple[float, float] = (0.0, 60.0),
    y_range: tuple[float, float] = (400.0, 1200.0),
    delta: float = DELTA_13C,
) -> SimulatedIons:
    key_theta, key_zj, key_gen = jr.split(key, 3)
    theta = sample_theta(key_theta, h_params)
    z, j = sample_z_and_j(key_zj, theta, n_ions)
    t, y = generative_model(key_gen, theta, z, j, jnp.array(t_range), jnp.array(y_range), delta)
    return SimulatedIons(np.asarray(t), np.asarray(y), np.asarray(z), np.asarray(j), theta)


def bin_ions(
    t: np.ndarray,
    y: np.ndarray,
    t_range: tuple[float, float],
    y_range: tuple[float, float],
    n_scans: int,
    mz_bin: float,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Histogram ions into scans (RT bins) and m/z bins, keeping only non-empty bins.

    Returns the scan retention times (bin centres) and, per scan, the sorted m/z bin
    centres with their ion counts. A dense (scans x bins) matrix is never built.
    """
    n_mz = int(np.ceil((y_range[1] - y_range[0]) / mz_bin))
    scan_width = (t_range[1] - t_range[0]) / n_scans
    inside = (t >= t_range[0]) & (t < t_range[1]) & (y >= y_range[0]) & (y < y_range[1])
    scan = np.floor((t[inside] - t_range[0]) / scan_width).astype(np.int64)
    mz_idx = np.floor((y[inside] - y_range[0]) / mz_bin).astype(np.int64)
    keys, counts = np.unique(scan * n_mz + mz_idx, return_counts=True)  # sorted by scan, then m/z
    key_scan, key_mz = keys // n_mz, keys % n_mz

    rt = t_range[0] + (np.arange(n_scans) + 0.5) * scan_width
    bounds = np.searchsorted(key_scan, np.arange(n_scans + 1))
    spectra = []
    for s in range(n_scans):
        sl = slice(bounds[s], bounds[s + 1])
        mz = y_range[0] + (key_mz[sl] + 0.5) * mz_bin
        spectra.append((mz.astype(np.float64), counts[sl].astype(np.float64)))
    return rt, spectra


def _b64(array: np.ndarray, dtype: str) -> str:
    return base64.b64encode(np.asarray(array, dtype=dtype).tobytes()).decode("ascii")


def write_mzml(
    path: str | Path,
    rt_minutes: np.ndarray,
    spectra: list[tuple[np.ndarray, np.ndarray]],
    y_range: tuple[float, float],
    profile: bool = False,
) -> None:
    """Write MS1 spectra as an (unindexed) mzML 1.1 file; no external dependency needed.

    m/z is stored as 64-bit and intensity as 32-bit floats, uncompressed. With
    `profile=False` every stored m/z bin is flagged as a centroid, otherwise as profile data.
    """
    mode_acc, mode_name = ("MS:1000128", "profile spectrum") if profile else ("MS:1000127", "centroid spectrum")
    out = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<mzML xmlns="http://psi.hupo.org/ms/mzml" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xsi:schemaLocation="http://psi.hupo.org/ms/mzml http://psidev.info/files/ms/mzML/xsd/mzML1.1.0.xsd" '
        'version="1.1.0">',
        '<cvList count="3">',
        '<cv id="MS" fullName="Proteomics Standards Initiative Mass Spectrometry Ontology" version="4.1.0" '
        'URI="https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo"/>',
        '<cv id="UO" fullName="Unit Ontology" version="09:04:2014" URI="https://raw.githubusercontent.com/bio-ontology-research-group/unit-ontology/master/unit.obo"/>',
        '<cv id="CUSTOM" fullName="synthetic data" version="1" URI="https://example.invalid"/>',
        "</cvList>",
        "<fileDescription><fileContent>",
        '<cvParam cvRef="MS" accession="MS:1000579" name="MS1 spectrum" value=""/>',
        f'<cvParam cvRef="MS" accession="{mode_acc}" name="{mode_name}" value=""/>',
        "</fileContent></fileDescription>",
        '<softwareList count="1"><software id="lfq_generative" version="1.0">'
        '<cvParam cvRef="MS" accession="MS:1000799" name="custom unreleased software tool" value="lfq generative model"/>'
        "</software></softwareList>",
        '<instrumentConfigurationList count="1"><instrumentConfiguration id="IC1">'
        '<cvParam cvRef="MS" accession="MS:1000031" name="instrument model" value="simulated"/>'
        "</instrumentConfiguration></instrumentConfigurationList>",
        '<dataProcessingList count="1"><dataProcessing id="dp1">'
        '<processingMethod order="0" softwareRef="lfq_generative">'
        '<cvParam cvRef="MS" accession="MS:1000544" name="Conversion to mzML" value=""/>'
        "</processingMethod></dataProcessing></dataProcessingList>",
        '<run id="synthetic_run" defaultInstrumentConfigurationRef="IC1">',
        f'<spectrumList count="{len(spectra)}" defaultDataProcessingRef="dp1">',
    ]
    for i, (rt, (mz, inten)) in enumerate(zip(rt_minutes, spectra)):
        tic = float(inten.sum())
        peaks = [
            '<cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="1"/>',
            f'<cvParam cvRef="MS" accession="{mode_acc}" name="{mode_name}" value=""/>',
            '<cvParam cvRef="MS" accession="MS:1000130" name="positive scan" value=""/>',
            f'<cvParam cvRef="MS" accession="MS:1000285" name="total ion current" value="{tic:.6g}"/>',
        ]
        if len(mz):
            base = int(np.argmax(inten))
            peaks += [
                f'<cvParam cvRef="MS" accession="MS:1000504" name="base peak m/z" value="{mz[base]:.6f}" unitCvRef="MS" unitAccession="MS:1000040" unitName="m/z"/>',
                f'<cvParam cvRef="MS" accession="MS:1000505" name="base peak intensity" value="{inten[base]:.6g}" unitCvRef="MS" unitAccession="MS:1000131" unitName="number of detector counts"/>',
                f'<cvParam cvRef="MS" accession="MS:1000528" name="lowest observed m/z" value="{mz[0]:.6f}" unitCvRef="MS" unitAccession="MS:1000040" unitName="m/z"/>',
                f'<cvParam cvRef="MS" accession="MS:1000527" name="highest observed m/z" value="{mz[-1]:.6f}" unitCvRef="MS" unitAccession="MS:1000040" unitName="m/z"/>',
            ]
        out.append(f'<spectrum index="{i}" id="scan={i + 1}" defaultArrayLength="{len(mz)}">')
        out += peaks
        out += [
            '<scanList count="1"><cvParam cvRef="MS" accession="MS:1000795" name="no combination" value=""/>',
            '<scan><cvParam cvRef="MS" accession="MS:1000016" name="scan start time" '
            f'value="{rt:.6f}" unitCvRef="UO" unitAccession="UO:0000031" unitName="minute"/>',
            '<scanWindowList count="1"><scanWindow>'
            f'<cvParam cvRef="MS" accession="MS:1000501" name="scan window lower limit" value="{y_range[0]}" unitCvRef="MS" unitAccession="MS:1000040" unitName="m/z"/>'
            f'<cvParam cvRef="MS" accession="MS:1000500" name="scan window upper limit" value="{y_range[1]}" unitCvRef="MS" unitAccession="MS:1000040" unitName="m/z"/>'
            "</scanWindow></scanWindowList></scan></scanList>",
        ]
        mz_b64, in_b64 = _b64(mz, "<f8"), _b64(inten, "<f4")
        out += [
            '<binaryDataArrayList count="2">',
            f'<binaryDataArray encodedLength="{len(mz_b64)}">'
            '<cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" value=""/>'
            '<cvParam cvRef="MS" accession="MS:1000576" name="no compression" value=""/>'
            '<cvParam cvRef="MS" accession="MS:1000514" name="m/z array" value="" unitCvRef="MS" unitAccession="MS:1000040" unitName="m/z"/>'
            f"<binary>{mz_b64}</binary></binaryDataArray>",
            f'<binaryDataArray encodedLength="{len(in_b64)}">'
            '<cvParam cvRef="MS" accession="MS:1000521" name="32-bit float" value=""/>'
            '<cvParam cvRef="MS" accession="MS:1000576" name="no compression" value=""/>'
            '<cvParam cvRef="MS" accession="MS:1000515" name="intensity array" value="" unitCvRef="MS" unitAccession="MS:1000131" unitName="number of detector counts"/>'
            f"<binary>{in_b64}</binary></binaryDataArray>",
            "</binaryDataArrayList>",
            "</spectrum>",
        ]
    out += ["</spectrumList>", "</run>", "</mzML>"]
    Path(path).write_text("\n".join(out), encoding="utf-8")


def ground_truth(sim: SimulatedIons, delta: float, intensity_per_ion: float) -> dict:
    """Per-feature truth to compare against FeatureFinder / the CAVI posterior."""
    theta = sim.theta
    features = []
    for k in range(theta.mu.shape[0]):
        in_k = sim.z == k + 1
        c = int(theta.c[k])
        features.append(
            {
                "feature": k + 1,
                "rt_apex_min": float(theta.mu[k]),
                "rt_sigma_min": float(np.sqrt(theta.sigma2[k])),
                "mono_mz": float(theta.m[k]),
                "charge": c,
                "n_carbons": int(theta.n[k]),
                "isotope_mz": [float(theta.m[k]) + jj * delta / c for jj in range(int(sim.j[in_k].max()) + 1 if in_k.any() else 1)],
                "n_ions": int(in_k.sum()),
                "total_intensity": float(in_k.sum()) * intensity_per_ion,
                "isotope_counts": np.bincount(sim.j[in_k]).tolist() if in_k.any() else [],
            }
        )
    return {
        "pi": np.asarray(theta.pi).tolist(),
        "p_13C": float(theta.p),
        "mz_sigma": float(np.sqrt(theta.s2)),
        "delta": delta,
        "intensity_per_ion": intensity_per_ion,
        "n_ions": int(sim.z.shape[0]),
        "n_background_ions": int((sim.z == 0).sum()),
        "features": features,
    }


def generate_spectrum(
    out_dir: str | Path = "data",
    name: str = "synthetic",
    seed: int = 0,
    n_ions: int = 1_000_000,
    t_range: tuple[float, float] = (0.0, 60.0),
    y_range: tuple[float, float] = (400.0, 1200.0),
    n_scans: int = 600,
    mz_bin: float = 0.01,
    intensity_per_ion: float = 100.0,
    background_alpha: float = 200.0,
    profile: bool = False,
    h_params: HyperParameters | None = None,
) -> tuple[Path, Path]:
    """Sample the generative model and store the LC-MS map as `<name>.mzML` plus `<name>.truth.json`.

    Each ion adds `intensity_per_ion` to the (scan, m/z bin) it falls into, so the
    intensity of a pixel is proportional to its effective ion count w_n.
    """
    h_params = h_params if h_params is not None else default_hyperparameters(background_alpha)
    sim = simulate_ions(jr.PRNGKey(seed), h_params, n_ions, t_range, y_range)
    rt, spectra = bin_ions(sim.t, sim.y, t_range, y_range, n_scans, mz_bin)
    spectra = [(mz, counts * intensity_per_ion) for mz, counts in spectra]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mzml_path = out_dir / f"{name}.mzML"
    truth_path = out_dir / f"{name}.truth.json"
    write_mzml(mzml_path, rt, spectra, y_range, profile=profile)
    truth_path.write_text(json.dumps(ground_truth(sim, DELTA_13C, intensity_per_ion), indent=2))
    return mzml_path, truth_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic LC-MS map as mzML.")
    parser.add_argument("--out", default="data", help="output directory")
    parser.add_argument("--name", default="synthetic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-ions", type=int, default=1_000_000)
    parser.add_argument("--n-scans", type=int, default=600)
    parser.add_argument("--mz-bin", type=float, default=0.01)
    parser.add_argument("--intensity-per-ion", type=float, default=100.0)
    parser.add_argument("--background-alpha", type=float, default=200.0)
    parser.add_argument("--profile", action="store_true", help="flag spectra as profile instead of centroided")
    args = parser.parse_args()
    mzml, truth = generate_spectrum(
        out_dir=args.out,
        name=args.name,
        seed=args.seed,
        n_ions=args.n_ions,
        n_scans=args.n_scans,
        mz_bin=args.mz_bin,
        intensity_per_ion=args.intensity_per_ion,
        background_alpha=args.background_alpha,
        profile=args.profile,
    )
    print(f"wrote {mzml}\nwrote {truth}")


if __name__ == "__main__":
    main()
