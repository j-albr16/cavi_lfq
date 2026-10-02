from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pyopenms as oms
from scipy.ndimage import uniform_filter
from scipy.special import gammainc
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Int, jaxtyped

from .elbo import elbo
from .generative import DELTA_13C
from .priors import CHARGES, HyperParameters, VariationalParams
from .variational_updates import (
    carbon_count,
    heavy_isotope,
    m_k,
    nig,
    peak_width,
    pi,
    q_c,
)
from .z_j import get_log_rho_all, get_r


def hyperparameters_from_candidates(
    rt: Array,  # (K,) RT apex in minutes
    mz: Array,  # (K,) monoisotopic m/z
    charge: Array,  # (K,) integer charge
    c_grid: Array | None = None,
    n_grid: tuple[int, int] = (5, 120),
    rt_sd: float = 0.15,  # minutes, typical elution peak width
    mz_ppm: float = 10.0,  # std of the candidate mass
    mz_sd: float = 0.01,  # Da, typical centroid m/z jitter (prior of the peak width s)
    charge_prob: float = 0.99,
    tau: float | None = None,  # absolute std of the candidate mass in Da, overrides `mz_ppm`
) -> HyperParameters:
    """Priors for K candidate features given as (RT apex, monoisotopic m/z, charge).

    The carbon count prior is centred on the averagine estimate (~0.0444 carbons per Da of
    neutral mass). `charge_prob` is the prior mass on the candidate's charge. Keep it high: the
    isotope peaks j >= 1 sit at m + j * Delta / c, and any mass left on a wrong charge makes their
    expected squared residual huge for a narrow peak width, so mean-field drops them in the first
    sweep and q(p) collapses to 0 (seen with charge_prob = 0.7 on real data).
    """
    rt = jnp.asarray(rt, dtype=jnp.float32)
    mz = jnp.asarray(mz, dtype=jnp.float32)
    charge = jnp.asarray(charge)
    K = rt.shape[0]

    rho = jnp.full((K, 4), (1 - charge_prob) / 3)
    one_hot = jax.nn.one_hot(jnp.clip(charge, 1, 4) - 1, 4, dtype=bool)
    rho = jnp.where(one_hot, charge_prob, rho)

    if c_grid is None:
        c_grid = jnp.arange(n_grid[0], n_grid[1], dtype=jnp.int32)
    carbons = 0.0444 * (mz - 1.007276) * jnp.clip(charge, 1, 4)  # averagine
    sd = 0.1 * carbons + 2.0
    n = jnp.exp(-0.5 * ((c_grid[None, :] - carbons[:, None]) / sd[:, None]) ** 2)
    n = n / n.sum(axis=1, keepdims=True)

    return HyperParameters(
        alpha=jnp.ones(K + 1),
        m_0=rt,
        nu_0=jnp.ones(K),
        a_0=jnp.full((K,), 3.0),
        b_0=jnp.full((K,), 2.0 * rt_sd**2),  # E[sigma^2] = b / (a - 1) = rt_sd^2
        m_hat=mz,
        tau=jnp.full((K,), tau) if tau is not None else mz * mz_ppm * 1e-6,
        rho=rho,
        n=n,
        c_grid=c_grid,
        a_p=jnp.array(1.07),  # natural 13C abundance
        b_p=jnp.array(98.93),
        a_s=jnp.array(5.0),
        b_s=jnp.array(4.0 * mz_sd**2),  # E[s^2] = mz_sd^2
    )


def random_hyperparameters(
    t: Float[Array, "N"],
    y: Float[Array, "N"],
    w: Float[Array, "N"],
    num_features: int = 30,
    seed: int = 0,
    init: str = "pixels",
    charge_probs: tuple[float, float, float, float] = (0.1, 0.5, 0.3, 0.1),
    mz_range: tuple[float, float] = (300.0, 2000.0),
    mz_sd: float = 0.05,
    tau: float = 0.1,
    rt_sd: float = 0.5,
    nu_0: float = 0.1,
    alpha: float = 1.0,
) -> HyperParameters:
    """Priors for `num_features` candidates scattered at random over the spectrum.

    No peak picker is used. The positions (RT, m/z) are drawn at random inside `mz_range`, a range
    that makes sense for peptides and proteins; `init` selects how:

    - `"pixels"`: at randomly chosen pixels, with probability proportional to the pixel weight, so
      bright regions get more candidates and every candidate starts on data;
    - `"uniform"`: uniformly over the RT and m/z window. A candidate that does not overlap any
      pixel never receives responsibility, so with narrow peaks almost none of them ever moves.

    The charge of a candidate is drawn from `charge_probs` (typical for peptides, only used for the
    averagine carbon-count prior); the prior over the charge is the same weak `charge_probs` for
    all. The remaining priors are broad: the mass of a candidate may move by about `tau` Da and its
    RT by about `rt_sd` min, `mz_sd` is the prior guess of the m/z peak width (it is learned), and
    `alpha` is the Dirichlet pseudo-count of every component.
    """
    rng = np.random.default_rng(seed)
    t_np, y_np, w_np = np.asarray(t), np.asarray(y), np.asarray(w)
    inside = (y_np >= mz_range[0]) & (y_np <= mz_range[1])
    if not inside.any():
        raise ValueError(f"no pixel with m/z in {mz_range}")
    if init == "pixels":
        idx = rng.choice(np.flatnonzero(inside), num_features, p=w_np[inside] / w_np[inside].sum())
        rt, mz = t_np[idx], y_np[idx]
    elif init == "uniform":
        lo, hi = max(float(y_np.min()), mz_range[0]), min(float(y_np.max()), mz_range[1])
        rt = rng.uniform(t_np.min(), t_np.max(), num_features)
        mz = rng.uniform(lo, hi, num_features)
    else:
        raise ValueError(f"init must be 'pixels' or 'uniform', got {init!r}")
    charge = rng.choice([1, 2, 3, 4], num_features, p=np.asarray(charge_probs))
    p = hyperparameters_from_candidates(rt, mz, charge, mz_sd=mz_sd, tau=tau, rt_sd=rt_sd)
    return p._replace(
        alpha=jnp.full((num_features + 1,), alpha),
        rho=jnp.tile(jnp.asarray(charge_probs, dtype=jnp.float32), (num_features, 1)),
        nu_0=jnp.full((num_features,), nu_0),
    )


def hyperparameters_from_features(path: str, **kwargs) -> HyperParameters:
    """Priors from a featureXML, e.g. the OpenMS FeatureFinderCentroided result.

    K is the number of features in the file; nothing is filtered.
    """
    fm = oms.FeatureMap()
    oms.FeatureXMLFile().load(str(path), fm)
    feats = list(fm)
    return hyperparameters_from_candidates(
        [f.getRT() / 60 for f in feats],  # minutes
        [f.getMZ() for f in feats],
        [f.getCharge() for f in feats],
        **kwargs,
    )


def append_candidates(
    p: HyperParameters, rt: Array, mz: Array, charge: Array, **kwargs
) -> HyperParameters:
    """Add components to a prior. The global parameters and the carbon grid are kept."""
    new = hyperparameters_from_candidates(rt, mz, charge, c_grid=p.c_grid, **kwargs)
    cat = lambda a, b: jnp.concatenate([a, b], axis=0)
    return p._replace(
        alpha=cat(p.alpha, new.alpha[1:]),  # alpha_0 stays the background entry
        m_0=cat(p.m_0, new.m_0),
        nu_0=cat(p.nu_0, new.nu_0),
        a_0=cat(p.a_0, new.a_0),
        b_0=cat(p.b_0, new.b_0),
        m_hat=cat(p.m_hat, new.m_hat),
        tau=cat(p.tau, new.tau),
        rho=cat(p.rho, new.rho),
        n=cat(p.n, new.n),
    )


@partial(jax.jit, static_argnames=("num_j",))
def _background_weight(v, p, t, y, w, num_j):
    """w_n r_n0: the weight that the background claims in each pixel."""
    T, Y = jnp.max(t) - jnp.min(t), jnp.max(y) - jnp.min(y)
    _, r_0 = get_r(v, p, y, t, num_j, T, Y)
    return w * r_0


def propose_candidates(
    v: VariationalParams,
    p: HyperParameters,
    t: Float[Array, "N"],
    y: Float[Array, "N"],
    w: Float[Array, "N"],
    num_j: int,
    n_new: int = 4,
    t_bin: float = 0.25,  # minutes
    y_bin: float | None = None,  # Da; default: twice the fitted peak width s
    min_ions: float = 30.0,
    alpha: float = 0.01,  # family-wise false positive level of the test below
) -> dict[str, np.ndarray]:
    """New candidates where the background claims more weight than a uniform background would.

    The residual map is the weight w_n r_n0 that the current fit gives to the background,
    histogrammed in (t, y) with bins of 0.25 min and twice the fitted peak width. A box of 5 x 1 bins counts as a hit if its residual weight is Poisson
    significant against the typical level of the map (Bonferroni over all boxes) and holds at least
    `min_ions`. Hits are taken greedily, strongest first. The charge is the one whose first isotope
    tooth, y + Delta / c, holds the most residual weight, if that is clearly above the noise
    (otherwise 2). The whole comb is then
    claimed at all retention times, so that neither its isotope peaks nor the several local
    maxima of a broad elution profile become candidates of their own. Two features with the same
    m/z at different RT are therefore found in successive rounds, not in the same one.
    """
    rw = np.asarray(_background_weight(v, p, t, y, w, num_j))
    t_np, y_np = np.asarray(t), np.asarray(y)
    s_fit = float(np.sqrt(v.b_s / (v.a_s - 1.0)))  # current m/z peak width
    if y_bin is None:
        y_bin = float(np.clip(2.0 * s_fit, 0.005, 0.2))
    half = int(np.ceil(4.0 * s_fit / y_bin)) + 1  # a peak spans about +-4 s: claim all of it
    t_edges = np.arange(t_np.min(), t_np.max() + t_bin, t_bin)
    y_edges = np.arange(y_np.min(), y_np.max() + y_bin, y_bin)
    H, _, _ = np.histogram2d(t_np, y_np, bins=[t_edges, y_edges], weights=rw)

    t_box = 5
    S = uniform_filter(H, size=(t_box, 1), mode="constant") * t_box  # weight in each box
    level = H[H <= np.quantile(H, 0.99)].mean()  # trimmed mean: the background level per bin
    mu = max(level * t_box, 1e-9)
    p_value = gammainc(np.maximum(S, 1e-12), mu)  # P(Poisson(mu) >= S) for non-integer S
    score = -np.log(np.maximum(p_value, 1e-300))
    score[S < min_ions] = 0.0
    threshold = -np.log(alpha / H.size)

    # Teeth j >= num_j of the existing features are not part of the model, so the fit leaves them
    # in the background. They are not new features: claim them (within +-4 sigma_t of the apex).
    c_best = np.argmax(np.asarray(v.q_c), axis=1)
    sigma_t = np.sqrt(np.asarray(v.b_k) / (np.asarray(v.a_k) - 1.0))
    for k in range(len(c_best)):
        rows = slice(
            max(int(np.floor((float(v.mu[k]) - 4 * sigma_t[k] - t_edges[0]) / t_bin)), 0),
            int(np.ceil((float(v.mu[k]) + 4 * sigma_t[k] - t_edges[0]) / t_bin)) + 1,
        )
        c = int(CHARGES[c_best[k]])
        for jj in range(num_j, num_j + 4):
            col = int(np.floor((float(v.m[k, c_best[k]]) + jj * DELTA_13C / c - y_edges[0]) / y_bin))
            score[rows, max(col - half, 0) : col + half + 1] = 0.0

    out = {"rt": [], "mz": [], "charge": [], "ions": []}
    for _ in range(n_new):
        i, j = np.unravel_index(np.argmax(score), score.shape)
        if score[i, j] < threshold:
            break
        t_c, y_c = 0.5 * (t_edges[i] + t_edges[i + 1]), 0.5 * (y_edges[j] + y_edges[j + 1])
        near = (np.abs(t_np - t_c) <= 1.5) & (np.abs(y_np - y_c) <= y_bin)
        rt = float(np.sum(rw[near] * t_np[near]) / np.sum(rw[near]))
        mz = float(np.sum(rw[near] * y_np[near]) / np.sum(rw[near]))
        in_rt = np.abs(t_np - rt) <= 0.5
        tooth = [
            np.sum(rw[in_rt & (np.abs(y_np - (mz + DELTA_13C / c)) <= y_bin / 2)]) for c in (1, 2, 3, 4)
        ]
        own = np.sum(rw[in_rt & (np.abs(y_np - mz) <= y_bin / 2)])
        needed = max(5.0 * level * (1.0 / t_bin), 0.01 * own)  # a tooth must stand out of the noise
        charge = int(np.argmax(tooth)) + 1 if max(tooth) > needed else 2
        out["rt"].append(rt)
        out["mz"].append(mz)
        out["charge"].append(charge)
        out["ions"].append(float(S[i, j]))
        # claim the whole comb at *all* retention times: a broad elution profile has several local
        # maxima along t, which would otherwise become several candidates of the same feature
        for jj in range(num_j):
            col = int(np.floor((mz + jj * DELTA_13C / charge - y_edges[0]) / y_bin))
            score[:, max(col - half, 0) : col + half + 1] = 0.0
    return {k: np.array(v_) for k, v_ in out.items()}


def fit_with_new_features(
    t: Float[Array, "N"],
    y: Float[Array, "N"],
    w: Float[Array, "N"],
    p: HyperParameters,
    num_j: int,
    num_iter: int,
    rounds: int = 2,
    n_new: int = 4,
    new_mass_sd: float = 0.03,  # Da; the proposed masses are only accurate to the bin size
    **propose_kwargs,
) -> tuple[HyperParameters, dict[str, np.ndarray]]:
    """Grow the candidate list: fit, look at what the fit leaves in the background, add, repeat.

    Candidates without support end with zero weight, so adding too many only costs time.
    Returns the extended prior and the proposed candidates (rt, mz, charge, ions).
    """
    added = {"rt": [], "mz": [], "charge": [], "ions": []}
    for _ in range(rounds):
        v, _, _ = fit(init_variational_params(p), p, t, y, w, num_j, num_iter)
        new = propose_candidates(v, p, t, y, w, num_j, n_new, **propose_kwargs)
        if len(new["rt"]) == 0:
            break
        p = append_candidates(p, new["rt"], new["mz"], new["charge"], tau=new_mass_sd)
        for k in added:
            added[k].extend(new[k].tolist())
    return p, {k: np.array(v_) for k, v_ in added.items()}


def get_gamma(w: Float[Array, "N"]) -> Float[Array, ""]:
    return 1 / jnp.min(w)


def load_spectrum(path: str, return_scale: bool = False):
    """Pixels (t in min, y, w) of the MS1 spectra; with `return_scale` also the intensity of one ion.

    w = I / min(I) is the effective ion count, so `scale = min(I)` converts counts back to the
    intensity units of the file (intensity = scale * w).
    """
    exp = oms.MSExperiment()
    oms.MzMLFile().load(path, exp)
    t, y, w = [], [], []
    for spec in exp:
        if spec.getMSLevel() != 1:
            continue  # only MS1 survey scans
        mz, inten = spec.get_peaks()
        t.append(jnp.full(mz.shape, spec.getRT() / 60))
        y.append(mz)
        w.append(inten)
    t, y, w = map(jnp.concatenate, (t, y, w))
    scale = float(1.0 / get_gamma(w))
    w = get_gamma(w) * w  # intensities -> effective ion counts
    return (t, y, w, scale) if return_scale else (t, y, w)


def step(
    v: VariationalParams,
    p: HyperParameters,
    t: Float[Array, "N"],
    y: Float[Array, "N"],
    w: Float[Array, "N"],
    num_j: int,
    T: Float[Array, ""],
    Y: Float[Array, ""],
) -> tuple[VariationalParams, Float[Array, ""]]:
    """One CAVI sweep in the order of the algorithm; returns the new q and its ELBO."""
    r, r_0 = get_r(v, p, y, t, num_j, T, Y)
    R = jnp.sum(r, axis=-1)  # N K
    N_k = jnp.sum(w[:, None] * R, axis=0)  # K
    N_0 = jnp.sum(w * r_0)  # background

    alpha = pi(p, jnp.concatenate([N_0[None], N_k]))
    nu, mu, a, b = nig(p, R, N_k, w, t)

    tau2, m = m_k(v, p, y, w, r, N_k)
    qc = q_c(v, p, y, w, r, m)
    v = v._replace(alpha=alpha, nu=nu, mu=mu, a_k=a, b_k=b, tau=tau2, m=m, q_c=qc)

    # q(s^2) uses the *updated* q(m, c), q(n) the old q(p), q(p) the new q(n)
    a_s, b_s = peak_width(v, p, N_k, w, r, qc, y)
    v = v._replace(a_s=a_s, b_s=b_s)
    q_n = carbon_count(v, p, N_k, w, r)
    v = v._replace(q_n=q_n)
    a_p, b_p = heavy_isotope(p, N_k, w, r, q_n)
    v = v._replace(a_p=a_p, b_p=b_p)

    log_rho, log_rho_0 = get_log_rho_all(v, p, y, t, num_j, T, Y)
    return v, elbo(v, p, w, r, r_0, log_rho, log_rho_0)


def init_variational_params(p: HyperParameters) -> VariationalParams:
    """Start q at the prior."""
    return VariationalParams(
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
        b_k=p.b_0,
        tau=p.tau**2,  # variational variance tau_tilde^2; the prior tau is a std
        m=jnp.broadcast_to(p.m_hat[:, None], (p.m_hat.shape[0], p.rho.shape[1])),  # K x C
    )


@partial(jax.jit, static_argnames=("num_j", "num_iter"))
def fit(
    v: VariationalParams,
    p: HyperParameters,
    t: Float[Array, "N"],
    y: Float[Array, "N"],
    w: Float[Array, "N"],
    num_j: int,
    num_iter: int,
) -> tuple[VariationalParams, Float[Array, "I"], VariationalParams]:
    """Run `num_iter` CAVI sweeps. Returns the final q, the ELBO trace and the history of q."""
    T = jnp.max(t) - jnp.min(t)  # size of the observed RT window
    Y = jnp.max(y) - jnp.min(y)  # size of the observed m/z window

    def body(v, _):
        v, e = step(v, p, t, y, w, num_j, T, Y)
        return v, (e, v)

    v, (elbos, history) = jax.lax.scan(body, v, None, length=num_iter)
    return v, elbos, history


def run(
    spectrum: str,  # path to an mzML file
    p: HyperParameters,
    num_j: int = 4,  # number of isotope peaks per feature (j = 0 .. num_j - 1)
    num_iter: int = 20,
) -> tuple[VariationalParams, Float[Array, "I"], VariationalParams]:
    t, y, w = load_spectrum(spectrum)  # file I/O stays outside of jit
    return fit(init_variational_params(p), p, t, y, w, num_j, num_iter)
