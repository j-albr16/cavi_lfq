"""Run CAVI on a spectrum from randomly scattered features and plot what happened.

No peak picker is used: `--num-features` candidates are scattered at random over the spectrum
(see `cavi.main.random_hyperparameters`) and CAVI decides which of them the data support.

Writes to `--out` (default plots/cavi_real):
  elbo.png                   ELBO per sweep and its increments
  metrics.png                convergence of the posterior summaries of the strongest components
  intensity.png              summed intensity of each reference feature, fitted vs. the reference
  fit_full.mp4               actual vs. fitted map over the whole spectrum, one frame per sweep
  fit_component<k>.mp4       the same, zoomed on one component (its isotope comb)
  fit_component<k>_3d.mp4    3D surface of the fitted expected counts for that component, coloured by uncertainty

Uncertainty is shown with colour. In the maps it is the posterior standard deviation of the
expected ion count per bin; in the projections the fitted curve is coloured by it. Both come from
Monte Carlo draws of all parameters from q, so they reflect the uncertainty of the fitted model.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, colors
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator
from scipy.special import ndtr
from scipy.stats import binom

from cavi.generative import DELTA_13C
from cavi.evaluate import compare_intensities, feature_intensities
from cavi.main import fit, init_variational_params, load_spectrum, random_hyperparameters
from cavi.priors import CHARGES, VariationalParams

# ---------------------------------------------------------------------------
# posterior predictive map
# ---------------------------------------------------------------------------

from tueplots import bundles, cycler, figsizes
from tueplots.constants.color import palettes

plt.rcParams.update(cycler.cycler(color=palettes.tue_plot))


class Window(NamedTuple):
    t_edges: np.ndarray  # (nt + 1,) bin edges in minutes
    y_edges: np.ndarray  # (ny + 1,) bin edges in m/z


class Frame(NamedTuple):
    mean: np.ndarray  # (nt, ny) expected ions per bin
    std: np.ndarray  # (nt, ny) posterior std of the expected ions per bin
    proj_t_mean: np.ndarray  # (nt,)
    proj_t_std: np.ndarray
    proj_y_mean: np.ndarray  # (ny,)
    proj_y_std: np.ndarray


def grid_edges(values: np.ndarray, lo: float, hi: float, group: int) -> np.ndarray:
    """Bin edges covering [lo, hi] that sit on the (regular) grid the data lives on.

    The data are stored on a regular grid (scans, m/z bins). Bins that cut across that grid
    would contain a varying number of grid cells and show a striping that is not in the model.
    `group` merges that many grid cells per bin.
    """
    u = np.unique(values)
    step = np.median(np.diff(u))
    origin = u[0] - step / 2
    size = step * group
    n_cells = int(np.ceil((u[-1] + step / 2 - origin) / size))
    i_lo = int(np.clip(np.floor((lo - origin) / size), 0, n_cells - 1))
    i_hi = int(np.clip(np.ceil((hi - origin) / size), i_lo + 1, n_cells))
    return origin + size * np.arange(i_lo, i_hi + 1)


def sigma_of(a, b):
    """Posterior mean of the standard deviation proxy sqrt(b / (a - 1)) of an inverse gamma."""
    return np.sqrt(b / np.maximum(a - 1.0, 1e-6))


def sample_posterior(v: VariationalParams, rng: np.random.Generator, S: int) -> dict:
    """Draw S parameter sets from q; q(c) and q(n) are kept analytic and marginalized later."""
    K = v.mu.shape[0]
    sigma2 = v.b_k / rng.gamma(v.a_k, size=(S, K))  # sigma_k^2 ~ IG(a, b)
    return dict(
        pi=rng.dirichlet(v.alpha, size=S),  # S K+1
        sigma_t=np.sqrt(sigma2),
        mu=v.mu + np.sqrt(sigma2 / v.nu) * rng.standard_normal((S, K)),
        m=v.m + np.sqrt(v.tau)[:, None] * rng.standard_normal((S, *v.m.shape)),  # S K C
        s=np.sqrt(v.b_s / rng.gamma(v.a_s, size=S)),
        p=rng.beta(v.a_p, v.b_p, size=S),
    )


def expected_map(
    v: VariationalParams,
    c_grid: np.ndarray,
    win: Window,
    n_total: float,
    t_size: float,
    y_size: float,
    num_j: int,
    rng: np.random.Generator,
    S: int,
) -> Frame:
    """Expected ion counts per (t, y) bin under q: mean and std over S posterior draws.

    A feature k contributes pi_k * N(t | mu_k, sigma_k^2) * sum_j P(j | k) sum_c q(c_k = c) N(y | m_kc + j Delta / c, s^2)
    with P(j | k) = sum_n q(n_k = n) Binom(j | n, p); the background is uniform over the whole window.
    """
    d = sample_posterior(v, rng, S)
    j = np.arange(num_j)
    nt, ny = len(win.t_edges) - 1, len(win.y_edges) - 1
    dt, dy = np.diff(win.t_edges), np.diff(win.y_edges)

    cdf_t = ndtr((win.t_edges[None, None] - d["mu"][..., None]) / d["sigma_t"][..., None])
    mass_t = np.diff(cdf_t, axis=-1)  # S K nt

    centers = d["m"][..., None] + j * DELTA_13C / np.asarray(CHARGES)[None, None, :, None]  # S K C J
    cdf_y = ndtr((win.y_edges - centers[..., None]) / d["s"][:, None, None, None, None])
    mass_y = np.einsum("kc,skcjy->skjy", v.q_c, np.diff(cdf_y, axis=-1))  # S K J ny

    pmf = binom.pmf(j[None, None, :], c_grid[None, :, None], d["p"][:, None, None])  # S G J
    iso = np.einsum("kg,sgj->skj", v.q_n, pmf)  # S K J

    feat = np.einsum("sk,skj,skt,skjy->sty", d["pi"][:, 1:], iso, mass_t, mass_y, optimize=True)
    bg = d["pi"][:, 0, None, None] * (dt[:, None] * dy[None, :])[None] / (t_size * y_size)
    maps = n_total * (feat + bg)  # S nt ny
    pt, py = maps.sum(2), maps.sum(1)
    return Frame(maps.mean(0), maps.std(0), pt.mean(0), pt.std(0), py.mean(0), py.std(0))


# ---------------------------------------------------------------------------
# video
# ---------------------------------------------------------------------------


def make_video(
    path: Path,
    win: Window,
    actual: np.ndarray,
    frames: list[Frame],
    labels: list[str],
    title: str,
    fps: int,
) -> Path:
    t_c = 0.5 * (win.t_edges[1:] + win.t_edges[:-1])
    y_c = 0.5 * (win.y_edges[1:] + win.y_edges[:-1])
    extent = [win.t_edges[0], win.t_edges[-1], win.y_edges[0], win.y_edges[-1]]
    log = lambda x: np.log10(1.0 + x)

    vmax = max(log(actual).max(), max(log(f.mean).max() for f in frames))
    std_max = max(log(f.std).max() for f in frames)
    act_t, act_y = actual.sum(1), actual.sum(0)

    plt.rcParams.update(bundles.icml2022(column="full", nrows=2, ncols=3, usetex=False))
    plt.rcParams["figure.figsize"] = (12.0, 7.0)
    fig = plt.figure()
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0])
    ax_act, ax_fit, ax_std = (fig.add_subplot(gs[0, i]) for i in range(3))
    ax_y, ax_t = fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[1, 2])

    def image(ax, data, vmax_, cmap, name):
        im = ax.imshow(data.T, origin="lower", extent=extent, aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax_)
        fig.colorbar(im, ax=ax, shrink=0.85, label=name)
        ax.set_xlabel(r"retention time $t$ (min)")
        return im

    im_act = image(ax_act, log(actual), vmax, "viridis", r"$\log_{10}(1+\mathrm{ions})$")
    im_fit = image(ax_fit, log(frames[0].mean), vmax, "viridis", r"$\log_{10}(1+\mathrm{ions})$")
    im_std = image(ax_std, log(frames[0].std), std_max, "magma", r"$\log_{10}(1+\mathrm{std})$")
    ax_act.set_ylabel(r"$m/z$")
    ax_act.set_title("actual")
    ax_fit.set_title("fitted (posterior mean)")
    ax_std.set_title("posterior uncertainty (std)")

    def curve(ax, x, edges, actual_, get_mean, get_std, xlabel):
        ax.stairs(actual_, edges, color="0.55", linewidth=0.8, label="actual")
        std_hi = max(get_std(f).max() for f in frames)
        norm = colors.Normalize(0.0, std_hi)
        lc = LineCollection([], cmap="magma", norm=norm, linewidth=2.2)
        ax.add_collection(lc)
        top = max(actual_.max(), max(get_mean(f).max() for f in frames) + std_hi)
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(0.0, 1.08 * top)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("ions per bin")
        return lc, norm

    lc_y, norm_y = curve(ax_y, y_c, win.y_edges, act_y, lambda f: f.proj_y_mean, lambda f: f.proj_y_std, r"$m/z$")
    lc_t, norm_t = curve(ax_t, t_c, win.t_edges, act_t, lambda f: f.proj_t_mean, lambda f: f.proj_t_std, r"retention time $t$ (min)")
    ax_y.set_title(r"projection on $m/z$: actual (grey) vs. fitted (colour = std)")
    ax_t.set_title(r"projection on $t$")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm_t, cmap="magma"), ax=ax_t, shrink=0.85, label="posterior std (ions per bin)")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm_y, cmap="magma"), ax=ax_y, shrink=0.85, label="posterior std (ions per bin)")
    suptitle = fig.suptitle(title)

    def segments(x, y):
        pts = np.stack([x, y], axis=1)
        return np.stack([pts[:-1], pts[1:]], axis=1)

    def update(i):
        f = frames[i]
        im_fit.set_data(log(f.mean).T)
        im_std.set_data(log(f.std).T)
        lc_y.set_segments(segments(y_c, f.proj_y_mean))
        lc_y.set_array(0.5 * (f.proj_y_std[:-1] + f.proj_y_std[1:]))
        lc_t.set_segments(segments(t_c, f.proj_t_mean))
        lc_t.set_array(0.5 * (f.proj_t_std[:-1] + f.proj_t_std[1:]))
        suptitle.set_text(f"{title}   |   {labels[i]}")
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    if animation.FFMpegWriter.isAvailable():
        writer, path = animation.FFMpegWriter(fps=fps, bitrate=4000), path.with_suffix(".mp4")
    else:
        writer, path = animation.PillowWriter(fps=fps), path.with_suffix(".gif")
    anim.save(path, writer=writer, dpi=130)
    plt.close(fig)
    return path


def make_video_3d(
    path: Path,
    win: Window,
    actual: np.ndarray,
    frames: list[Frame],
    labels: list[str],
    title: str,
    fps: int,
) -> Path:
    """3D surface of the fitted expected counts over (t, m/z), one frame per sweep.

    Height: posterior mean of the expected ions per bin. Colour: posterior std over the draws.
    The actual counts are a grey wireframe. The camera turns slowly so the shape is readable.
    """
    t_c = 0.5 * (win.t_edges[1:] + win.t_edges[:-1])
    y_c = 0.5 * (win.y_edges[1:] + win.y_edges[:-1])
    T, Y = np.meshgrid(t_c, y_c, indexing="ij")
    zmax = 1.05 * max(actual.max(), max(f.mean.max() for f in frames))
    norm = colors.Normalize(0.0, max(f.std.max() for f in frames))
    cmap = plt.get_cmap("YlOrRd")  # pale = certain, red = uncertain; stays visible on the grey panes

    plt.rcParams.update(bundles.icml2022(column="full", nrows=1, ncols=1, usetex=False))
    plt.rcParams["figure.figsize"] = (8.5, 6.5)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_wireframe(T, Y, actual, color="0.45", linewidth=0.35, alpha=0.7)
    ax.set_xlabel(r"retention time $t$ (min)")
    ax.set_ylabel(r"$m/z$")
    ax.set_zlabel("ions per bin")
    ax.set_zlim(0.0, zmax)
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.6, pad=0.1, label="posterior std (ions per bin)"
    )
    ax.plot([], [], color="0.45", label="actual")
    ax.legend(loc="upper left")
    suptitle = fig.suptitle(title)
    surf = [None]

    def update(i):
        if surf[0] is not None:
            surf[0].remove()
        f = frames[i]
        face = cmap(norm(f.std))
        face[..., 3] = 0.85  # slightly transparent, so the actual counts show through
        surf[0] = ax.plot_surface(
            T, Y, f.mean, facecolors=face, shade=False, linewidth=0, antialiased=False, rstride=1, cstride=1
        )
        ax.view_init(elev=28, azim=-75 + 45 * i / max(len(frames) - 1, 1))
        suptitle.set_text(f"{title}   |   {labels[i]}")
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    if animation.FFMpegWriter.isAvailable():
        writer, path = animation.FFMpegWriter(fps=fps, bitrate=4000), path.with_suffix(".mp4")
    else:
        writer, path = animation.PillowWriter(fps=fps), path.with_suffix(".gif")
    anim.save(path, writer=writer, dpi=120)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# static plots
# ---------------------------------------------------------------------------


def plot_elbo(elbos: np.ndarray, out: Path) -> Path:
    plt.rcParams.update(bundles.icml2022(column="full", nrows=1, ncols=2, usetex=False))
    fig, (ax, ax_d) = plt.subplots(1, 2)
    it = np.arange(1, len(elbos) + 1)
    ax.plot(it, elbos, "o-", markersize=3)
    ax.set_xlabel("CAVI sweep")
    ax.set_ylabel("ELBO")
    ax.set_title("ELBO")

    delta = np.diff(elbos)
    ax_d.semilogy(it[1:], np.maximum(np.abs(delta), 1e-12), "o-", markersize=3, label=r"$|\Delta\mathrm{ELBO}|$")
    dips = delta < 0
    if dips.any():  # float32 noise once converged; shown so it is not hidden
        ax_d.semilogy(it[1:][dips], np.abs(delta[dips]), "x", color="C3", label="decrease (numerical noise)")
    ax_d.set_xlabel("CAVI sweep")
    ax_d.set_ylabel(r"$|\Delta\mathrm{ELBO}|$")
    ax_d.set_title("convergence")
    ax_d.legend()
    path = out / "elbo.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


ALIVE = 30.0  # a component with more expected ions than this counts as supported by the data
TOL_MZ, TOL_RT = 0.03, 0.5  # Da, min: a component matches a reference feature within these


def reference_from_features(path: Path) -> dict:
    """OpenMS features as an answer key. They are only used to score the result, never as a prior."""
    import pyopenms as oms

    fm = oms.FeatureMap()
    oms.FeatureXMLFile().load(str(path), fm)
    return {
        "source": "featureXML",
        "features": [
            {"feature": i + 1, "rt_apex_min": f.getRT() / 60, "mono_mz": f.getMZ(), "charge": f.getCharge(),
             "intensity": f.getIntensity()}
            for i, f in enumerate(fm)
        ],
    }


def _load_reference(path: Path) -> dict:
    if path.suffix != ".json":
        return reference_from_features(path)
    ref = json.loads(path.read_text())
    for f in ref["features"]:  # a simulation's truth calls the intensity of a feature total_intensity
        f.setdefault("intensity", f.get("total_intensity"))
    return ref


def find_reference(spectrum: Path, given: Path | None) -> dict | None:
    """`--reference`, else the truth file of a simulation, else the OpenMS example's featureXML."""
    if given is not None:
        return _load_reference(given)
    candidates = [
        spectrum.with_name(f"{spectrum.stem}.truth.json"),
        spectrum.with_name(f"{spectrum.stem.replace('_input', '_1_output')}.featureXML"),
        spectrum.with_name(f"{spectrum.stem}.featureXML"),
    ]
    for c in candidates:
        if c.exists():
            return _load_reference(c)
    return None


def match_reference(v: VariationalParams, p, reference: dict | None) -> list[int | None]:
    """For each reference feature the supported component closest in m/z (within the tolerances)."""
    if not reference:
        return []
    n_k = np.asarray(v.alpha[1:] - np.asarray(p.alpha)[1:])
    alive = np.flatnonzero(n_k > ALIVE)
    best_c = np.argmax(np.asarray(v.q_c), axis=1)
    out = []
    for f in reference["features"]:
        match, dist = None, np.inf
        for k in alive:
            dm = abs(float(v.m[k, best_c[k]]) - f["mono_mz"])
            if dm < TOL_MZ and abs(float(v.mu[k]) - f["rt_apex_min"]) < TOL_RT and dm < dist:
                match, dist = int(k), dm
        out.append(match)
    return out


def zoom_limits(arrays: list[np.ndarray], burn: int, include: float | None = None, pad: float = 0.15) -> tuple[float, float]:
    """y-limits that fit the values from sweep `burn` on (axis 0 of every array are the sweeps).

    The first sweeps start from the prior and are typically far from where the values land, which
    would squeeze the converged part into a line; they are left out of the limits.
    `include` (e.g. 0) is kept inside the limits.
    """
    late = np.concatenate([np.asarray(a)[burn:].ravel() for a in arrays])
    lo, hi = float(np.min(late)), float(np.max(late))
    if include is not None:
        lo, hi = min(lo, include), max(hi, include)
    span = hi - lo if hi > lo else max(abs(hi), 1e-12) * 0.1
    return lo - pad * span, hi + pad * span


def zoom_axis(ax, arrays: list[np.ndarray], burn: int, include: float | None = None) -> None:
    """Zoom the y-axis on the converged part and say so if earlier sweeps are cut off."""
    lo, hi = zoom_limits(arrays, burn, include)
    ax.set_ylim(lo, hi)
    early = np.concatenate([np.asarray(a)[:burn].ravel() for a in arrays])
    if early.size and (early.min() < lo or early.max() > hi):
        ax.text(0.98, 0.03, f"sweeps 0-{burn - 1} partly off scale", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6.5, color="0.4")


def plot_metrics(states: list[VariationalParams], p, reference: dict | None, out: Path, top: int = 12) -> Path:
    K = states[0].mu.shape[0]
    it = np.arange(len(states))
    alpha0 = np.asarray(p.alpha)
    n_ions = np.array([s.alpha[1:] - alpha0[1:] for s in states])  # sweeps x K
    total = float(states[-1].alpha.sum() - alpha0.sum())  # all ions, background included
    final = states[-1]
    matches = match_reference(final, p, reference)
    shown = list(np.argsort(-n_ions[-1])[:top])
    cmap = plt.get_cmap("tab20")
    color = {k: cmap(i % 20) for i, k in enumerate(shown)}

    c_f = np.argmax(final.q_c, axis=1)
    m_dev = np.array([s.m[np.arange(K), c_f] - final.m[np.arange(K), c_f] for s in states])  # sweeps x K
    mu_all = np.array([s.mu for s in states])  # sweeps x K
    mu_sd = np.array([np.sqrt(s.b_k / (np.maximum(s.a_k - 1, 1e-6) * s.nu)) for s in states])
    m_sd = np.array([np.sqrt(s.tau) for s in states])
    sigma_mz = np.array([sigma_of(s.a_s, s.b_s) for s in states])
    p_mean = np.array([s.a_p / (s.a_p + s.b_p) for s in states])
    p_lo, p_hi = np.array([binom_ci(s.a_p, s.b_p) for s in states]).T

    plt.rcParams.update(bundles.icml2022(column="full", nrows=2, ncols=4, usetex=False))
    plt.rcParams["figure.figsize"] = (13.0, 6.5)
    fig, axs = plt.subplots(2, 4)
    ax_n, ax_alive, ax_mu, ax_m, ax_s, ax_p, ax_c, ax_ng = axs.ravel()

    for k in shown:
        ax_n.semilogy(it, np.maximum(n_ions[:, k], 1e-1), color=color[k])
        mu_k = mu_all[:, k]
        ax_mu.plot(it, mu_k, color=color[k])
        ax_mu.fill_between(it, mu_k - 2 * mu_sd[:, k], mu_k + 2 * mu_sd[:, k], color=color[k], alpha=0.2)
        ax_m.plot(it, m_dev[:, k], color=color[k])
        ax_m.fill_between(it, m_dev[:, k] - 2 * m_sd[:, k], m_dev[:, k] + 2 * m_sd[:, k], color=color[k], alpha=0.2)
    for r, k in zip(reference["features"] if reference else [], matches):
        if k in color:  # the answer key for the components that matched a reference feature
            ax_mu.axhline(r["rt_apex_min"], color=color[k], linestyle=":")
            ax_m.axhline(r["mono_mz"] - final.m[k, c_f[k]], color=color[k], linestyle=":")
    ax_n.axhline(ALIVE, color="k", linewidth=0.5, linestyle="--")
    ax_n.set_title(f"expected ions $N_k$ (top {len(shown)} components)")
    ax_mu.set_title(r"RT apex $\mu_k$ ($\pm 2$ sd)")
    ax_m.set_title(r"mono. $m/z$ minus its final value (Da, $\pm 2$ sd)")

    ax_alive.plot(it, (n_ions > ALIVE).sum(axis=1), color="C0")
    ax_alive.set_ylabel(f"components with $N_k > {ALIVE:.0f}$", color="C0")
    ax_frac = ax_alive.twinx()
    ax_frac.plot(it, n_ions.sum(axis=1) / total, color="C3")
    ax_frac.set_ylabel("share of ions in features", color="C3")
    ax_frac.set_ylim(0.0, 1.0)
    ax_alive.set_title("how many components the data support")

    ax_s.plot(it, sigma_mz)
    ax_s.set_title(r"$m/z$ peak width $\sqrt{b_s/(a_s-1)}$")
    ax_p.plot(it, p_mean)
    ax_p.fill_between(it, p_lo, p_hi, alpha=0.25)
    ax_p.set_title(r"$^{13}$C probability $p$ (90% interval)")
    burn = min(2, len(states) - 1)  # the prior and the first sweep do not set the scale of these panels
    zoom_axis(ax_m, [m_dev[:, shown] - 2 * m_sd[:, shown], m_dev[:, shown] + 2 * m_sd[:, shown]], burn, include=0.0)
    zoom_axis(ax_s, [sigma_mz], burn)
    zoom_axis(ax_p, [p_lo, p_hi, p_mean], burn)
    for ax in (ax_n, ax_alive, ax_mu, ax_m, ax_s, ax_p):
        ax.set_xlabel("CAVI sweep")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    rows = [str(k + 1) for k in shown]
    im = ax_c.imshow(final.q_c[shown], vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax_c.set_xticks(range(4), [str(int(c)) for c in CHARGES])
    ax_c.set_yticks(range(len(shown)), rows)
    ax_c.set_xlabel("charge $c$")
    ax_c.set_ylabel("component $k$")
    ax_c.set_title(r"final $q(c_k)$" + (" (x: reference)" if reference else ""))
    for r, k in zip(reference["features"] if reference else [], matches):
        if k in shown and 1 <= r["charge"] <= 4:
            ax_c.plot(r["charge"] - 1, shown.index(k), "x", color="C3")
    fig.colorbar(im, ax=ax_c, shrink=0.8)

    im = ax_ng.imshow(final.q_n[shown], vmin=0, vmax=final.q_n[shown].max(), cmap="Blues", aspect="auto")
    grid = np.asarray(p.c_grid)
    ticks = np.arange(0, len(grid), max(len(grid) // 6, 1))
    ax_ng.set_xticks(ticks, [str(int(grid[i])) for i in ticks])
    ax_ng.set_yticks(range(len(shown)), rows)
    ax_ng.set_xlabel("carbon count $n$")
    ax_ng.set_title(r"final $q(n_k)$")
    fig.colorbar(im, ax=ax_ng, shrink=0.8)

    if reference:
        found = sum(m is not None for m in matches)
        fig.suptitle(
            f"{found}/{len(matches)} reference features matched by a supported component "
            f"(m/z within {TOL_MZ} Da, RT within {TOL_RT} min)"
        )
    path = out / "metrics.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_intensity(
    result: dict, features: list[dict], total_intensity: float, out: Path
) -> tuple[Path, dict]:
    """Summed intensity of every reference feature: fitted (all its components) against the reference.

    `total_intensity` is the summed intensity of all pixels in the file: no pixel sum can exceed it,
    which bounds the ratio below when the reference intensity is a model estimate.
    """
    ref = np.array([f["intensity"] for f in features], dtype=float)
    fit_ = result["intensity"]
    stats = compare_intensities(fit_, ref)
    ceiling = total_intensity / ref.sum()
    order = np.argsort(-ref)
    ok = fit_ > 0
    missing = [int(i) + 1 for i in np.flatnonzero(~ok)]

    plt.rcParams.update(bundles.icml2022(column="full", nrows=1, ncols=2, usetex=False))
    plt.rcParams["figure.figsize"] = (11.0, 4.6)
    fig, (ax, ax_r) = plt.subplots(1, 2)

    lim = (0.5 * min(ref.min(), fit_[ok].min()), 2.0 * max(ref.max(), fit_.max()))
    xs = np.array(lim)
    ax.loglog(xs, xs, color="0.6", linestyle="--", label=r"$y = x$")
    ax.loglog(xs, stats["median_ratio"] * xs, color="C0", label=f"median ratio {stats['median_ratio']:.2f}")
    ax.loglog(xs, ceiling * xs, color="C3", linestyle=":", label=f"ceiling {ceiling:.2f}")
    sc = ax.scatter(ref[ok], fit_[ok], c=result["n_components"][ok], cmap="viridis", s=36, zorder=3)
    for i in np.flatnonzero(ok):
        ax.annotate(str(i + 1), (ref[i], fit_[i]), textcoords="offset points", xytext=(4, 3), fontsize=7)
    fig.colorbar(sc, ax=ax, shrink=0.85, label="components of the feature")
    ax.set_xlabel("reference intensity of the feature")
    ax.set_ylabel("fitted summed intensity")
    ax.set_title(
        rf"Pearson (log) $r = {stats.get('pearson_log', float('nan')):.2f}$, "
        rf"Spearman $\rho = {stats.get('spearman', float('nan')):.2f}$"
    )
    ax.legend(loc="upper left")

    x = np.arange(len(order))
    bars = ax_r.bar(x[ok[order]], stats["ratio"][order][ok[order]], color="C0")
    ax_r.axhline(stats["median_ratio"], color="C0", linestyle="-", linewidth=0.8)
    ax_r.axhline(ceiling, color="C3", linestyle=":")
    ax_r.axhline(1.0, color="0.6", linestyle="--")
    for xi, o in zip(x, order):
        if ok[o]:
            ax_r.text(xi, stats["ratio"][o], f"{result['n_components'][o]}", ha="center", va="bottom", fontsize=7)
        else:  # no component: the label sits at the bottom of the axes, a log axis has no zero
            ax_r.text(xi, 0.03, "none", ha="center", va="bottom", fontsize=7, transform=ax_r.get_xaxis_transform())
    ax_r.set_xlim(-0.6, len(order) - 0.4)
    ax_r.set_xticks(x, [str(o + 1) for o in order])
    ax_r.set_xlabel("reference feature (by decreasing intensity)")
    ax_r.set_ylabel("fitted / reference intensity")
    ax_r.set_yscale("log")
    ax_r.set_title("ratio per feature (numbers: components)")
    fig.suptitle(
        f"{int(ok.sum())}/{len(ref)} features have a component"
        + (f" (none for {missing})" if missing else "")
        + f"; the rest of the file: {result['other']:.0f} in other components, {result['background']:.0f} background"
    )
    path = out / "intensity.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path, stats


def binom_ci(a, b, level=0.9):
    from scipy.stats import beta

    return beta.ppf((1 - level) / 2, a, b), beta.ppf(1 - (1 - level) / 2, a, b)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

DEFAULT_SPECTRUM = ROOT / "data" / "FeatureFinderCentroided_1_input.mzML"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spectrum", type=Path,
                    default=DEFAULT_SPECTRUM if DEFAULT_SPECTRUM.exists() else ROOT / "data" / "synthetic.mzML")
    ap.add_argument("--out", type=Path, default=ROOT / "plots" / "cavi_real")
    ap.add_argument("--reference", type=Path, default=None,
                    help="truth.json or featureXML; only used to score the result (default: look next to the spectrum)")
    ap.add_argument("--num-features", type=int, default=30, help="number of random candidates")
    ap.add_argument("--init", choices=["pixels", "uniform"], default="pixels",
                    help="pixels: at pixels drawn proportionally to intensity; uniform: over the whole window")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-iter", type=int, default=40)
    ap.add_argument("--num-j", type=int, default=5, help="isotope peaks per feature")
    ap.add_argument("--samples", type=int, default=12, help="posterior draws for the uncertainty")
    ap.add_argument("--component", type=int, default=None, help="component for the zoomed videos (default: the largest)")
    ap.add_argument("--fps", type=int, default=3)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-3d", action="store_true", help="skip the 3D surface video of the zoom")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    t, y, w, scale = load_spectrum(str(args.spectrum), return_scale=True)
    cells = len(t) * args.num_features * args.num_j * 4  # the largest intermediate arrays
    if cells > 1.5e8:
        raise SystemExit(
            f"{len(t)} pixels x {args.num_features} candidates x {args.num_j} isotopes x 4 charges is too much "
            "memory; use fewer --num-features or crop the map (OpenMS FileFilter -rt/-mz)"
        )
    reference = find_reference(args.spectrum, args.reference)
    p = random_hyperparameters(t, y, w, num_features=args.num_features, seed=args.seed, init=args.init)
    print(f"{len(t)} pixels, {float(w.sum()):.0f} effective ions, {args.num_features} random candidates "
          f"({args.init}), reference: {'yes' if reference else 'no'}")

    v, elbos, history = fit(init_variational_params(p), p, t, y, w, args.num_j, args.num_iter)
    elbos = np.asarray(elbos)
    n_k = np.asarray(v.alpha[1:] - p.alpha[1:])
    print(f"ELBO {elbos[0]:.6g} -> {elbos[-1]:.6g} after {args.num_iter} sweeps; "
          f"{int((n_k > ALIVE).sum())} of {args.num_features} components supported, "
          f"peak width {float(sigma_of(v.a_s, v.b_s)):.4f} Da")
    matches = match_reference(v, p, reference)
    if reference:
        print(f"{sum(m is not None for m in matches)}/{len(matches)} reference features matched")

    v0 = jax.tree.map(np.asarray, init_variational_params(p))
    hist = jax.tree.map(np.asarray, history)
    states = [v0] + [VariationalParams(*(f[i] for f in hist)) for i in range(args.num_iter)]
    labels = ["random start (before the first sweep)"] + [f"sweep {i}, ELBO {e:.5g}" for i, e in enumerate(elbos, 1)]

    print("wrote", plot_elbo(elbos, args.out))
    print("wrote", plot_metrics(states, p, reference, args.out))
    if reference and all(f.get("intensity") for f in reference["features"]):
        res = feature_intensities(v, p, reference["features"], args.num_j, scale)
        path, stats = plot_intensity(res, reference["features"], float(w.sum()) * scale, args.out)
        print("wrote", path)
        print(f"intensity vs reference: median ratio {stats['median_ratio']:.2f}, Pearson (log) "
              f"{stats.get('pearson_log', float('nan')):.2f}, Spearman {stats.get('spearman', float('nan')):.2f} "
              f"over {stats['n_compared']}/{len(res['intensity'])} features")
    if args.no_video:
        return

    t_np, y_np, w_np = map(np.asarray, (t, y, w))
    t_lo, t_hi, y_lo, y_hi = t_np.min(), t_np.max(), y_np.min(), y_np.max()
    t_size, y_size = float(t_hi - t_lo), float(y_hi - y_lo)
    n_total = float(w_np.sum())
    c_grid = np.asarray(p.c_grid)

    def build(win: Window):
        frames = [
            expected_map(s, c_grid, win, n_total, t_size, y_size, args.num_j, np.random.default_rng(0), args.samples)
            for s in states
        ]
        actual, _, _ = np.histogram2d(t_np, y_np, bins=[win.t_edges, win.y_edges], weights=w_np)
        return actual, frames

    # whole spectrum: about 240 x 800 bins, on the grid of the data
    n_scans = len(np.unique(t_np))
    n_mz_cells = round((y_hi - y_lo) / np.median(np.diff(np.unique(y_np))))
    win = Window(
        grid_edges(t_np, t_lo, t_hi, max(1, round(n_scans / 240))),
        grid_edges(y_np, y_lo, y_hi, max(1, round(n_mz_cells / 800))),
    )
    actual, frames = build(win)
    print("wrote", make_video(args.out / "fit_full", win, actual, frames, labels, "whole spectrum", args.fps))

    # one component: window around its final posterior
    k = (args.component - 1) if args.component else int(np.argmax(n_k))
    c = int(np.argmax(states[-1].q_c[k]))
    mu_k, sd_k = states[-1].mu[k], sigma_of(states[-1].a_k[k], states[-1].b_k[k])
    m_k = states[-1].m[k, c]
    comb = (args.num_j - 1) * DELTA_13C / float(CHARGES[c])
    win_k = Window(
        grid_edges(t_np, mu_k - 4.0 * sd_k, mu_k + 4.0 * sd_k, 1),
        # bins ~0.005 Da wide: the m/z grid of real data is far finer than the peak width, and
        # bins narrower than that only show the discreteness of the centroids
        grid_edges(y_np, m_k - 0.4, m_k + comb + 0.4, max(1, round(0.005 / np.median(np.diff(np.unique(y_np)))))),
    )
    actual, frames = build(win_k)
    title = rf"component {k + 1} ($c={c + 1}$)"
    print("wrote", make_video(args.out / f"fit_component{k + 1}", win_k, actual, frames, labels, title, args.fps))
    if not args.no_3d:
        print("wrote", make_video_3d(args.out / f"fit_component{k + 1}_3d", win_k, actual, frames, labels, title, args.fps))


if __name__ == "__main__":
    main()
