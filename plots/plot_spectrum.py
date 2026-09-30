import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tueplots import bundles

from cavi.priors import HyperParameters, sample_theta, sample_z_and_j
from cavi.generative import generative_model


def build_params() -> HyperParameters:
    K, G = 4, 5
    c_grid = jnp.array([30, 40, 50, 60, 70], dtype=jnp.int32)

    # carbon-count prior per feature, peaked near a different grid slot each
    n_probs = jnp.eye(K, G) * 0.6 + jnp.full((K, G), 0.4 / G)
    n_probs = n_probs / n_probs.sum(axis=-1, keepdims=True)

    alpha = jnp.full((K + 1,), 2.0)
    alpha = alpha.at[0].set(2000.)
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


def main() -> None:
    key = jax.random.PRNGKey(4)
    key_theta, key_zj, key_gen = jax.random.split(key, 3)

    params = build_params()
    theta = sample_theta(key_theta, params)

    n_ions = 5_000_000
    z, j = sample_z_and_j(key_zj, theta, n_ions)

    t_range = jnp.array([0.0, 60.0])
    y_range = jnp.array([400.0, 1200.0])
    delta = 1.00336

    t, y = generative_model(key_gen, theta, z, j, t_range, y_range, delta)

    # one high-resolution histogram; every panel below is a crop or a block-sum
    # downsample of THIS same array, never a re-histogram / re-sample
    t_bins_fine, y_bins_fine = 600, 16_000  # 0.1 min/bin, 0.05 Da/bin
    counts_fine, t_edges_fine, y_edges_fine = jnp.histogram2d(
        t, y, bins=[t_bins_fine, y_bins_fine],
        range=[t_range.tolist(), y_range.tolist()],
    )
    t_centers_fine = 0.5 * (t_edges_fine[:-1] + t_edges_fine[1:])
    y_centers_fine = 0.5 * (y_edges_fine[:-1] + y_edges_fine[1:])

    def crop(t_win, y_win):
        ti = jnp.where((t_centers_fine >= t_win[0]) & (t_centers_fine <= t_win[1]))[0]
        yi = jnp.where((y_centers_fine >= y_win[0]) & (y_centers_fine <= y_win[1]))[0]
        sub = counts_fine[ti[0]:ti[-1] + 1, yi[0]:yi[-1] + 1]
        tc = t_centers_fine[ti[0]:ti[-1] + 1]
        yc = y_centers_fine[yi[0]:yi[-1] + 1]
        TT, YY = jnp.meshgrid(tc, yc, indexing="ij")
        return sub, TT, YY

    # overview: block-sum the fine grid down to a renderable size (exact same
    # total counts, just coarser bins) instead of a separate low-res histogram
    t_factor, y_factor = 4, 80
    counts_overview = counts_fine.reshape(
        t_bins_fine // t_factor, t_factor, y_bins_fine // y_factor, y_factor
    ).sum(axis=(1, 3))
    t_centers_overview = t_centers_fine.reshape(-1, t_factor).mean(axis=1)
    y_centers_overview = y_centers_fine.reshape(-1, y_factor).mean(axis=1)
    T_ov, Y_ov = jnp.meshgrid(t_centers_overview, y_centers_overview, indexing="ij")

    # pick the feature with the most ions to zoom into
    feature_counts = jnp.array([(z == k).sum() for k in range(1, theta.mu.shape[0] + 1)])
    zoom_k = int(jnp.argmax(feature_counts)) + 1
    idx = zoom_k - 1
    mu_z, sigma_z = theta.mu[idx], jnp.sqrt(theta.sigma2[idx])
    m_z, c_z = theta.m[idx], theta.c[idx]
    spacing = delta / c_z

    counts_med, T_med, Y_med = crop(
        (mu_z - 6 * sigma_z, mu_z + 6 * sigma_z),
        (m_z - 3 * spacing, m_z + 12 * spacing),
    )
    counts_tight, T_tight, Y_tight = crop(
        (mu_z - 3 * sigma_z, mu_z + 3 * sigma_z),
        (m_z - spacing, m_z + 4 * spacing),
    )

    panels = [
        (T_ov, Y_ov, counts_overview,
         rf"Full map ($K={theta.mu.shape[0]}$ features, $N={n_ions}$ ions, "
         rf"{t_factor * 0.1:.2g} min $\times$ {y_factor * 0.05:.2g} Da bins)"),
        (T_med, Y_med, counts_med,
         rf"Feature {zoom_k} neighborhood (0.1 min $\times$ 0.05 Da bins)"),
        (T_tight, Y_tight, counts_tight,
         rf"Isotope comb of feature {zoom_k} ($c={int(c_z)}$, "
         rf"$\Delta/c\approx{float(spacing):.3f}$ Da)"),
    ]

    plt.rcParams.update(bundles.icml2022(column="full", nrows=1, ncols=3))
    plt.rcParams["figure.figsize"] = (16.0, 5.5)

    fig = plt.figure()
    for i, (X, Y_, Z, title) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        surf = ax.plot_surface(
            X, Y_, Z,
            cmap="viridis",
            linewidth=0,
            antialiased=True,
            rstride=1,
            cstride=1,
        )
        ax.set_xlabel(r"retention time $t$ (min)")
        ax.set_ylabel(r"$m/z$")
        ax.set_zlabel("ion count")
        ax.set_title(title)
        ax.view_init(elev=35, azim=-60)
        fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12, label="ion count")

    out_path = Path(__file__).resolve().parent / "spectrum_3d.png"
    fig.savefig(out_path, dpi=200)
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
