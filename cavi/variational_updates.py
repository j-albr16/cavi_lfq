import jax
import jax.numpy as jnp
from jax.scipy.special import digamma
from jaxtyping import Array, Float

from cavi.z_j import log_binom
from cavi.priors import CHARGES, DELTA, HyperParameters, VariationalParams


def pi(params: HyperParameters, N: Float[Array, "K+1"]) -> Float[Array, "K+1"]:
    return params.alpha + N


def nig(
    params: HyperParameters,
    R: Float[Array, "N K"],
    N: Float[Array, "K"],
    w: Float[Array, "N"],
    t: Float[Array, "N"],
) -> tuple[Float[Array, "K"], Float[Array, "K"], Float[Array, "K"], Float[Array, "K"]]:
    nu = params.nu_0 + N
    mu = (params.nu_0 * params.m_0 + jnp.sum(w[:, None] * R * t[:, None], axis=0)) / nu
    a = params.a_0 + N / 2
    b = params.b_0 + 0.5 * (
        params.nu_0 * (params.m_0 - mu) ** 2
        + jnp.sum(w[:, None] * R * (t[:, None] - mu[None, :]) ** 2, axis=0)
    )
    return nu, mu, a, b


def _isotope_residual(y: Float[Array, "N"], J: int) -> Float[Array, "N J C"]:
    j = jnp.arange(J)
    return y[:, None, None] - j[None, :, None] * DELTA / CHARGES[None, None, :]


def m_k(
    params: VariationalParams,
    h_params: HyperParameters,
    y: Float[Array, "N"],
    w: Float[Array, "N"],
    r: Float[Array, "N K J"],
    N: Float[Array, "K"],
) -> tuple[Float[Array, "K"], Float[Array, "K C"]]:
    """q(m_k | c_k = c) = Normal(m[k, c], tau2[k]) for every feature and charge."""
    _lambda = params.a_s / params.b_s
    u = _isotope_residual(y, r.shape[-1])  # N J C
    wr = w[:, None, None] * r  # N K J
    tau2 = 1 / (_lambda * N + 1 / h_params.tau**2)  # K
    m = tau2[:, None] * (
        _lambda * jnp.einsum("nkj,njc->kc", wr, u)
        + (h_params.m_hat / h_params.tau**2)[:, None]
    )
    return tau2, m


def q_c(
    params: VariationalParams,
    h_params: HyperParameters,
    y: Float[Array, "N"],
    w: Float[Array, "N"],
    r: Float[Array, "N K J"],
    m: Float[Array, "K C"],
) -> Float[Array, "K C"]:
    _lambda = params.a_s / params.b_s
    u = _isotope_residual(y, r.shape[-1])  # N J C
    wr = w[:, None, None] * r  # N K J
    res = u[:, None, :, :] - m[None, :, None, :]  # N K J C
    data = _lambda * jnp.einsum("nkj,nkjc->kc", wr, res**2)
    prior = (h_params.m_hat[:, None] - m) ** 2 / h_params.tau[:, None] ** 2
    ell = jnp.log(h_params.rho) - 0.5 * (data + prior)
    return jax.nn.softmax(ell, axis=-1)


def peak_width(
    params: VariationalParams,
    h_params: HyperParameters,
    N: Float[Array, "K"],
    w: Float[Array, "N"],
    r: Float[Array, "N K J"],
    qc: Float[Array, "K C"],
    y: Float[Array, "N"],
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    u = _isotope_residual(y, r.shape[-1])  # N J C
    res = params.m[None, :, None, :] - u[:, None, :, :]  # N K J C
    # params.tau holds the variance tau_tilde^2
    e = params.tau[None, :, None] + jnp.sum(qc[None, :, None, :] * res**2, axis=-1)
    a_s = h_params.a_s + 0.5 * jnp.sum(N)
    b_s = h_params.b_s + 0.5 * jnp.sum(w[:, None, None] * r * e)
    return a_s, b_s


def carbon_count(
    params: VariationalParams,
    h_params: HyperParameters,
    N: Float[Array, "K"],
    w: Float[Array, "N"],
    r: Float[Array, "N K J"],
) -> Float[Array, "K G"]:
    """q(n_k = C) on the grid h_params.c_grid, normalized per feature."""
    j = jnp.arange(r.shape[-1])
    C = h_params.c_grid  # G
    wr = w[:, None, None] * r  # N K J
    bin_sum = jnp.einsum("nkj,gj->kg", wr, log_binom(C[:, None], j[None, :]))
    e_log_1mp = digamma(params.b_p) - digamma(params.a_p + params.b_p)
    ell = jnp.log(h_params.n) + bin_sum + N[:, None] * C[None, :] * e_log_1mp
    return jax.nn.softmax(ell, axis=-1)


def heavy_isotope(
    h_params: HyperParameters,
    N: Float[Array, "K"],
    w: Float[Array, "N"],
    r: Float[Array, "N K J"],
    q_n: Float[Array, "K G"],
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    j = jnp.arange(r.shape[-1])
    S = jnp.sum(w[:, None, None] * r * j[None, None, :])
    a_p = h_params.a_p + S
    b_p = h_params.b_p + jnp.sum(N * (q_n @ h_params.c_grid)) - S
    return a_p, b_p
