from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
from beartype import beartype as typechecker
from jax.scipy.special import digamma, gammaln, logsumexp
from jaxtyping import Array, ArrayLike, Float, Int, PRNGKeyArray, jaxtyped

from cavi.priors import CHARGES, DELTA, HyperParameters, VariationalParams


def dirichlet(params: VariationalParams, k: Int[Array, ""]) -> Float[Array, ""]:
    return digamma(params.alpha[k]) - digamma(jnp.sum(params.alpha))


def log_binom(n, k):
    # log(n choose k) using gammaln (log-gamma function)
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def isotope_pattern(
    params: VariationalParams,
    h_params: HyperParameters,
    j: Int[Array, ""],
    k: Int[Array, ""],
) -> Float[Array, ""]:
    log_bin = log_binom(h_params.c_grid, j)
    e_n = params.q_n[k - 1] @ h_params.c_grid
    e_log_p = digamma(params.a_p) - digamma(params.a_p + params.b_p)
    e_log_1mp = digamma(params.b_p) - digamma(params.a_p + params.b_p)
    return params.q_n[k - 1] @ log_bin + j * e_log_p + (e_n - j) * e_log_1mp


def retention_time(
    params: VariationalParams, t: Float[Array, ""], k: Int[Array, ""]
) -> Float[Array, ""]:
    return (
        -1
        / 2
        * (
            jnp.log(2 * jnp.pi)
            + jnp.log(params.b_k[k-1])
            - digamma(params.a_k[k-1])
            + 1 / params.nu[k-1]
            + (params.a_k[k-1] / params.b_k[k-1]) * (t - params.mu[k-1]) ** 2
        )
    )


def m_z_signal(
    params: VariationalParams,
    y: Float[Array, ""],
    k: Int[Array, ""],
    j: Int[Array, ""],
) -> Float[Array, ""]:
    _lambda = params.a_s / params.b_s
    mass_steps = jnp.sum(
        params.q_c[k-1, :].T * (params.m[k-1, :] + (j * DELTA) / CHARGES - y)**2
    )
    return (
        -1
        / 2
        * (
            jnp.log(2 * jnp.pi)
            + jnp.log(params.b_s)
            - digamma(params.a_s)
            + _lambda * (params.tau[k - 1] + mass_steps)
        )
    )


def get_log_rho_0(params: VariationalParams, T: int, Y: int) -> Float[Array, ""]:
    return digamma(params.alpha[0]) - digamma(jnp.sum(params.alpha)) - jnp.log(T * Y)


def get_log_rho(
    params: VariationalParams,
    h_params: HyperParameters,
    t: Float[Array, ""],
    y: Float[Array, ""],
    k: Int[Array, ""],
    j: Int[Array, ""],
) -> Float[Array, ""]:
    return (
        dirichlet(params, k)
        + isotope_pattern(params, h_params, j, k)
        + retention_time(params, t, k)
        + m_z_signal(params, y, k, j)
    )


def get_log_rho_all(
    params: VariationalParams,
    h_params: HyperParameters,
    y: Float[Array, "N"],
    t: Float[Array, "N"],
    num_j: int,
    T: int,
    Y: int,
) -> tuple[Float[Array, "N K J"], Float[Array, ""]]:
    k = jnp.arange(1, params.m.shape[0] + 1)  # 0 is the background
    j = jnp.arange(num_j)

    log_rho_j = jax.vmap(get_log_rho, in_axes=(None, None, None, None, None, 0))
    log_rho_jk = jax.vmap(log_rho_j, in_axes=(None, None, None, None, 0, None))
    log_rho_jkn = jax.vmap(log_rho_jk, in_axes=(None, None, 0, 0, None, None))
    log_rho = log_rho_jkn(params, h_params, y, t, k, j)
    log_rho_0 = get_log_rho_0(params, T, Y)
    return log_rho, log_rho_0


def get_r(
    params: VariationalParams,
    h_params: HyperParameters,
    y: Float[Array, "N"],
    t: Float[Array, "N"],
    num_j: int,
    T: int,
    Y: int,
) -> tuple[Float[Array, "N K J"], Float[Array, "N"]]:
    log_rho, log_rho_0 = get_log_rho_all(params, h_params, y, t, num_j, T, Y)

    log_denom = jnp.logaddexp(
        log_rho_0, logsumexp(log_rho, axis=(1, 2))
    )  # N
    r = jnp.exp(log_rho - log_denom[:, None, None])
    r_0 = jnp.exp(log_rho_0 - log_denom)
    return r, r_0
