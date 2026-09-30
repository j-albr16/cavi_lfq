import jax.numpy as jnp
from jax.scipy.special import betaln, digamma, gammaln, xlogy
from jaxtyping import Array, Float

from cavi.priors import HyperParameters, VariationalParams


# ---------------------------------------------------------------------------
# KL building blocks (elementwise / batched over leading axes)
# ---------------------------------------------------------------------------


def kl_categorical(q: Float[Array, "... G"], p: Float[Array, "... G"]) -> Float[Array, "..."]:
    """KL(Cat(q) || Cat(p)) over the last axis."""
    return jnp.sum(xlogy(q, q) - xlogy(q, p), axis=-1)


def kl_dirichlet(alpha_q: Float[Array, "D"], alpha_p: Float[Array, "D"]) -> Float[Array, ""]:
    """KL(Dir(alpha_q) || Dir(alpha_p))."""
    return (
        gammaln(jnp.sum(alpha_q))
        - jnp.sum(gammaln(alpha_q))
        - gammaln(jnp.sum(alpha_p))
        + jnp.sum(gammaln(alpha_p))
        + jnp.sum((alpha_q - alpha_p) * (digamma(alpha_q) - digamma(jnp.sum(alpha_q))))
    )


def kl_normal(m_q, v_q, m_p, v_p):
    """KL(N(m_q, v_q) || N(m_p, v_p)) with variances v."""
    return 0.5 * (jnp.log(v_p / v_q) + (v_q + (m_q - m_p) ** 2) / v_p - 1.0)


def kl_beta(a_q, b_q, a_p, b_p):
    """KL(Beta(a_q, b_q) || Beta(a_p, b_p))."""
    return (
        betaln(a_p, b_p)
        - betaln(a_q, b_q)
        + (a_q - a_p) * digamma(a_q)
        + (b_q - b_p) * digamma(b_q)
        + (a_p - a_q + b_p - b_q) * digamma(a_q + b_q)
    )


def kl_inverse_gamma(a_q, b_q, a_p, b_p):
    """KL(IG(a_q, b_q) || IG(a_p, b_p)), shape a and scale b.

    Invariant under the bijection x -> 1/x, so it equals the Gamma(shape, rate) KL.
    """
    return (
        (a_q - a_p) * digamma(a_q)
        - gammaln(a_q)
        + gammaln(a_p)
        + a_p * (jnp.log(b_q) - jnp.log(b_p))
        + a_q * (b_p - b_q) / b_q
    )


# ---------------------------------------------------------------------------
# KL per latent distribution: D_KL(q(theta_i) || p(theta_i))
# ---------------------------------------------------------------------------


def kl_pi(params: VariationalParams, h_params: HyperParameters) -> Float[Array, ""]:
    """q(pi) = Dir(alpha_tilde) against the prior Dir(alpha), background included."""
    return kl_dirichlet(params.alpha, h_params.alpha)


def kl_nig(params: VariationalParams, h_params: HyperParameters) -> Float[Array, "K"]:
    """q(mu_k, sigma_k^2) = NIG(mu, nu, a, b) against NIG(m_0, nu_0, a_0, b_0), per feature.

    mu | sigma^2 ~ N(m, sigma^2 / nu), sigma^2 ~ IG(a, b). Chain rule:
    KL = KL(IG || IG) + E_q(sigma^2)[KL(N(mu, sigma^2/nu) || N(m_0, sigma^2/nu_0))],
    with E[1 / sigma^2] = a / b.
    """
    kl_sigma = kl_inverse_gamma(params.a_k, params.b_k, h_params.a_0, h_params.b_0)
    kl_mu = 0.5 * (
        jnp.log(params.nu / h_params.nu_0)
        + h_params.nu_0 / params.nu
        - 1.0
        + h_params.nu_0 * (params.mu - h_params.m_0) ** 2 * params.a_k / params.b_k
    )
    return kl_sigma + kl_mu


def kl_mass_charge(params: VariationalParams, h_params: HyperParameters) -> Float[Array, "K"]:
    """q(m_k, c_k) = q(c_k) N(m_kc, tau2_k) against p(c_k) p(m_k) = Cat(rho_k) N(m_hat_k, tau_k^2).

    params.tau holds the variational variance tau2_k, h_params.tau the prior std.
    """
    kl_c = kl_categorical(params.q_c, h_params.rho)  # K
    kl_m = kl_normal(
        params.m, params.tau[:, None], h_params.m_hat[:, None], h_params.tau[:, None] ** 2
    )  # K C
    return kl_c + jnp.sum(params.q_c * kl_m, axis=-1)


def kl_carbon_count(params: VariationalParams, h_params: HyperParameters) -> Float[Array, "K"]:
    """q(n_k) on the grid against the empirical prior table p(n_k | m_hat_k)."""
    return kl_categorical(params.q_n, h_params.n)


def kl_heavy_isotope(params: VariationalParams, h_params: HyperParameters) -> Float[Array, ""]:
    """q(p) = Beta(a_p, b_p) against Beta(a_p, b_p) prior."""
    return kl_beta(params.a_p, params.b_p, h_params.a_p, h_params.b_p)


def kl_peak_width(params: VariationalParams, h_params: HyperParameters) -> Float[Array, ""]:
    """q(s^2) = IG(a_s, b_s) against the prior IG(a_s, b_s)."""
    return kl_inverse_gamma(params.a_s, params.b_s, h_params.a_s, h_params.b_s)


def kl_total(params: VariationalParams, h_params: HyperParameters) -> Float[Array, ""]:
    return (
        kl_pi(params, h_params)
        + jnp.sum(kl_nig(params, h_params))
        + jnp.sum(kl_mass_charge(params, h_params))
        + jnp.sum(kl_carbon_count(params, h_params))
        + kl_heavy_isotope(params, h_params)
        + kl_peak_width(params, h_params)
    )


# ---------------------------------------------------------------------------
# ELBO
# ---------------------------------------------------------------------------


def elbo(
    params: VariationalParams,
    h_params: HyperParameters,
    w: Float[Array, "N"],
    r: Float[Array, "N K J"],
    r_0: Float[Array, "N"],
    log_rho: Float[Array, "N K J"],
    log_rho_0: Float[Array, ""],
) -> Float[Array, ""]:
    """L = sum_n w_n [sum_kj r (log rho - log r) + r_0 (log rho_0 - log r_0)] - sum_i KL(q_i || p_i).

    `log_rho` / `log_rho_0` must be evaluated at the current variational parameters
    (see `cavi.z_j.get_log_rho_all`). The pixel weights w_n enter like in the updates:
    pixel n counts w_n times.
    """
    per_pixel = jnp.sum(r * log_rho - xlogy(r, r), axis=(1, 2)) + (
        r_0 * log_rho_0 - xlogy(r_0, r_0)
    )
    return jnp.sum(w * per_pixel) - kl_total(params, h_params)
