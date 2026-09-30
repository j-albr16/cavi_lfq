import jax
import jax.numpy as jnp
from jaxtyping import Float, ArrayLike, jaxtyped, Array, Int, PRNGKeyArray
from beartype import beartype as typechecker
from typing import NamedTuple
import jax.random as jr

CHARGES = jnp.array([1, 2, 3, 4], dtype=jnp.int32)  # doc: c_k in {1, 2, 3, 4}
DELTA = 1.0


class HyperParameters(NamedTuple):
    alpha: Float[Array, "K+1"] # account 1 for background
    m_0: Float[Array, "K"]
    nu_0: Float[Array, "K"]
    a_0: Float[Array, "K"]
    b_0: Float[Array, "K"]
    m_hat: Float[Array, "K"]
    tau: Float[Array, "K"]
    rho: Float[Array, "K 4"]
    n: Float[Array, "K G"]  # over k features and g possible C atom counts
    c_grid: Int[Array, "G"]
    a_p: Float[Array, ""]
    b_p: Float[Array, ""]
    a_s: Float[Array, ""]
    b_s: Float[Array, ""]

class VariationalParams(NamedTuple):
    alpha: Float[Array, "K"]
    a_p: Float[Array, ""]
    b_p: Float[Array, ""]
    a_s: Float[Array, ""]
    b_s: Float[Array, ""]
    q_n: Float[Array, "K G"]
    q_c: Float[Array, "K 4"]
    mu: Float[Array, "K"]
    nu: Float[Array, "K"]
    a_k: Float[Array, "K"]
    b_k: Float[Array, "K"]
    tau: Float[Array, "K"]
    m: Float[Array, "K 4"]

class Theta(NamedTuple):
    pi: Float[Array, "K+1"]      # mixing weights incl. background
    mu: Float[Array, "K"]        # RT apex
    sigma2: Float[Array, "K"]    # RT variance
    m: Float[Array, "K"]         # monoisotopic m/z
    c: Int[Array, "K"]           # charge
    n: Int[Array, "K"]           # carbon count
    p: Float[Array, ""]          # 13C probability
    s2: Float[Array, ""]         # m/z variance

@jaxtyped(typechecker=typechecker)
def sample_normal_inverse_gamma(
    key: PRNGKeyArray,
    m_0: Float[ArrayLike, "K"],
    nu_0: Float[ArrayLike, "K"],
    a_0: Float[ArrayLike, "K"],
    b_0: Float[ArrayLike, "K"],
) -> tuple[Float[Array, "K"], Float[Array, "K"]]:
    m_0 = jnp.asarray(m_0)
    nu_0 = jnp.asarray(nu_0)
    a_0 = jnp.asarray(a_0)
    b_0 = jnp.asarray(b_0)

    key_ig, key_n = jr.split(key)

    gamma_samples = jr.gamma(key_ig, a_0)
    inv_gamma_samples = b_0 / gamma_samples  # sigma^2 ~ IG(a_0, b_0)

    # mu | sigma^2 ~ N(m_0, sigma^2 / nu_0)
    normal_samples = jr.normal(key_n, shape=jnp.shape(m_0))
    mu_samples = m_0 + normal_samples * jnp.sqrt(inv_gamma_samples / nu_0)

    return jnp.array(mu_samples), jnp.array(inv_gamma_samples)


@jaxtyped(typechecker=typechecker)
def sample_theta(key: PRNGKeyArray, params: HyperParameters) -> Theta:
    key_mu, key_pi, key_m, key_c, key_n, key_p, key_s2 = jr.split(key, 7)

    mu, sigma2 = sample_normal_inverse_gamma(
        key_mu,
        params.m_0,
        params.nu_0,
        params.a_0,
        params.b_0,
    )

    s2 = params.b_s / jr.gamma(key_s2, params.a_s)  # s^2 ~ IG(a_s, b_s)

    return Theta(
        pi=jr.dirichlet(key_pi, params.alpha),
        mu=mu,
        sigma2=sigma2,
        m=jnp.array(params.m_hat) + jnp.array(params.tau) * jnp.array(jr.normal(key_m, shape=jnp.shape(params.m_hat))),
        c=CHARGES[jr.categorical(key_c, jnp.log(params.rho), axis=-1)],
        n=params.c_grid[jr.categorical(key_n, jnp.log(params.n), axis=-1)],
        p=jr.beta(key_p, params.a_p, params.b_p),
        s2=jnp.array(s2),
    )

@jaxtyped(typechecker=typechecker)
def sample_z_and_j(
    key: PRNGKeyArray, theta: Theta, n_pixels: int
) -> tuple[Int[Array, "n_pixels"], Int[Array, "n_pixels"]]:
    key_z, key_j = jr.split(key)

    # z_n ~ Cat(pi); z_n = 0 is background, z_n = k >= 1 is feature k
    z = jr.categorical(key_z, jnp.log(theta.pi), shape=(n_pixels,))

    # j_n | z_n = k >= 1 ~ Binomial(n_k, p); background pixels carry no isotope pattern
    k_idx = jnp.clip(z - 1, 0, theta.n.shape[0] - 1)
    n_k = theta.n[k_idx].astype(jnp.float32)
    j = jr.binomial(key_j, n_k, theta.p, shape=(n_pixels,)).astype(jnp.int32)
    j = jnp.where(z == 0, 0, j)

    return z, j
    







