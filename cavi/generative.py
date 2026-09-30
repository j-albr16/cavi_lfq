import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Float, Int, ArrayLike, Array, PRNGKeyArray, jaxtyped
from beartype import beartype as typechecker

from cavi.priors import Theta


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
