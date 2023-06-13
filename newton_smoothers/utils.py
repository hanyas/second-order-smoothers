import jax
import jax.numpy as jnp
import jax.scipy as jsc


logdet = lambda x: jnp.linalg.slogdet(x)[1]


def weighted_sqr_dist(x, mu, cov):
    sqr_dist = 0.5 * jnp.dot(x - mu, jsc.linalg.solve(cov, x - mu))
    return sqr_dist


def mvn_logpdf(x, mu, cov):
    d = cov.shape[0]
    dist = -0.5 * jnp.dot(x - mu, jsc.linalg.solve(cov, x - mu))
    norm = 0.5 * logdet(cov) + 0.5 * d * jnp.log(2.0 * jnp.pi)
    return dist - norm


def mvn_logpdf_chol(x, mu, chol_cov):
    d = chol_cov.shape[0]
    y = jsc.linalg.solve_triangular(chol_cov, (x - mu), lower=True)
    dist = -0.5 * jnp.sum(y * y, axis=-1)
    norm = (
        jnp.sum(jnp.log(jnp.diag(chol_cov))) + 0.5 * d * jnp.log(2 * jnp.pi)
    )
    return dist - norm


def none_or_shift(x, shift):
    if x is None:
        return None
    if shift > 0:
        return jax.tree_map(lambda z: z[shift:], x)
    return jax.tree_map(lambda z: z[:shift], x)


def none_or_concat(x, y, position=1):
    if x is None or y is None:
        return None
    if position == 1:
        return jax.tree_map(
            lambda a, b: jnp.concatenate([a[None, ...], b]), y, x
        )
    else:
        return jax.tree_map(
            lambda a, b: jnp.concatenate([b, a[None, ...]]), y, x
        )
