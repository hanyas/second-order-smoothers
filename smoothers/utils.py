from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsc

from jax import custom_vjp, vjp
from jax.custom_derivatives import closure_convert
from jax.flatten_util import ravel_pytree
from jax.lax import while_loop


logdet = lambda x: jnp.linalg.slogdet(x)[1]


def mvn_logpdf(x, mu, cov):
    dim = cov.shape[0]
    norm = jnp.dot(x - mu, jsc.linalg.solve(cov, x - mu))
    normalizer = logdet(cov) / 2.0 + dim * jnp.log(2.0 * jnp.pi) / 2.0
    return - 0.5 * norm - normalizer


def mvn_logpdf_chol(x, mu, chol_cov):
    dim = chol_cov.shape[0]
    y = jsc.linalg.solve_triangular(chol_cov, (x - mu), lower=True)
    norm = jnp.sum(y * y, axis=-1)
    normalizer = jnp.sum(jnp.log(jnp.diag(chol_cov)))\
                 + dim * jnp.log(2 * jnp.pi) / 2.0
    return -0.5 * norm - normalizer


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
        return jax.tree_map(lambda a, b: jnp.concatenate([a[None, ...], b]), y, x)
    else:
        return jax.tree_map(lambda a, b: jnp.concatenate([b, a[None, ...]]), y, x)


# FIXED POINT UTIL

def fixed_point(f, x0, criterion):
    converted_fn, aux_args = closure_convert(f, x0)
    return _fixed_point(converted_fn, aux_args, x0, criterion)


@partial(custom_vjp, nondiff_argnums=(0, 3))
def _fixed_point(f, params, x0, criterion):
    return __fixed_point(f, params, x0, criterion)[0]


def _fixed_point_fwd(f, params, x0, criterion):
    x_star, n_iter = __fixed_point(f, params, x0, criterion)
    return x_star, (params, x_star, n_iter)


def _fixed_point_rev(f, _criterion, res, x_star_bar):
    params, x_star, n_iter = res
    _, vjp_theta = vjp(lambda p: f(x_star, *p), params)
    theta_bar, = vjp_theta(__fixed_point(partial(_rev_iter, f),
                                         (params, x_star, x_star_bar),
                                         x_star_bar,
                                         lambda i, *_: i < n_iter + 1)[0])
    return theta_bar, jax.tree_map(jnp.zeros_like, x_star)


def _rev_iter(f, u, *packed):
    params, x_star, x_star_bar = packed
    _, vjp_x = vjp(lambda x: f(x, *params), x_star)
    ravelled_x_star_bar, unravel_fn = ravel_pytree(x_star_bar)
    ravelled_vjp_x_u, _ = ravel_pytree(vjp_x(u)[0])
    return unravel_fn(ravelled_x_star_bar + ravelled_vjp_x_u)


def __fixed_point(f, params, x0, criterion):
    def cond_fun(carry):
        i, x_prev, x = carry
        return criterion(i, x_prev, x)

    def body_fun(carry):
        i, _, x = carry
        return i + 1, x, f(x, *params)

    n_iter, _, x_star = while_loop(cond_fun, body_fun, (1, x0, f(x0, *params)))
    return x_star, n_iter


_fixed_point.defvjp(_fixed_point_fwd, _fixed_point_rev)
