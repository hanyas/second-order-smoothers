from typing import Callable

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from optsmooth.base import MVNStandard, FunctionalModel
from optsmooth.batch.utils import (
    log_posterior,
    residual_vector,
    blk_diag_matrix,
    line_search
)


def _bfgs_hess_update(B, s, y):
    # Implements a damped BFGS, see Nocedal et al.

    def accept(args):
        B, s, y = args
        return 1.0

    def reject(args):
        B, s, y = args
        return (
            0.8
            * jnp.dot(jnp.dot(s, B), s)
            / (jnp.dot(jnp.dot(s, B), s) - jnp.dot(s, y))
        )

    a = jax.lax.cond(
        jnp.dot(s, y) >= 0.2 * jnp.dot(jnp.dot(s, B), s),
        accept,
        reject,
        (B, s, y),
    )
    r = a * y + (1.0 - a) * B @ s
    return (
        B
        + jnp.outer(r, r) / jnp.dot(s, r)
        - jnp.outer(B @ s, B @ s) / jnp.dot(s, B @ s)
    )


def _bfgs_step(grad: jnp.ndarray, hess: jnp.ndarray):
    dx = -jnp.linalg.solve(hess, grad)
    return dx


def _line_search_bfgs(
    x0: jnp.ndarray,
    fun: Callable,
    gd: Callable,
    hs: jnp.ndarray,
    k: int,
):
    def body(carry, _):
        x, gd, hs = carry
        # approx. Newton step
        dx = _bfgs_step(gd, hs)
        # line search
        xn = line_search(x, dx, fun)
        # BFGS Hessian update
        gdn = jax.grad(fun)(xn)
        hsn = _bfgs_hess_update(hs, xn - x, gdn - gd)
        return (xn, gdn, hsn), fun(xn)

    (xn, _, _), fn = jax.lax.scan(body, (x0, gd, hs), jnp.arange(k))
    return xn, fn


def line_search_iterated_batch_bfgs_smoother(
    init_nominal_mean: jnp.ndarray,
    observations: jnp.ndarray,
    initial_dist: MVNStandard,
    transition_model: FunctionalModel,
    observation_model: FunctionalModel,
    nb_iter: int = 10,
):
    flat_init_nominal_mean, _unflatten = ravel_pytree(init_nominal_mean)

    def _flat_log_posterior(flat_state):
        _state = _unflatten(flat_state)
        return log_posterior(
            _state,
            observations,
            initial_dist,
            transition_model,
            observation_model,
        )

    def _flat_residual_vector(flat_state):
        _state = _unflatten(flat_state)
        return residual_vector(
            _state,
            observations,
            initial_dist,
            transition_model,
            observation_model,
        )

    weight_matrix = blk_diag_matrix(
        init_nominal_mean,
        observations,
        initial_dist,
        transition_model,
        observation_model,
    )

    init_cost = _flat_log_posterior(flat_init_nominal_mean)
    init_gard = jax.grad(_flat_log_posterior)(flat_init_nominal_mean)
    _jac = jax.jacobian(_flat_residual_vector)(flat_init_nominal_mean)
    init_hess = _jac.T @ weight_matrix @ _jac

    flat_nominal_mean, costs = _line_search_bfgs(
        x0=flat_init_nominal_mean,
        fun=_flat_log_posterior,
        gd=init_gard,
        hs=init_hess,
        k=nb_iter,
    )

    return _unflatten(flat_nominal_mean), jnp.hstack((init_cost, costs))
