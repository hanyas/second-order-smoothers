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


def _gauss_newton_step(
    x: jnp.ndarray, rsd: Callable, weight: jnp.ndarray
):
    r = rsd(x)
    J = jax.jacobian(rsd)(x)
    W = weight

    dx = -jnp.linalg.solve(J.T @ W @ J, jnp.dot(J.T @ W, r))
    return dx


def _line_search_gauss_newton(
    x0: jnp.ndarray, fun: Callable, rsd: Callable, weight: jnp.ndarray, k: int
):
    def body(carry, _):
        x = carry
        dx = _gauss_newton_step(x, rsd, weight)
        xn = line_search(x, dx, fun)
        return xn, fun(xn)

    xn, fn = jax.lax.scan(body, x0, jnp.arange(k))
    return xn, fn


def line_search_iterated_batch_gauss_newton_smoother(
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

    flat_nominal_mean, costs = _line_search_gauss_newton(
        x0=flat_init_nominal_mean,
        fun=_flat_log_posterior,
        rsd=_flat_residual_vector,
        weight=weight_matrix,
        k=nb_iter,
    )

    return _unflatten(flat_nominal_mean), jnp.hstack((init_cost, costs))
