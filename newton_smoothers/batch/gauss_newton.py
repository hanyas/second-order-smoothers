from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy as jsc

from jax.flatten_util import ravel_pytree

from newton_smoothers.base import MVNStandard, FunctionalModel
from newton_smoothers.batch.utils import (
    log_posterior_cost,
    residual_vector,
    block_diag_matrix,
)


def _gauss_newton_step(
    x: jnp.ndarray, residual: Callable, weights: jnp.ndarray
):
    r = residual(x)
    J = jax.jacobian(residual)(x)
    W = weights

    d = x.shape[0]
    grad = jnp.dot(J.T @ W, r)
    hess = J.T @ W @ J

    dx = -jsc.linalg.solve(hess, grad)
    df = -jnp.dot(dx, grad) - 0.5 * jnp.dot(jnp.dot(dx, hess), dx)
    return dx, df


@partial(jax.jit, static_argnums=(1, 2, 4))
def _gauss_newton(
    x0: jnp.ndarray,
    fun: Callable,
    residual: Callable,
    weights: jnp.ndarray,
    k: int,
):
    def body(carry, _):
        x = carry
        dx, df = _gauss_newton_step(x, residual, weights)
        return x + dx, fun(x + dx)

    xn, fn = jax.lax.scan(body, x0, jnp.arange(k))
    return xn, fn


def iterated_batch_gauss_newton_smoother(
    init_nominal: jnp.ndarray,
    observations: jnp.ndarray,
    init_dist: MVNStandard,
    transition_model: FunctionalModel,
    observation_model: FunctionalModel,
    nb_iter: int = 10,
):
    flat_init_nominal, _unflatten = ravel_pytree(init_nominal)

    def _flat_log_posterior_cost(flat_state):
        _state = _unflatten(flat_state)
        return log_posterior_cost(
            _state,
            observations,
            init_dist,
            transition_model,
            observation_model,
        )

    def _flat_residual_vector(flat_state):
        _state = _unflatten(flat_state)
        return residual_vector(
            _state,
            observations,
            init_dist,
            transition_model,
            observation_model,
        )

    weight_matrix = block_diag_matrix(
        init_nominal,
        observations,
        init_dist,
        transition_model,
        observation_model,
    )

    init_cost = _flat_log_posterior_cost(flat_init_nominal)

    flat_nominal, costs = _gauss_newton(
        x0=flat_init_nominal,
        fun=_flat_log_posterior_cost,
        residual=_flat_residual_vector,
        weights=weight_matrix,
        k=nb_iter,
    )
    return _unflatten(flat_nominal), jnp.hstack((init_cost, costs))
