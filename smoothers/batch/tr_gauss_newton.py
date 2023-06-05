from typing import Callable

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from smoothers.base import MVNStandard, FunctionalModel
from smoothers.batch.utils import (
    log_posterior_cost,
    residual_vector,
    block_diag_matrix,
    trust_region_update,
)


def _gauss_newton_step(
    x: jnp.ndarray, lmbda: float, residual: Callable, weights: jnp.ndarray
):
    r = residual(x)
    J = jax.jacobian(residual)(x)
    W = weights

    d = x.shape[0]
    grad = jnp.dot(J.T @ W, r)
    hess_reg = J.T @ W @ J + lmbda * jnp.eye(d)

    dx = -jnp.linalg.solve(hess_reg, grad)
    df = -jnp.dot(dx, grad) - 0.5 * jnp.dot(jnp.dot(dx, hess_reg), dx)
    return dx, df


def _trust_region_gauss_newton(
    x0: jnp.ndarray,
    fun: Callable,
    residual: Callable,
    weights: jnp.ndarray,
    k: int,
    lmbda: float,
    nu: float,
):
    def body(carry, _):
        x, lmbda, nu = carry
        sub = lambda x, lmbda: _gauss_newton_step(x, lmbda, residual, weights)
        xn, fn, lmbda, nu = trust_region_update(x, sub, fun, lmbda, nu)
        return (xn, lmbda, nu), fn

    (xn, _, _), fn = jax.lax.scan(body, (x0, lmbda, nu), jnp.arange(k))
    return xn, fn


def trust_region_iterated_batch_gauss_newton_smoother(
    init_nominal: jnp.ndarray,
    observations: jnp.ndarray,
    init_dist: MVNStandard,
    transition_model: FunctionalModel,
    observation_model: FunctionalModel,
    nb_iter: int = 10,
    lmbda: float = 1e2,
    nu: float = 2.0,
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

    flat_nominal, costs = _trust_region_gauss_newton(
        x0=flat_init_nominal,
        fun=_flat_log_posterior_cost,
        residual=_flat_residual_vector,
        weights=weight_matrix,
        k=nb_iter,
        lmbda=lmbda,
        nu=nu,
    )
    return _unflatten(flat_nominal), jnp.hstack((init_cost, costs))
