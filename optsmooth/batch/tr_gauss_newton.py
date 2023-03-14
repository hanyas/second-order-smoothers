from typing import Callable

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from optsmooth.base import MVNStandard, FunctionalModel
from optsmooth.batch.utils import (
    log_posterior,
    residual_vector,
    blk_diag_matrix,
    trust_region,
)


def _gauss_newton_step(
    x: jnp.ndarray, lmbda: float, rsd: Callable, weight: jnp.ndarray
):
    r = rsd(x)
    J = jax.jacobian(rsd)(x)
    W = weight

    d = x.shape[0]
    grad = jnp.dot(J.T @ W, r)
    hess_reg = J.T @ W @ J + lmbda * jnp.eye(d)

    dx = -jnp.linalg.solve(hess_reg, grad)
    df = -jnp.dot(dx, grad) - 0.5 * jnp.dot(jnp.dot(dx, hess_reg), dx)
    return dx, df


def _trust_region_gauss_newton(
    x0: jnp.ndarray,
    fun: Callable,
    rsd: Callable,
    weight: jnp.ndarray,
    k: int,
    lmbda: float,
    nu: float,
):
    def body(carry, _):
        x, lmbda, nu = carry
        sub = lambda x, lmbda: _gauss_newton_step(x, lmbda, rsd, weight)
        xn, fn, lmbda, nu = trust_region(x, sub, fun, lmbda, nu)
        return (xn, lmbda, nu), fn

    (xn, _, _), fn = jax.lax.scan(body, (x0, lmbda, nu), jnp.arange(k))
    return xn, fn


def trust_region_iterated_batch_gauss_newton_smoother(
    init_nominal_mean: jnp.ndarray,
    observations: jnp.ndarray,
    initial_dist: MVNStandard,
    transition_model: FunctionalModel,
    observation_model: FunctionalModel,
    nb_iter: int = 10,
    lmbda: float = 1e2,
    nu: float = 2.0,
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

    flat_nominal_mean, costs = _trust_region_gauss_newton(
        x0=flat_init_nominal_mean,
        fun=_flat_log_posterior,
        rsd=_flat_residual_vector,
        weight=weight_matrix,
        k=nb_iter,
        lmbda=lmbda,
        nu=nu,
    )
    return _unflatten(flat_nominal_mean), jnp.hstack((init_cost, costs))
