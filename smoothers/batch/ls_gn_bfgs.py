from typing import Callable

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from smoothers.base import MVNStandard, FunctionalModel
from smoothers.batch.utils import (
    log_posterior_cost,
    residual_vector,
    block_diag_matrix,
    line_search_update,
)


def _gn_bfgs_hess_update(S, s, yd, yc):
    aux = (
        jnp.outer(yc - S @ s, yd) / jnp.dot(yd, s)
        + jnp.outer(yd, yc - S @ s) / jnp.dot(yd, s)
        - jnp.outer(yc - S @ s, s) @ jnp.outer(yd, yd) / jnp.dot(yd, s) ** 2
    )

    tau = jnp.minimum(
        1.0, jnp.abs(jnp.dot(s, yc)) / jnp.abs(jnp.dot(s, jnp.dot(S, s)))
    )
    return tau * S + aux


def _gn_bfgs_step(
    x: jnp.ndarray,
    r: jnp.ndarray,
    J: jnp.ndarray,
    W: jnp.ndarray,
    bfgs_hess: jnp.ndarray,
):
    grad = jnp.dot(J.T @ W, r)
    hess = J.T @ W @ J + bfgs_hess
    dx = -jnp.linalg.solve(hess, grad)
    return dx


def _line_search_gn_bfgs(
    x0: jnp.ndarray,
    fun: Callable,
    residual: Callable,
    weights: jnp.ndarray,
    k: int,
):
    W = weights

    def body(carry, _):
        x, rp, Jp, bfgs_hess = carry

        dx = _gn_bfgs_step(x, rp, Jp, W, bfgs_hess)
        xn = line_search_update(x, dx, fun)

        # GN-BFGS hessian update
        rn = residual(xn)
        Jn = jax.jacobian(residual)(xn)

        yd = jnp.dot(Jn.T @ W, rn) - jnp.dot(Jp.T @ W, rp)
        yc = jnp.dot(Jn.T @ W, rn) - jnp.dot(Jp.T @ W, rn)

        bfgs_hess = _gn_bfgs_hess_update(bfgs_hess, xn - x, yd, yc)
        return (xn, rn, Jn, bfgs_hess), fun(xn)

    rp = residual(x0)
    Jp = jax.jacobian(residual)(x0)
    bfgs_hess = 1e-6 * jnp.eye(x0.shape[0])

    (xn, _, _, _), fn = jax.lax.scan(
        body, (x0, rp, Jp, bfgs_hess), jnp.arange(k)
    )
    return xn, fn


def line_search_iterated_batch_gn_bfgs_smoother(
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

    flat_nominal, costs = _line_search_gn_bfgs(
        x0=flat_init_nominal,
        fun=_flat_log_posterior_cost,
        residual=_flat_residual_vector,
        weights=weight_matrix,
        k=nb_iter,
    )

    return _unflatten(flat_nominal), jnp.hstack((init_cost, costs))
