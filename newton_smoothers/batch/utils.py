from typing import Callable

import jax
from jax import numpy as jnp, scipy as jsc
from jax.flatten_util import ravel_pytree

from jaxopt import BacktrackingLineSearch

from newton_smoothers.base import MVNStandard, FunctionalModel
from newton_smoothers.utils import weighted_sqr_dist


def log_posterior_cost(
    states: jnp.ndarray,
    observations: jnp.ndarray,
    init_dist: MVNStandard,
    transition_model: FunctionalModel,
    observation_model: FunctionalModel,
):
    x0 = states[0]
    xp, xn = states[:-1], states[1:]
    yn = observations

    m0, P0 = init_dist
    f, (_, Q) = transition_model
    h, (_, R) = observation_model

    xn_mu = jax.vmap(f)(xp)
    yn_mu = jax.vmap(h)(xn)

    cost = weighted_sqr_dist(x0, m0, P0)
    cost += jnp.sum(jax.vmap(weighted_sqr_dist, in_axes=(0, 0, None))(xn, xn_mu, Q))
    cost += jnp.sum(jax.vmap(weighted_sqr_dist, in_axes=(0, 0, None))(yn, yn_mu, R))
    return cost


def residual_vector(
    states: jnp.ndarray,
    observations: jnp.ndarray,
    init_dist: MVNStandard,
    transition_model: FunctionalModel,
    observation_model: FunctionalModel,
):
    m0, P0 = init_dist
    f = transition_model.function
    h = observation_model.function

    xp, xn = states[:-1], states[1:]
    yn = observations

    r0 = states[0] - m0
    rx = xn - jax.vmap(f)(xp)
    ry = yn - jax.vmap(h)(xn)

    rxy, _ = ravel_pytree(jnp.hstack((rx, ry)))
    r = jnp.hstack((r0, rxy))
    return r


def block_diag_matrix(
    states: jnp.ndarray,
    observations: jnp.ndarray,
    init_dist: MVNStandard,
    transition_model: FunctionalModel,
    observation_model: FunctionalModel,
):
    _, P0 = init_dist
    _, Q = transition_model.mvn
    _, R = observation_model.mvn
    T, _ = states.shape

    QR = jsc.linalg.block_diag(jnp.linalg.inv(Q), jnp.linalg.inv(R))

    blk_QR = jnp.kron(jnp.eye(T - 1), QR)
    blk_QR = jsc.linalg.block_diag(jnp.linalg.inv(P0), blk_QR)
    return blk_QR


def line_search_update(
    x: jnp.ndarray,
    dx: jnp.ndarray,
    fun: Callable,
    maxiter: int = 100
):
    ls = BacktrackingLineSearch(
        fun=fun,
        maxiter=maxiter,
        condition="strong-wolfe"
    )

    alpha, _ = ls.run(
        init_stepsize=1.0,
        params=x,
        descent_direction=dx,
    )
    xn = x + alpha * dx
    return xn


def trust_region_update(
    x: jnp.ndarray, sub: Callable, fun: Callable, lmbda: float, nu: float
):
    dx, df = sub(x, lmbda)
    xn = x + dx

    f, fn = fun(x), fun(xn)
    rho = (f - fn) / df

    def accept(args):
        lmbda, nu = args
        lmbda = lmbda * jnp.maximum(1.0 / 3.0, 1.0 - (2.0 * rho - 1) ** 3)
        lmbda = jnp.maximum(1e-16, lmbda)
        return xn, fn, lmbda, 2.0

    def reject(args):
        lmbda, nu = args
        lmbda = jnp.minimum(1e16, lmbda)
        return x, f, lmbda * nu, 2.0 * nu

    xn, fn, lmbda, nu = jax.lax.cond(
        (rho > 0.0) & (df > 0.0),
        accept,
        reject,
        operand=(lmbda, nu),
    )

    return xn, fn, lmbda, nu
