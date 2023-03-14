from typing import Callable

import jax
from jax import numpy as jnp, scipy as jsc
from jax.flatten_util import ravel_pytree

from jax.scipy.stats import multivariate_normal as mvn
from jaxopt import BacktrackingLineSearch

from optsmooth import MVNStandard, FunctionalModel


def log_posterior(
    states: jnp.ndarray,
    observations: jnp.ndarray,
    initial_dist: MVNStandard,
    transition_model: FunctionalModel,
    observation_model: FunctionalModel,
):
    x0 = states[0]
    xp, xn = states[:-1], states[1:]
    yn = observations

    m0, P0 = initial_dist
    f, (_, Q) = transition_model
    h, (_, R) = observation_model

    xn_mu = jax.vmap(f)(xp)
    yn_mu = jax.vmap(h)(xn)

    cost = - mvn.logpdf(x0, m0, P0)
    cost -= jnp.sum(jax.vmap(mvn.logpdf, in_axes=(0, 0, None))(xn, xn_mu, Q))
    cost -= jnp.sum(jax.vmap(mvn.logpdf, in_axes=(0, 0, None))(yn, yn_mu, R))
    return cost


def residual_vector(
    states: jnp.ndarray,
    observations: jnp.ndarray,
    initial_dist: MVNStandard,
    transition_model: FunctionalModel,
    observation_model: FunctionalModel,
):
    m0, P0 = initial_dist
    f = transition_model.function
    h = observation_model.function

    xp, xn = states[:-1], states[1:]
    yn = observations

    T, nx = states.shape
    _, ny = observations.shape
    n = nx + ny

    r0 = states[0] - m0
    rx = xn - jax.vmap(f)(xp)
    ry = yn - jax.vmap(h)(xn)

    rxy, _ = ravel_pytree(jnp.hstack((rx, ry)))
    r = jnp.hstack((r0, rxy))
    return r


def blk_diag_matrix(
    states: jnp.ndarray,
    observations: jnp.ndarray,
    initial_dist: MVNStandard,
    transition_model: FunctionalModel,
    observation_model: FunctionalModel,
):
    _, P0 = initial_dist
    _, Q = transition_model.mvn
    _, R = observation_model.mvn
    T, _ = states.shape

    QR = jsc.linalg.block_diag(jnp.linalg.inv(Q), jnp.linalg.inv(R))

    blk_QR = jnp.kron(jnp.eye(T - 1), QR)
    blk_QR = jsc.linalg.block_diag(jnp.linalg.inv(P0), blk_QR)
    return blk_QR


def line_search(x: jnp.ndarray,
                dx: jnp.ndarray,
                fun: Callable,
                maxiter: int = 100):

    ls = BacktrackingLineSearch(fun=fun, maxiter=maxiter,
                                condition="strong-wolfe")

    alpha, _ = ls.run(
        init_stepsize=1.0,
        params=x,
        descent_direction=dx,
    )
    xn = x + alpha * dx
    return xn
