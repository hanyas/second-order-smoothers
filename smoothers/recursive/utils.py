from typing import Callable

import jax
from jax import numpy as jnp
from jax import scipy as jsc
from jax.scipy.stats import multivariate_normal as mvn

from smoothers import MVNStandard, FunctionalModel
from smoothers.base import LinearTransition, LinearObservation
from smoothers.utils import none_or_concat, none_or_shift


def filtering(
    observations: jnp.ndarray,
    init_dist: MVNStandard,
    linear_transition: LinearTransition,
    linear_observation: LinearObservation,
):
    def _predict(F, b, Q, x):
        m, P = x

        m = F @ m + b
        P = Q + F @ P @ F.T
        return MVNStandard(m, P)

    def _update(H, c, R, x, y):
        m, P = x

        S = R + H @ P @ H.T
        chol_S = jnp.linalg.cholesky(S)
        G = P @ jsc.linalg.cho_solve((chol_S, True), H).T

        y_hat = H @ m + c
        y_diff = y - y_hat
        m = m + G @ y_diff
        P = P - G @ S @ G.T
        return MVNStandard(m, P)

    def body(carry, args):
        xf = carry
        y, lt, lo = args

        F, b, Q = lt
        H, c, R = lo

        xp = _predict(F, b, Q, xf)
        xf = _update(H, c, R, xp, y)
        return xf, xf

    x0 = init_dist
    ys = observations

    lts = linear_transition
    los = linear_observation

    _, Xf = jax.lax.scan(body, x0, (ys, lts, los))
    return none_or_concat(Xf, x0, 1)


def smoothing(
    linear_transition: LinearTransition,
    filter_trajectory: MVNStandard,
):
    def _smooth(F, b, Q, xf, xs):
        mf, Pf = xf
        ms, Ps = xs

        mean_diff = ms - (b + F @ mf)
        S = F @ Pf @ F.T + Q
        cov_diff = Ps - S

        gain = Pf @ jnp.linalg.solve(S, F).T
        ms = mf + gain @ mean_diff
        Ps = Pf + gain @ cov_diff @ gain.T
        return MVNStandard(ms, Ps)

    def body(carry, args):
        xs = carry
        xf, lt = args

        F, b, Q = lt
        xs = _smooth(F, b, Q, xf, xs)
        return xs, xs

    xl = jax.tree_map(lambda z: z[-1], filter_trajectory)

    Xf = none_or_shift(filter_trajectory, -1)
    lts = linear_transition

    _, Xs = jax.lax.scan(
        body,
        xl,
        (Xf, lts),
        reverse=True,
    )

    Xs = none_or_concat(Xs, xl, -1)
    return Xs


def log_posterior_cost(
    states: jnp.ndarray,
    observations: jnp.ndarray,
    init_dist: MVNStandard,
    transition_model: FunctionalModel,
    observation_model: FunctionalModel,
):
    xp, xn = states[:-1], states[1:]
    yn = observations

    m0, P0 = init_dist
    f, (_, Q) = transition_model
    h, (_, R) = observation_model

    xn_mu = jax.vmap(f)(xp)
    yn_mu = jax.vmap(h)(xn)

    cost = -mvn.logpdf(states[0], m0, P0)
    cost -= jnp.sum(jax.vmap(mvn.logpdf, in_axes=(0, 0, None))(xn, xn_mu, Q))
    cost -= jnp.sum(jax.vmap(mvn.logpdf, in_axes=(0, 0, None))(yn, yn_mu, R))
    return cost


def approx_log_posterior_cost(
    states: jnp.ndarray,
    observations: jnp.ndarray,
    init_dist: MVNStandard,
    linear_transition: LinearTransition,
    linear_observation: LinearObservation,
):
    xp, xn = states[:-1], states[1:]
    yn = observations

    m0, P0 = init_dist
    F_x, b, Q = linear_transition
    H_x, c, R = linear_observation

    xn_mu = jnp.einsum("nij,nj->ni", F_x, xp) + b
    yn_mu = jnp.einsum("nij,nj->ni", H_x, xn) + c

    cost = -mvn.logpdf(states[0], m0, P0)
    cost -= jnp.sum(jax.vmap(mvn.logpdf)(xn, xn_mu, Q))
    cost -= jnp.sum(jax.vmap(mvn.logpdf)(yn, yn_mu, R))
    return cost


def linearize_state_space_model(
    transition_model: FunctionalModel,
    observation_model: FunctionalModel,
    linearization_method: Callable,
    nominal_trajectory: MVNStandard,
):
    curr_nominal = none_or_shift(nominal_trajectory, -1)
    next_nominal = none_or_shift(nominal_trajectory, 1)

    Fs, bs, Qs = jax.vmap(linearization_method, in_axes=(None, 0))(
        transition_model, curr_nominal
    )
    Hs, cs, Rs = jax.vmap(linearization_method, in_axes=(None, 0))(
        observation_model, next_nominal
    )

    return LinearTransition(Fs, bs, Qs), LinearObservation(Hs, cs, Rs)
