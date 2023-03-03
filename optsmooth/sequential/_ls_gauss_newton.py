from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy as jsc

from jaxopt import BacktrackingLineSearch

from optsmooth._base import MVNStandard, FunctionalModel
from optsmooth._base import LinearTransition, LinearObservation
from optsmooth._utils import mvn_logpdf, none_or_shift, none_or_concat
from optsmooth.sequential._smoothing import smoothing


def log_posterior(
    states: jnp.ndarray,
    observations: jnp.ndarray,
    initial_dist: MVNStandard,
    transition_model: FunctionalModel,
    observation_model: FunctionalModel,
):
    xp, xn = states[:-1], states[1:]
    yn = observations

    m0, P0 = initial_dist
    f, (_, Q) = transition_model
    h, (_, R) = observation_model

    xn_mu = jax.vmap(f)(xp)
    yn_mu = jax.vmap(h)(xn)

    cost = -mvn_logpdf(states[0], m0, P0)
    cost -= jnp.sum(jax.vmap(mvn_logpdf, in_axes=(0, 0, None))(xn, xn_mu, Q))
    cost -= jnp.sum(jax.vmap(mvn_logpdf, in_axes=(0, 0, None))(yn, yn_mu, R))
    return cost


def line_search_iterated_recursive_gauss_newton_smoother(
    observations: jnp.ndarray,
    initial_dist: MVNStandard,
    transition_model: FunctionalModel,
    observation_model: FunctionalModel,
    linearization_method: Callable,
    quadratization_method: Callable,
    init_nominal_traj: MVNStandard,
    nb_iter: int = 10,
):
    init_cost = log_posterior(
        init_nominal_traj.mean,
        observations,
        initial_dist,
        transition_model,
        observation_model,
    )

    def _gauss_newton_step(nominal_traj):
        return _recursive_gauss_newton_step(
            nominal_traj,
            observations,
            initial_dist,
            transition_model,
            observation_model,
            linearization_method,
            quadratization_method,
        )

    def _log_posterior(states):
        return log_posterior(
            states,
            observations,
            initial_dist,
            transition_model,
            observation_model,
        )

    def body(carry, _):
        nominal_traj = carry

        smoothed_traj = _gauss_newton_step(nominal_traj)

        x0 = nominal_traj.mean
        dx = smoothed_traj.mean - nominal_traj.mean

        ls = BacktrackingLineSearch(fun=_log_posterior, maxiter=100)
        alpha, _ = ls.run(
            init_stepsize=1.0,
            params=x0,
            descent_direction=dx,
        )
        xn = x0 + alpha * dx

        smoothed_traj = MVNStandard(mean=xn, cov=smoothed_traj.cov)
        cost = _log_posterior(smoothed_traj.mean)
        return smoothed_traj, cost

    nominal_traj, costs = jax.lax.scan(
        body, init_nominal_traj, jnp.arange(nb_iter)
    )

    return nominal_traj, jnp.hstack((init_cost, costs))


def _filtering(
    observations: jnp.ndarray,
    initial_dist: MVNStandard,
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
        G = jnp.linalg.solve(S.T, H @ P.T).T

        m = m + G @ (y - (H @ m + c))
        P = P - G @ S @ G.T
        return MVNStandard(m, P)

    def body(carry, args):
        xf = carry
        y, F, b, Q, H, c, R = args

        xp = _predict(F, b, Q, xf)
        xf = _update(H, c, R, xp, y)
        return xf, xf

    xf0 = initial_dist
    y = observations

    F, b, Q = linear_transition
    H, c, R = linear_observation

    _, Xfs = jax.lax.scan(body, xf0, (y, F, b, Q, H, c, R))
    return none_or_concat(Xfs, xf0, 1)


def _build_gauss_newton_state_space(
    transition_model: FunctionalModel,
    observation_model: FunctionalModel,
    linearization_method: Callable,
    nominal_trajectory: MVNStandard,
):
    curr_nominal = none_or_shift(nominal_trajectory, -1)
    next_nominal = none_or_shift(nominal_trajectory, 1)

    F, b, Q = jax.vmap(linearization_method, in_axes=(None, 0))(
        transition_model, curr_nominal
    )
    H, c, R = jax.vmap(linearization_method, in_axes=(None, 0))(
        observation_model, next_nominal
    )

    return LinearTransition(F, b, Q), LinearObservation(H, c, R)


def _approx_log_posterior(
    states: jnp.ndarray,
    observations: jnp.ndarray,
    initial_dist: MVNStandard,
    transition_model: LinearTransition,
    observation_model: LinearObservation,
):
    xp, xn = states[:-1], states[1:]
    yn = observations

    m0, P0 = initial_dist
    F_x, b, Q = transition_model
    H_x, c, R = observation_model

    xn_mu = jnp.einsum("nij,nj->ni", F_x, xp) + b
    yn_mu = jnp.einsum("nij,nj->ni", H_x, xn) + c

    cost = -mvn_logpdf(states[0], m0, P0)
    cost -= jnp.sum(jax.vmap(mvn_logpdf)(xn, xn_mu, Q))
    cost -= jnp.sum(jax.vmap(mvn_logpdf)(yn, yn_mu, R))
    return cost


def _recursive_gauss_newton_step(
    nominal_traj: MVNStandard,
    observations: jnp.ndarray,
    initial_dist: MVNStandard,
    transition_model: FunctionalModel,
    observation_model: FunctionalModel,
    linearization_method: Callable,
    quadratization_method: Callable,
):
    lin_trns_mdl, lin_obs_mdl = _build_gauss_newton_state_space(
        transition_model, observation_model, linearization_method, nominal_traj
    )

    filtered_traj = _filtering(
        observations, initial_dist, lin_trns_mdl, lin_obs_mdl
    )

    smoothed_traj = smoothing(
        transition_model,
        filtered_traj,
        linearization_method,
        nominal_traj,
    )

    return smoothed_traj
