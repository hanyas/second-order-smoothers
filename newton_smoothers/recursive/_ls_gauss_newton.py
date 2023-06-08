from typing import Callable

import jax
import jax.numpy as jnp

from jaxopt import BacktrackingLineSearch

from newton_smoothers.base import MVNStandard, FunctionalModel
from newton_smoothers.recursive.kalman import filtering, smoothing
from newton_smoothers.recursive.utils import (
    log_posterior_cost,
    linearize_state_space_model,
)


def line_search_iterated_recursive_gauss_newton_smoother(
    init_nominal: MVNStandard,
    observations: jnp.ndarray,
    init_dist: MVNStandard,
    transition_model: FunctionalModel,
    observation_model: FunctionalModel,
    linearization_method: Callable,
    nb_iter: int = 10,
):
    init_cost = log_posterior_cost(
        init_nominal.mean,
        observations,
        init_dist,
        transition_model,
        observation_model,
    )

    def _gauss_newton_step(nominal_trajectory):
        return _recursive_gauss_newton_step(
            observations,
            init_dist,
            transition_model,
            observation_model,
            linearization_method,
            nominal_trajectory,
        )

    def _log_posterior_cost(states):
        return log_posterior_cost(
            states,
            observations,
            init_dist,
            transition_model,
            observation_model,
        )

    def body(carry, _):
        nominal_trajectory = carry

        _smoothed_trajectory = _gauss_newton_step(nominal_trajectory)

        x0 = nominal_trajectory.mean
        dx = _smoothed_trajectory.mean - nominal_trajectory.mean

        ls = BacktrackingLineSearch(fun=_log_posterior_cost, maxiter=100)
        alpha, _ = ls.run(
            init_stepsize=1.0,
            params=x0,
            descent_direction=dx,
        )
        xn = x0 + alpha * dx

        _smoothed_trajectory = MVNStandard(
            mean=xn, cov=_smoothed_trajectory.cov
        )
        cost = _log_posterior_cost(_smoothed_trajectory.mean)
        return _smoothed_trajectory, cost

    smoothed_trajectory, costs = jax.lax.scan(
        body, init_nominal, jnp.arange(nb_iter)
    )

    return smoothed_trajectory, jnp.hstack((init_cost, costs))


def _recursive_gauss_newton_step(
    observations: jnp.ndarray,
    init_dist: MVNStandard,
    transition_model: FunctionalModel,
    observation_model: FunctionalModel,
    linearization_method: Callable,
    nominal_trajectory: MVNStandard,
):
    (
        linear_transition_model,
        linear_observation_model,
    ) = linearize_state_space_model(
        transition_model,
        observation_model,
        linearization_method,
        nominal_trajectory,
    )

    filtered_trajectory = filtering(
        observations,
        init_dist,
        linear_transition_model,
        linear_observation_model,
    )

    smoothed_trajectory = smoothing(
        linear_transition_model, filtered_trajectory
    )

    return smoothed_trajectory
