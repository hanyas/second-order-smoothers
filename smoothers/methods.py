from typing import Callable, Optional, Union

import jax
from jax import numpy as jnp

from smoothers.utils import fixed_point
from smoothers.base import MVNStandard, FunctionalModel
from smoothers.recursive._filtering import filtering as seq_filtering
from smoothers.recursive.utils import smoothing as seq_smoothing


def filtering(
    observations: jnp.ndarray,
    init_dist: MVNStandard,
    transition_model: FunctionalModel,
    observation_model: FunctionalModel,
    linearization_method: Callable,
    nominal_trajectory: Optional[MVNStandard] = None,
    return_loglikelihood: bool = False,
):
    return seq_filtering(
        observations,
        init_dist,
        transition_model,
        observation_model,
        linearization_method,
        nominal_trajectory,
        return_loglikelihood,
    )


def smoothing(
    transition_model: FunctionalModel,
    filter_trajectory: MVNStandard,
    linearization_method: Callable,
    nominal_trajectory: Optional[MVNStandard] = None,
):
    return seq_smoothing(
        transition_model,
        filter_trajectory,
        linearization_method,
        nominal_trajectory,
    )


def filter_smoother(
    observations: jnp.ndarray,
    init_dist: MVNStandard,
    transition_model: FunctionalModel,
    observation_model: FunctionalModel,
    linearization_method: Callable,
    nominal_trajectory: Optional[MVNStandard] = None,
):
    filter_trajectory = filtering(
        observations,
        init_dist,
        transition_model,
        observation_model,
        linearization_method,
        nominal_trajectory,
    )
    return smoothing(
        transition_model,
        filter_trajectory,
        linearization_method,
        nominal_trajectory,
    )


def _default_criterion(_i, prev_nominal_traj, curr_nominal_traj):
    return (
        jnp.mean((prev_nominal_traj.mean - curr_nominal_traj.mean) ** 2) > 1e-6
    )


def iterated_smoothing(
    observations: jnp.ndarray,
    init_dist: MVNStandard,
    transition_model: FunctionalModel,
    observation_model: FunctionalModel,
    linearization_method: Callable,
    init_nominal_trajectory: Optional[MVNStandard] = None,
    criterion: Callable = _default_criterion,
    return_loglikelihood: bool = False,
):
    if init_nominal_trajectory is None:
        init_nominal_trajectory = filter_smoother(
            observations,
            init_dist,
            transition_model,
            observation_model,
            linearization_method,
            None,
        )

    def fun_to_iter(curr_nominal_traj):
        return filter_smoother(
            observations,
            init_dist,
            transition_model,
            observation_model,
            linearization_method,
            curr_nominal_traj,
        )

    nominal_traj = fixed_point(fun_to_iter, init_nominal_trajectory, criterion)
    if return_loglikelihood:
        _, ell = filtering(
            observations,
            init_dist,
            transition_model,
            observation_model,
            linearization_method,
            nominal_traj,
            return_loglikelihood=True,
        )
        return nominal_traj, ell
    return nominal_traj
