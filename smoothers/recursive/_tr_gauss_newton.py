from typing import Callable

import jax
import jax.numpy as jnp

from smoothers.base import MVNStandard, FunctionalModel
from smoothers.base import LinearTransition, LinearObservation
from smoothers.recursive.kalman import filtering, smoothing
from smoothers.recursive.utils import (
    log_posterior_cost,
    approx_log_posterior_cost,
)
from smoothers.recursive.utils import linearize_state_space_model


def trust_region_iterated_recursive_gauss_newton_smoother(
    init_nominal: MVNStandard,
    observations: jnp.ndarray,
    init_dist: MVNStandard,
    transition_model: FunctionalModel,
    observation_model: FunctionalModel,
    linearization_method: Callable,
    lmbda: float = 1e2,
    nu: float = 2.0,
    nb_iter: int = 10,
):
    init_cost = log_posterior_cost(
        init_nominal.mean,
        observations,
        init_dist,
        transition_model,
        observation_model,
    )

    def _gauss_newton_step(nominal_trajectory, lmbda):
        return _recursive_gauss_newton_step(
            observations,
            init_dist,
            transition_model,
            observation_model,
            linearization_method,
            nominal_trajectory,
            lmbda,
        )

    def body(carry, _):
        nominal_trajectory, lmbda, nu = carry

        _smoothed_trajectory, approx_cost_diff = _gauss_newton_step(
            nominal_trajectory, lmbda
        )

        nominal_cost = log_posterior_cost(
            nominal_trajectory.mean,
            observations,
            init_dist,
            transition_model,
            observation_model,
        )

        _smoothed_cost = log_posterior_cost(
            _smoothed_trajectory.mean,
            observations,
            init_dist,
            transition_model,
            observation_model,
        )

        true_cost_diff = nominal_cost - _smoothed_cost
        cost_ratio = true_cost_diff / approx_cost_diff

        def accept(args):
            lmbda, nu = args
            lmbda = lmbda * jnp.maximum(
                1.0 / 3.0, 1.0 - (2.0 * cost_ratio - 1) ** 3
            )
            lmbda = jnp.maximum(1e-16, lmbda)
            return _smoothed_trajectory, _smoothed_cost, lmbda, 2.0

        def reject(args):
            lmbda, nu = args
            lmbda = jnp.minimum(1e16, lmbda)
            return nominal_trajectory, nominal_cost, lmbda * nu, 2.0 * nu

        smoothed_trajectory, cost, lmbda, nu = jax.lax.cond(
            (cost_ratio > 0.0) & (approx_cost_diff > 0.0),
            accept,
            reject,
            operand=(lmbda, nu),
        )
        return (smoothed_trajectory, lmbda, nu), cost

    (smoothed_trajectory, _, _), costs = jax.lax.scan(
        body, (init_nominal, lmbda, nu), jnp.arange(nb_iter)
    )

    return smoothed_trajectory, jnp.hstack((init_cost, costs))


def _modified_state_space_model(
    observations: jnp.ndarray,
    init_dist: MVNStandard,
    linear_transtion_model: LinearTransition,
    linear_observation_model: LinearObservation,
    nominal_trajectory: MVNStandard,
    lmbda: float,
):
    m0, P0 = init_dist
    ys = observations

    Fs, bs, Qs = linear_transtion_model
    Hs, cs, Rs = linear_observation_model

    nx = Qs.shape[-1]
    ny = Rs.shape[-1]

    from jax.scipy.linalg import block_diag

    # first time step
    l0 = m0
    L0 = jnp.linalg.inv(jnp.linalg.inv(P0) + lmbda * jnp.eye(nx))

    # observed time steps
    def _modified_observation_model(H, c, R):
        pH = jnp.vstack((H, jnp.eye(nx)))
        pc = jnp.hstack((c, jnp.zeros((nx,))))
        pR = block_diag(R, 1.0 / lmbda * jnp.eye(nx))
        return pH, pc, pR

    pHs, pcs, pRs = jax.vmap(_modified_observation_model)(
        Hs,
        cs,
        Rs,
    )

    # modified observations
    next_nominal = nominal_trajectory.mean[1:]
    zs = jnp.hstack((ys, next_nominal))

    return (
        zs,
        MVNStandard(l0, L0),
        LinearTransition(Fs, bs, Qs),
        LinearObservation(pHs, pcs, pRs),
    )


def _recursive_gauss_newton_step(
    observations: jnp.ndarray,
    init_dist: MVNStandard,
    transition_model: FunctionalModel,
    observation_model: FunctionalModel,
    linearization_method: Callable,
    nominal_trajectory: MVNStandard,
    lmbda: float,
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

    (
        modified_observations,
        modified_init_dist,
        modified_transition_model,
        modified_observation_model,
    ) = _modified_state_space_model(
        observations,
        init_dist,
        linear_transition_model,
        linear_observation_model,
        nominal_trajectory,
        lmbda,
    )

    filtered_trajectory = filtering(
        modified_observations,
        modified_init_dist,
        modified_transition_model,
        modified_observation_model,
    )

    smoothed_trajectory = smoothing(
        modified_transition_model,
        filtered_trajectory,
    )

    approx_cost_diff = approx_log_posterior_cost(
        nominal_trajectory.mean,
        modified_observations,
        modified_init_dist,
        modified_transition_model,
        modified_observation_model,
    ) - approx_log_posterior_cost(
        smoothed_trajectory.mean,
        modified_observations,
        modified_init_dist,
        modified_transition_model,
        modified_observation_model,
    )

    return smoothed_trajectory, approx_cost_diff
