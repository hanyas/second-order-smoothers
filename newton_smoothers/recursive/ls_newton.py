from typing import Callable

import jax
import jax.numpy as jnp

from jaxopt import BacktrackingLineSearch

from newton_smoothers.base import MVNStandard, FunctionalModel
from newton_smoothers.base import QuadraticTransition, QuadraticObservation
from newton_smoothers.base import LinearTransition, LinearObservation
from newton_smoothers.recursive.kalman import filtering, smoothing
from newton_smoothers.recursive.utils import (
    log_posterior_cost,
    approx_log_posterior_cost,
)
from newton_smoothers.recursive.utils import quadratize_state_space_model


def line_search_iterated_recursive_newton_smoother(
    init_nominal: MVNStandard,
    observations: jnp.ndarray,
    init_dist: MVNStandard,
    transition_model: FunctionalModel,
    observation_model: FunctionalModel,
    quadratization_method: Callable,
    nb_iter: int = 10,
):
    init_cost = log_posterior_cost(
        init_nominal.mean,
        observations,
        init_dist,
        transition_model,
        observation_model,
    )

    def _newton_step(nominal_trajectory):
        return _recursive_newton_step(
            observations,
            init_dist,
            transition_model,
            observation_model,
            quadratization_method,
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

        _smoothed_trajectory = _newton_step(nominal_trajectory)

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


def _modified_state_space_model(
    observations: jnp.ndarray,
    init_dist: MVNStandard,
    quadratic_transition_model: QuadraticTransition,
    quadratic_observation_model: QuadraticObservation,
    nominal_trajectory: MVNStandard,
    lmbda: float,
):
    m0, P0 = init_dist

    ys = observations
    x0 = nominal_trajectory.mean[0]
    xp = nominal_trajectory.mean[:-1]
    xn = nominal_trajectory.mean[1:]

    F_xx, F_x, f0, Q = quadratic_transition_model
    H_xx, H_x, h0, R = quadratic_observation_model

    nx = Q.shape[-1]
    ny = R.shape[-1]

    from jax.scipy.linalg import solve

    def _transition_params(F_xx, F_x, f0, Q, xn, xp):
        F = F_x
        b = f0 - F_x @ xp
        Psi = -jnp.einsum("ijk,k->ij", F_xx.T, solve(Q, xn - f0))
        return F, b, Psi

    def _observation_params(H_xx, H_x, h0, R, yn, xn):
        H = H_x
        c = h0 - H_x @ xn
        Gamma = -jnp.einsum("ijk,k->ij", H_xx.T, solve(R, yn - h0))
        return H, c, Gamma

    F, b, Psi = jax.vmap(_transition_params)(F_xx, F_x, f0, Q, xn, xp)
    H, c, Gamma = jax.vmap(_observation_params)(H_xx, H_x, h0, R, ys, xn)

    from jax.scipy.linalg import block_diag

    # first time step
    _Phi0 = Psi[0] + lmbda * jnp.eye(nx)
    L0 = jnp.linalg.inv(jnp.linalg.inv(P0) + _Phi0)
    l0 = L0 @ (jnp.linalg.inv(P0) @ m0 + _Phi0 @ x0)

    # observed time steps
    def _modified_observation_model(H, c, R, Psi, Gamma):
        mH = jnp.vstack((H, jnp.eye(nx)))
        mc = jnp.hstack((c, jnp.zeros((nx,))))
        mR = block_diag(R, jnp.linalg.inv(Psi + Gamma + lmbda * jnp.eye(nx)))
        return mH, mc, mR

    _Psi = jnp.stack((*Psi[1:], jnp.zeros((nx, nx))))
    mH, mc, mR = jax.vmap(_modified_observation_model)(H, c, R, _Psi, Gamma)

    # modified observations
    zs = jnp.hstack((ys, xn))

    return (
        zs,
        MVNStandard(l0, L0),
        LinearTransition(F, b, Q),
        LinearObservation(mH, mc, mR),
    )


def _regularized_recursive_newton_step(
    observations: jnp.ndarray,
    init_dist: MVNStandard,
    transition_model: FunctionalModel,
    observation_model: FunctionalModel,
    quadratization_method: Callable,
    nominal_trajectory: MVNStandard,
    lmbda: float,
):
    (
        quadratic_transition_model,
        quadratic_observation_model,
    ) = quadratize_state_space_model(
        transition_model,
        observation_model,
        quadratization_method,
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
        quadratic_transition_model,
        quadratic_observation_model,
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


def _recursive_newton_step(
    observations: jnp.ndarray,
    init_dist: MVNStandard,
    transition_model: FunctionalModel,
    observation_model: FunctionalModel,
    quadratization_method: Callable,
    nominal_trajectory: MVNStandard,
):

    def _step_func(lmbda):
        return _regularized_recursive_newton_step(
            observations,
            init_dist,
            transition_model,
            observation_model,
            quadratization_method,
            nominal_trajectory,
            lmbda,
        )

    def _modify_direction():
        def cond(carry):
            lmbda = carry
            _, df = _step_func(lmbda)
            return df <= 0.0

        def body(carry):
            lmbda = carry
            lmbda = lmbda * 10.0
            return lmbda

        lmbda = jax.lax.while_loop(
            cond,
            body,
            1e-8,
        )
        return _step_func(lmbda)[0]

    # try with no regularization first
    _smoothed_trajectory, approx_cost_diff = _step_func(1e-32)
    # assert approx_cost_diff != jnp.nan

    # modify direciton if necessary
    smoothed_trajectory = jax.lax.cond(
        approx_cost_diff > 0.0,
        lambda: _smoothed_trajectory,
        _modify_direction,
    )

    return smoothed_trajectory
