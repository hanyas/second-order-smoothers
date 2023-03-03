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


def line_search_iterated_recursive_newton_smoother(
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

    def _newton_step(nominal_traj):
        return _recursive_newton_step(
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

        smoothed_traj = _newton_step(nominal_traj)

        x0 = nominal_traj.mean
        dx = smoothed_traj.mean - nominal_traj.mean

        ls = BacktrackingLineSearch(fun=_log_posterior, maxiter=100)
        alpha, _ = ls.run(
            init_stepsize=1.0,
            params=x0,
            descent_direction=dx,
        )

        smoothed_traj = MVNStandard(mean=x0 + alpha * dx, cov=smoothed_traj.cov)
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


def _build_pseudo_hessians(
    observations: jnp.ndarray,
    transition_model: FunctionalModel,
    observation_model: FunctionalModel,
    quadratization_method: Callable,
    nominal_traj: MVNStandard,
):
    y = observations
    f = transition_model.function
    h = observation_model.function

    curr_nominal = none_or_shift(nominal_traj, -1)
    next_nominal = none_or_shift(nominal_traj, 1)

    F_xx, F_x, b, Q = jax.vmap(quadratization_method, in_axes=(None, 0))(
        transition_model, curr_nominal
    )
    H_xx, H_x, c, R = jax.vmap(quadratization_method, in_axes=(None, 0))(
        observation_model, next_nominal
    )

    from jax.scipy.linalg import solve

    def _trns_hess_fcn(F_xx, Q, m_next, m_curr):
        return -jnp.einsum("ijk,k->ij", F_xx.T, solve(Q, m_next - f(m_curr)))

    def _obs_hess_fcn(H_xx, R, y, m_next):
        return -jnp.einsum("ijk,k->ij", H_xx.T, solve(R, y - h(m_next)))

    m_curr = curr_nominal.mean
    m_next = next_nominal.mean

    trns_hess = jax.vmap(_trns_hess_fcn)(F_xx, Q, m_next, m_curr)
    obs_hess = jax.vmap(_obs_hess_fcn)(H_xx, R, y, m_next)

    return (
        LinearTransition(F_x, b, Q),
        LinearObservation(H_x, c, R),
        trns_hess,
        obs_hess,
    )


def _build_newton_state_space(
    observations: jnp.ndarray,
    initial_dist: MVNStandard,
    transition_model: LinearTransition,
    observation_model: LinearObservation,
    transition_hess: jnp.ndarray,
    observation_hess: jnp.ndarray,
    nominal_traj: MVNStandard,
    lmbda: float,
):
    m0, P0 = initial_dist
    y = observations

    F_x, b, Q = transition_model
    H_x, c, R = observation_model

    Phi = transition_hess
    Gamma = observation_hess

    nx = Q.shape[-1]
    ny = R.shape[-1]

    from jax.scipy.linalg import block_diag, inv

    # first time step
    l0 = m0
    L0 = jnp.linalg.inv(jnp.linalg.inv(P0) + Phi[0] + lmbda * jnp.eye(nx))

    # intermediate time steps
    def _build_observation_model(H_x, c, R, Phi, Gamma):
        G_x = jnp.vstack((H_x, jnp.eye(nx)))
        q = jnp.hstack((c, jnp.zeros((nx,))))
        W = block_diag(R, inv(Phi + Gamma + lmbda * jnp.eye(nx)))
        return G_x, q, W

    G_x, q, W = jax.vmap(_build_observation_model)(
        H_x[:-1], c[:-1], R[:-1], Phi[1:], Gamma[:-1]
    )

    # final time step
    _G_x = jnp.vstack((H_x[-1], jnp.eye(nx)))
    _q = jnp.hstack((c[-1], jnp.zeros((nx,))))
    _W = block_diag(R[-1], inv(Gamma[-1] + lmbda * jnp.eye(nx)))

    G_x = jnp.stack((*G_x, _G_x))
    q = jnp.vstack((q, _q))
    W = jnp.stack((*W, _W))

    # pseudo observations
    m_next = nominal_traj.mean[1:]
    z = jnp.hstack((y, m_next))

    return (
        z,
        MVNStandard(l0, L0),
        LinearTransition(F_x, b, Q),
        LinearObservation(G_x, q, W),
    )


# This is somewhat ugly
def _regularized_recursive_newton_step(
    nominal_traj: MVNStandard,
    observations: jnp.ndarray,
    initial_dist: MVNStandard,
    linear_transition_model: LinearTransition,
    linear_observation_model: LinearObservation,
    transition_hess: jnp.ndarray,
    observation_hess: jnp.ndarray,
    nonlin_transition_model: FunctionalModel,
    nonlin_observation_model: FunctionalModel,
    linearization_method: Callable,
    lmbda: float,
):
    # create Newton state space model
    (
        psdo_obs,
        psdo_init,
        psdo_trns_mdl,
        psdo_obs_mdl,
    ) = _build_newton_state_space(
        observations,
        initial_dist,
        linear_transition_model,
        linear_observation_model,
        transition_hess,
        observation_hess,
        nominal_traj,
        lmbda,
    )

    # filtering on Newton SSM
    filtered_traj = _filtering(
        psdo_obs, psdo_init, psdo_trns_mdl, psdo_obs_mdl
    )

    # smoothing on standard SSM
    smoothed_traj = smoothing(
        nonlin_transition_model,
        filtered_traj,
        linearization_method,
        nominal_traj,
    )

    approx_cost_diff = _approx_log_posterior(
        nominal_traj.mean,
        psdo_obs,
        psdo_init,
        psdo_trns_mdl,
        psdo_obs_mdl,
    ) - _approx_log_posterior(
        smoothed_traj.mean,
        psdo_obs,
        psdo_init,
        psdo_trns_mdl,
        psdo_obs_mdl,
    )

    return smoothed_traj, approx_cost_diff


def _recursive_newton_step(
    nominal_traj: MVNStandard,
    observations: jnp.ndarray,
    initial_dist: MVNStandard,
    transition_model: FunctionalModel,
    observation_model: FunctionalModel,
    linearization_method: Callable,
    quadratization_method: Callable,
):

    # pre-build psdo_hess
    lin_trns_mdl, lin_obs_mdl, trns_hess, obs_hess = _build_pseudo_hessians(
        observations,
        transition_model,
        observation_model,
        quadratization_method,
        nominal_traj,
    )

    def _local_step_func(lmbda):
        return _regularized_recursive_newton_step(
            nominal_traj,
            observations,
            initial_dist,
            lin_trns_mdl,
            lin_obs_mdl,
            trns_hess,
            obs_hess,
            transition_model,
            observation_model,
            linearization_method,
            lmbda,
        )

    def _modify_direction():
        def cond(carry):
            lmbda = carry
            _, df = _local_step_func(lmbda)
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
        return _local_step_func(lmbda)[0]

    # try with no regularization first
    _smoothed_traj, approx_cost_diff = _local_step_func(0.0)

    # modify direciton if necessary
    smoothed_traj = jax.lax.cond(
        approx_cost_diff > 0.0, lambda: _smoothed_traj, _modify_direction,
    )

    return smoothed_traj
