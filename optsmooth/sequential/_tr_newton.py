from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy as jsc

from optsmooth.base import MVNStandard, FunctionalModel
from optsmooth.base import LinearTransition, LinearObservation
from optsmooth.utils import mvn_logpdf, none_or_shift, none_or_concat
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


def trust_region_iterated_recursive_newton_smoother(
    observations: jnp.ndarray,
    initial_dist: MVNStandard,
    transition_model: FunctionalModel,
    observation_model: FunctionalModel,
    linearization_method: Callable,
    quadratization_method: Callable,
    init_nominal_traj: MVNStandard,
    lmbda: float = 1e2,
    nu: float = 2.0,
    nb_iter: int = 10,
):
    init_cost = log_posterior(
        init_nominal_traj.mean,
        observations,
        initial_dist,
        transition_model,
        observation_model,
    )

    def _newton_step(nominal_traj, lmbda):
        return _recursive_newton_step(
            nominal_traj,
            observations,
            initial_dist,
            transition_model,
            observation_model,
            linearization_method,
            quadratization_method,
            lmbda,
        )

    def body(carry, _):
        nominal_traj, lmbda, nu = carry

        _smoothed_traj, approx_cost_diff = _newton_step(nominal_traj, lmbda)

        nominal_cost = log_posterior(
            nominal_traj.mean,
            observations,
            initial_dist,
            transition_model,
            observation_model,
        )

        _smoothed_cost = log_posterior(
            _smoothed_traj.mean,
            observations,
            initial_dist,
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
            return _smoothed_traj, _smoothed_cost, lmbda, 2.0

        def reject(args):
            lmbda, nu = args
            lmbda = jnp.minimum(1e16, lmbda)
            return nominal_traj, nominal_cost, lmbda * nu, 2.0 * nu

        smoothed_traj, cost, lmbda, nu = jax.lax.cond(
            (cost_ratio > 0.0) & (approx_cost_diff > 0.0),
            accept,
            reject,
            operand=(lmbda, nu),
        )
        return (smoothed_traj, lmbda, nu), cost

    (nominal_traj, _, _), costs = jax.lax.scan(
        body, (init_nominal_traj, lmbda, nu), jnp.arange(nb_iter)
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

    F, b, Q = transition_model
    H, c, R = observation_model

    Phi = transition_hess
    Gamma = observation_hess

    nx = Q.shape[-1]
    ny = R.shape[-1]

    from jax.scipy.linalg import block_diag, inv

    # first time step
    l0 = m0
    L0 = jnp.linalg.inv(jnp.linalg.inv(P0) + Phi[0] + lmbda * jnp.eye(nx))

    # intermediate time steps
    def _build_observation_model(H, c, R, Phi, Gamma):
        G = jnp.vstack((H, jnp.eye(nx)))
        q = jnp.hstack((c, jnp.zeros((nx,))))
        W = block_diag(R, inv(Phi + Gamma + lmbda * jnp.eye(nx)))
        return G, q, W

    G, q, W = jax.vmap(_build_observation_model)(
        H[:-1], c[:-1], R[:-1], Phi[1:], Gamma[:-1]
    )

    # final time step
    _G = jnp.vstack((H[-1], jnp.eye(nx)))
    _q = jnp.hstack((c[-1], jnp.zeros((nx,))))
    _W = block_diag(R[-1], inv(Gamma[-1] + lmbda * jnp.eye(nx)))

    G = jnp.stack((*G, _G))
    q = jnp.vstack((q, _q))
    W = jnp.stack((*W, _W))

    # pseudo observations
    m_next = nominal_traj.mean[1:]
    z = jnp.hstack((y, m_next))

    return (
        z,
        MVNStandard(l0, L0),
        LinearTransition(F, b, Q),
        LinearObservation(G, q, W),
    )


def _recursive_newton_step(
    nominal_traj: MVNStandard,
    observations: jnp.ndarray,
    initial_dist: MVNStandard,
    transition_model: FunctionalModel,
    observation_model: FunctionalModel,
    linearization_method: Callable,
    quadratization_method: Callable,
    lmbda: float,
):
    # pre-build psdo_hess
    lin_trns_mdl, lin_obs_mdl, trns_hess, obs_hess = _build_pseudo_hessians(
        observations,
        transition_model,
        observation_model,
        quadratization_method,
        nominal_traj,
    )

    # create Newton state space model
    (
        psdo_obs,
        psdo_init,
        psdo_trns_mdl,
        psdo_obs_mdl,
    ) = _build_newton_state_space(
        observations,
        initial_dist,
        lin_trns_mdl,
        lin_obs_mdl,
        trns_hess,
        obs_hess,
        nominal_traj,
        lmbda,
    )

    # filtering on Newton SSM
    filtered_traj = _filtering(
        psdo_obs, psdo_init, psdo_trns_mdl, psdo_obs_mdl
    )

    # smoothing on standard SSM
    smoothed_traj = smoothing(
        transition_model,
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
