from typing import Callable

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve
from smoothopt._base import MVNStandard, FunctionalModel
from smoothopt._base import LinearTransition, LinearObservation
from smoothopt._utils import none_or_shift, none_or_concat
from smoothopt.sequential._smoothing import smoothing as gauss_newton_smoothing


logdet = lambda x: jnp.linalg.slogdet(x)[1]


def _iterated_recursive_gauss_newton_smoother(observations: jnp.ndarray, initial_dist: MVNStandard,
                                              transition_model: FunctionalModel, observation_model: FunctionalModel,
                                              linearization_method: Callable, nominal_trajectory: MVNStandard,
                                              n_iter: int = 10):

    init_cost = log_posterior(nominal_trajectory.mean, observations,
                              initial_dist, transition_model, observation_model)

    def _gauss_newton_update(nominal_trajectory):
        psdo_obs, psdo_initial, \
            psdo_trans_mdl, psdo_obs_mdl = gauss_newton_state_space(observations,
                                                              initial_dist,
                                                              transition_model,
                                                              observation_model,
                                                              linearization_method,
                                                              nominal_trajectory)

        filtered_trajectory = gauss_newton_filtering(psdo_obs,
                                               psdo_initial,
                                               psdo_trans_mdl, psdo_obs_mdl)

        smoothed_trajectory = gauss_newton_smoothing(transition_model, filtered_trajectory,
                                               linearization_method, nominal_trajectory)
        return smoothed_trajectory

    def body(carry, _):
        nominal_trajectory, last_cost = carry

        smoothed_trajectory = _gauss_newton_update(nominal_trajectory)

        # line search
        def cond(carry):
            _, backtrack, new_cost = carry
            return jnp.logical_and((new_cost > last_cost), (backtrack < 50))

        def body(carry):
            alpha, backtrack, _ = carry
            alpha = 0.5 * alpha
            backtrack = backtrack + 1
            interpolated_mean = nominal_trajectory.mean\
                                + alpha * (smoothed_trajectory.mean - nominal_trajectory.mean)
            new_cost = log_posterior(interpolated_mean, observations,
                                     initial_dist, transition_model, observation_model)
            return alpha, backtrack, new_cost

        alpha, backtrack = 1.0, 0
        new_cost = log_posterior(smoothed_trajectory.mean, observations,
                                 initial_dist, transition_model, observation_model)
        alpha, _, new_cost = jax.lax.while_loop(cond, body, (alpha, backtrack, new_cost))

        # update smoothed trajectory
        interpolated_mean = nominal_trajectory.mean \
                            + alpha * (smoothed_trajectory.mean - nominal_trajectory.mean)

        smoothed_trajectory = MVNStandard(interpolated_mean, smoothed_trajectory.cov)
        return (smoothed_trajectory, new_cost), new_cost

    (nominal_trajectory, _), costs = jax.lax.scan(body,
                                                  (nominal_trajectory, init_cost),
                                                  jnp.arange(n_iter))

    return nominal_trajectory, jnp.hstack((init_cost, costs))


def gauss_newton_filtering(observations: jnp.ndarray,
                     initial_dist: MVNStandard,
                     linear_transition: LinearTransition,
                     linear_observation: LinearObservation):

    y = observations

    F_x, f0, Q = linear_transition
    H_x, h0, R = linear_observation

    xf0 = initial_dist

    def _predict(F, f0, Q, x):
        m, P = x

        m = F @ m + f0
        P = Q + F @ P @ F.T
        return MVNStandard(m, P)

    def _update(H, h0, R, x, y):
        m, P = x

        y_hat = H @ m + h0
        y_diff = y - y_hat
        S = R + H @ P @ H.T
        # chol_S = jnp.linalg.cholesky(S)
        # G = P @ cho_solve((chol_S, True), H).T
        G = jnp.linalg.solve(S.T, H @ P.T).T

        m = m + G @ y_diff
        P = P - G @ S @ G.T
        return MVNStandard(m, P)

    def body(carry, args):
        xf = carry
        y, F_x, f0, Q, H_x, h0, R = args

        xp = _predict(F_x, f0, Q, xf)  # prediction
        xf = _update(H_x, h0, R, xp, y)  # innovation

        return xf, xf

    xf, xfs = jax.lax.scan(body, xf0, (y, F_x, f0, Q, H_x, h0, R))
    xfs = none_or_concat(xfs, xf0, 1)
    return xfs


def gauss_newton_state_space(observations: jnp.ndarray,
                       initial_dist: MVNStandard,
                       transition_model: FunctionalModel,
                       observation_model: FunctionalModel,
                       linearization_method: Callable,
                       nominal_trajectory: MVNStandard):

    curr_nominal = none_or_shift(nominal_trajectory, -1)
    next_nominal = none_or_shift(nominal_trajectory, 1)

    F_x, Q, f0 = jax.vmap(linearization_method, in_axes=(None, 0))(transition_model, curr_nominal)
    H_x, R, h0 = jax.vmap(linearization_method, in_axes=(None, 0))(observation_model, next_nominal)

    return observations, initial_dist,\
        LinearTransition(F_x, f0, Q),\
        LinearObservation(H_x, h0, R)


def log_posterior(states, observations,
                  initial_dist, transition_model, observation_model):

    xp, xn = states[:-1], states[1:]
    yn = observations

    m0, P0 = initial_dist
    f, (_, Q) = transition_model
    h, (_, R) = observation_model

    _xn = jax.vmap(f)(xp)
    _yn = jax.vmap(h)(xn)

    cost = jnp.sum(jax.vmap(mvn_loglikelihood, in_axes=(0, None))(xn - _xn, Q)
                   + jax.vmap(mvn_loglikelihood, in_axes=(0, None))(yn - _yn, R))\
           + mvn_loglikelihood(states[0] - m0, P0)

    return - cost


def mvn_loglikelihood(x, cov):
    dim = cov.shape[0]
    normalizer = logdet(cov) / 2.0 + dim * jnp.log(2.0 * jnp.pi) / 2.0
    norm = jnp.dot(x, solve(cov, x))
    return - 0.5 * norm - normalizer
