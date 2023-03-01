from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy as jsc
from jax.scipy.linalg import solve, cho_solve
from jax.flatten_util import ravel_pytree

from jaxopt import BacktrackingLineSearch

from smoothopt._base import MVNStandard, FunctionalModel
from smoothopt._base import LinearTransition, LinearObservation
from smoothopt._utils import none_or_shift, none_or_concat
from smoothopt.sequential._smoothing import smoothing as newton_smoothing


logdet = lambda x: jnp.linalg.slogdet(x)[1]


def _iterated_batch_newton_smoother(observations: jnp.ndarray, initial_dist: MVNStandard,
                                    transition_model: FunctionalModel, observation_model: FunctionalModel,
                                    quadratization_method: Callable, nominal_mean: jnp.ndarray,
                                    n_iter: int = 10):

    flat_nominal_mean, unravel = ravel_pytree(nominal_mean)

    def _flattend_log_posterior(flat_state):
        _state = unravel(flat_state)
        return log_posterior(_state, observations, initial_dist,
                             transition_model, observation_model)

    init_cost = _flattend_log_posterior(flat_nominal_mean)

    ls = BacktrackingLineSearch(fun=_flattend_log_posterior,
                                maxiter=100, condition="strong-wolfe")

    def body(carry, _):
        nominal_mean, last_cost = carry

        jac = jax.jacobian(_flattend_log_posterior)(nominal_mean)
        hess = jax.hessian(_flattend_log_posterior)(nominal_mean)

        # jac, hess = jac_and_hess(observations,
        #                          initial_dist,
        #                          transition_model,
        #                          observation_model,
        #                          quadratization_method,
        #                          unravel(nominal_mean))

        def _modify_hessian_cond(carry):
            hess, jac, search_dir, lmbda = carry
            hess_reg = hess + lmbda * jnp.eye(hess.shape[0])
            approx_cost_diff = - jnp.dot(search_dir, jac) \
                               - 0.5 * jnp.dot(jnp.dot(search_dir, hess_reg), search_dir)
            return approx_cost_diff <= 0.0

        def _modify_hessian_body(carry):
            hess, jac, _, lmbda = carry
            lmbda = lmbda * 10.0
            hess_reg = hess + lmbda * jnp.eye(hess.shape[0])
            search_dir = - jnp.linalg.solve(hess_reg, jac)
            return hess, jac, search_dir, lmbda

        def _modify_hessian(args):
            hess, jac, search_dir = args
            _, _, search_dir, lmbda = jax.lax.while_loop(_modify_hessian_cond,
                                                         _modify_hessian_body,
                                                         (hess, jac, search_dir, 1e-8))
            return search_dir, lmbda

        def _keep_hessian(args):
            hess, jac, search_dir = args
            return search_dir, 0.0

        search_dir = - jnp.linalg.solve(hess, jac)
        approx_cost_diff = - jnp.dot(search_dir, jac) \
                           - 0.5 * jnp.dot(jnp.dot(search_dir, hess), search_dir)

        # modify Hessian if necessary
        search_dir, _ = jax.lax.cond(approx_cost_diff > 0.0,
                                     _keep_hessian, _modify_hessian,
                                     (hess, jac, search_dir))

        # line search optimal step size
        alpha, state = ls.run(init_stepsize=1.0, params=nominal_mean,
                              descent_direction=search_dir)

        # update smoothed mean
        smoothed_mean = nominal_mean + alpha * search_dir

        new_cost = _flattend_log_posterior(smoothed_mean)
        return (smoothed_mean, new_cost), new_cost

    (flat_nominal_mean, _), costs = jax.lax.scan(body, (flat_nominal_mean, init_cost),
                                                 jnp.arange(n_iter))

    nominal_mean = unravel(flat_nominal_mean)
    return nominal_mean, jnp.hstack((init_cost, costs))


def _iterated_recursive_newton_smoother(observations: jnp.ndarray, initial_dist: MVNStandard,
                                        transition_model: FunctionalModel, observation_model: FunctionalModel,
                                        quadratization_method: Callable, linearization_method: Callable,
                                        nominal_trajectory: MVNStandard,
                                        n_iter: int = 10):

    init_cost = log_posterior(nominal_trajectory.mean, observations,
                              initial_dist, transition_model, observation_model)

    def _newton_update(nominal_trajectory, lmbda):
        psdo_obs, psdo_initial, \
            psdo_trans_mdl, psdo_obs_mdl = newton_state_space(observations,
                                                              initial_dist,
                                                              transition_model,
                                                              observation_model,
                                                              quadratization_method,
                                                              nominal_trajectory,
                                                              lmbda)

        filtered_trajectory = newton_filtering(psdo_obs,
                                               psdo_initial,
                                               psdo_trans_mdl, psdo_obs_mdl)

        smoothed_trajectory = newton_smoothing(transition_model, filtered_trajectory,
                                               linearization_method, nominal_trajectory)

        new_approximate_cost = approx_log_posterior(smoothed_trajectory.mean, psdo_obs,
                                                    psdo_initial, psdo_trans_mdl, psdo_obs_mdl)

        last_approximate_cost = approx_log_posterior(nominal_trajectory.mean, psdo_obs,
                                                     psdo_initial, psdo_trans_mdl, psdo_obs_mdl)

        approx_cost_diff = last_approximate_cost - new_approximate_cost
        return smoothed_trajectory, approx_cost_diff

    def body(carry, _):
        nominal_trajectory, last_cost = carry

        def _modify_hessian_cond(carry):
            lmbda = carry
            smoothed_trajectory, approx_cost_diff = _newton_update(nominal_trajectory, lmbda)
            return approx_cost_diff <= 0.0

        def _modify_hessian_body(carry):
            lmbda = carry
            return lmbda * 10.0

        def _modify_hessian(args):
            _ = args
            lmbda = jax.lax.while_loop(_modify_hessian_cond,
                                       _modify_hessian_body, 1e-8)
            smoothed_trajectory, _ = _newton_update(nominal_trajectory, lmbda)
            return smoothed_trajectory, lmbda

        def _keep_hessian(args):
            smoothed_trajectory = args
            return smoothed_trajectory, 0.0

        smoothed_trajectory, approx_cost_diff = _newton_update(nominal_trajectory, 1e-16)

        # regularize model if necessary
        smoothed_trajectory, _ = jax.lax.cond(approx_cost_diff > 0.0,
                                              _keep_hessian, _modify_hessian,
                                              smoothed_trajectory)

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


def newton_filtering(observations: jnp.ndarray,
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


def newton_state_space(observations: jnp.ndarray,
                       initial_dist: MVNStandard,
                       transition_model: FunctionalModel,
                       observation_model: FunctionalModel,
                       quadratization_method: Callable,
                       nominal_trajectory: MVNStandard,
                       lmbda: float = 1e-2):

    y = observations
    m0, P0 = initial_dist

    f, _ = transition_model
    h, _ = observation_model

    curr_nominal = none_or_shift(nominal_trajectory, -1)
    next_nominal = none_or_shift(nominal_trajectory, 1)

    F_xx, F_x, Q, f0 = jax.vmap(quadratization_method, in_axes=(None, 0))(transition_model, curr_nominal)
    H_xx, H_x, R, h0 = jax.vmap(quadratization_method, in_axes=(None, 0))(observation_model, next_nominal)

    nx = Q.shape[-1]
    ny = R.shape[-1]

    def _dynamics_hessian(F_xx, Q, m_next, m_curr):
        return - jnp.einsum('ijk,k->ij', F_xx.T, jnp.linalg.solve(Q, m_next - f(m_curr)))

    def _observation_hessian(H_xx, R, y, m_next):
        return - jnp.einsum('ijk,k->ij', H_xx.T, jnp.linalg.solve(R, y - h(m_next)))

    m_curr = curr_nominal.mean
    m_next = next_nominal.mean

    Phi = jax.vmap(_dynamics_hessian)(F_xx, Q, m_next, m_curr)
    Gamma = jax.vmap(_observation_hessian)(H_xx, R, y, m_next)

    # first time step
    l0 = m0
    L0 = jnp.linalg.inv(jnp.linalg.inv(P0) + Phi[0] + lmbda * jnp.eye(nx))

    # intermediate time steps
    def _observation_model(H_x, h0, R, Phi, Gamma):
        G_x = jnp.vstack((H_x, jnp.eye(nx)))
        g0 = jnp.hstack((h0, jnp.zeros((nx,))))
        W = jsc.linalg.block_diag(R, jnp.linalg.inv(Phi + Gamma + lmbda * jnp.eye(nx)))
        return G_x, g0, W

    G_x, g0, W = jax.vmap(_observation_model)(H_x[:-1], h0[:-1], R[:-1],
                                              Phi[1:], Gamma[:-1])

    # final time step
    _G_x = jnp.vstack((H_x[-1], jnp.eye(nx)))
    _g0 = jnp.hstack((h0[-1], jnp.zeros((nx,))))
    _W = jsc.linalg.block_diag(R[-1], jnp.linalg.inv(Gamma[-1] + lmbda * jnp.eye(nx)))

    G_x = jnp.stack((*G_x, _G_x))
    g0 = jnp.vstack((g0, _g0))
    W = jnp.stack((*W, _W))

    # pseudo observations
    z = jnp.hstack((y, m_next))

    return z, MVNStandard(l0, L0),\
        LinearTransition(F_x, f0, Q),\
        LinearObservation(G_x, g0, W)


def jac_and_hess(observations: jnp.ndarray,
                 initial_dist: MVNStandard,
                 transition_model: FunctionalModel,
                 observation_model: FunctionalModel,
                 quadratization_method: Callable,
                 nominal_mean: jnp.ndarray):

    y = observations
    m0, P0 = initial_dist

    f, (_, Q) = transition_model
    h, (_, R) = observation_model

    m_curr = nominal_mean[:-1]
    m_next = nominal_mean[1:]

    f0, F_x, F_xx = jax.vmap(quadratization_method, in_axes=(None, 0))(f, m_curr)
    h0, H_x, H_xx = jax.vmap(quadratization_method, in_axes=(None, 0))(h, m_next)

    T = nominal_mean.shape[0]
    nx = Q.shape[-1]
    ny = R.shape[-1]

    def _dynamics_hessian(F_xx, m_next, m_curr):
        return - jnp.einsum('ijk,k->ij', F_xx.T, jnp.linalg.solve(Q, m_next - f(m_curr)))

    def _observation_hessian(H_xx, y, m_next):
        return - jnp.einsum('ijk,k->ij', H_xx.T, jnp.linalg.solve(R, y - h(m_next)))

    Phi = jax.vmap(_dynamics_hessian)(F_xx, m_next, m_curr)
    Gamma = jax.vmap(_observation_hessian)(H_xx, y, m_next)

    P0_inv = jnp.linalg.inv(P0)
    Q_inv = jnp.linalg.inv(Q)
    R_inv = jnp.linalg.inv(R)

    hess = jnp.zeros((T * nx, T * nx))
    hess_aux = jnp.zeros((T * nx, T * nx))

    # off-diagonal
    def _off_diagonal(carry, args):
        F_x, t = args
        hess = carry

        val = - Q_inv @ F_x
        hess = jax.lax.dynamic_update_slice(hess, val, ((t + 1) * nx, t * nx))
        hess = jax.lax.dynamic_update_slice(hess, val.T, (t * nx, (t + 1) * nx))
        return hess, _

    hess, _ = jax.lax.scan(_off_diagonal, hess, (F_x, jnp.arange(T - 1)))

    # first diagonal
    val = P0_inv\
          + F_x[0].T @ Q_inv @ F_x[0]\
          + Phi[0]

    hess = hess.at[:nx, :nx].set(val)

    # intermediate diagonal
    def _diagonal(carry, args):
        F_x, H_x, Phi, Gamma, t = args
        hess = carry

        val = Q_inv\
              + F_x.T @ Q_inv @ F_x\
              + H_x.T @ R_inv @ H_x\
              + Phi + Gamma

        hess = jax.lax.dynamic_update_slice(hess, val, (t * nx, t * nx))
        return hess, _

    hess, _ = jax.lax.scan(_diagonal, hess, (F_x[1:], H_x[:-1],
                                             Phi[1:], Gamma[:-1],
                                             jnp.arange(1, T - 1)))

    # last diagonal
    val = Q_inv \
          + H_x[-1].T @ R_inv @ H_x[-1] \
          + Gamma[-1]

    hess = hess.at[-nx:, -nx:].set(val)

    # jac vector
    jac = jnp.zeros((T * nx, ))

    m = nominal_mean

    # first vector
    val = P0_inv @ (m[0] - m0)\
          - F_x[0].T @ Q_inv @ (m[1] - f(m[0]))
    jac = jac.at[:nx].set(val)

    # intermediate vectors
    def _intermediate(carry, args):
        y, m_prev, m_curr, m_next,\
            F_x, H_x, t = args
        jac = carry

        val = Q_inv @ (m_curr - f(m_prev))\
              - F_x.T @ Q_inv @ (m_next - f(m_curr))\
              - H_x.T @ R_inv @ (y - h(m_curr))

        jac = jax.lax.dynamic_update_slice(jac, val, (t * nx, ))
        return jac, _

    m_prev = nominal_mean[:-2]
    m_curr = nominal_mean[1:-1]
    m_next = nominal_mean[2:]

    jac, _ = jax.lax.scan(_intermediate, jac, (y[:-1], m_prev,
                                               m_curr, m_next,
                                               F_x[1:], H_x[:-1],
                                               jnp.arange(1, T - 1)))

    # last vector
    val = Q_inv @ (m[-1] - f(m[-2])) \
          - H_x[-1].T @ R_inv @ (y[-1] - h(m[-1]))

    jac = jac.at[-nx:].set(val)

    return jac, hess


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


def approx_log_posterior(states, observations,
                         initial_dist, transition_model, observation_model):

    xp, xn = states[:-1], states[1:]
    yn = observations

    m0, P0 = initial_dist
    F_x, f0, Q = transition_model
    H_x, h0, R = observation_model

    _xn = jnp.einsum('nij,nj->ni', F_x, xp) + f0
    _yn = jnp.einsum('nij,nj->ni', H_x, xn) + h0

    cost = jnp.sum(jax.vmap(mvn_loglikelihood)(xn - _xn, Q))\
           + jnp.sum(jax.vmap(mvn_loglikelihood)(yn - _yn, R))\
           + mvn_loglikelihood(states[0] - m0, P0)

    return - cost


def mvn_loglikelihood(x, cov):
    dim = cov.shape[0]
    normalizer = logdet(cov) / 2.0 + dim * jnp.log(2.0 * jnp.pi) / 2.0
    norm = jnp.dot(x, solve(cov, x))
    return - 0.5 * norm - normalizer
