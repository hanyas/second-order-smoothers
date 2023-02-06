from typing import Callable, Optional

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg
from jax.scipy.linalg import cho_solve

from parsmooth._base import MVNStandard, FunctionalModel, are_inputs_compatible
from parsmooth._utils import none_or_shift, none_or_concat
from parsmooth.sequential._smoothing_Newton import smoothing as newton_smoothing


def mvn_loglikelihood(x, chol_cov):
    """multivariate normal"""
    dim = chol_cov.shape[0]
    y = jlinalg.solve_triangular(chol_cov, x, lower=True)
    normalizing_constant = (
            jnp.sum(jnp.log(jnp.abs(jnp.diag(chol_cov)))) + dim * jnp.log(2 * jnp.pi) / 2.0
    )
    norm_y = jnp.sum(y * y, -1)
    return -0.5 * norm_y - normalizing_constant


def L(predict_trajectory, update_trajectory, z, measurement_fun, dynamic_fun, chol_Q, chol_R):
    mp_nominal = predict_trajectory
    mu_nominal = update_trajectory
    cost = 2 * mvn_loglikelihood(mu_nominal - dynamic_fun(mp_nominal), chol_Q) + 2 * mvn_loglikelihood(z - measurement_fun(mu_nominal), chol_R)
    return -cost


def state_space_cost(x, ys, observation_function, transition_function, Q, R, m0, P0):
    x0 = x[0]
    predict_traj =x[:-1]
    update_traj = x[1:]
    vmapped_fun = jax.vmap(L, in_axes=[0, 0, 0, None, None, None, None])
    return jnp.sum(vmapped_fun(predict_traj, update_traj, ys, observation_function, transition_function, jnp.linalg.cholesky(Q), jnp.linalg.cholesky(R))) - 2 * mvn_loglikelihood(x0 - m0, jnp.linalg.cholesky(P0))


def LM_Newton(current_nomminal_trajectory: Optional[MVNStandard],
              ys: jnp.ndarray,
              init: MVNStandard,
              observation_model: FunctionalModel,
              transition_model: FunctionalModel,
              linearization_method_hessian: Callable,
              linearization_method,
              nu: jnp.ndarray = 10.,
              lam: jnp.ndarray = 1e-2,
              n_iter: int = 10):
    m0, P0 = init
    transition_function, Q = transition_model[0], transition_model[1].cov
    observation_function, R = observation_model[0], observation_model[1].cov
    J1 = state_space_cost(current_nomminal_trajectory.mean, ys, observation_function, transition_function, Q, R, m0, P0)

    def body(carry, _):
        nominal_trajectory, lam_param, J = carry
        nu = 10.
        filtered_newton = newton_filtering_lam(ys, init, transition_model, observation_model, linearization_method_hessian, nominal_trajectory, lam_param, True)
        smoothed_newton = newton_smoothing(transition_model, filtered_newton, linearization_method, nominal_trajectory)

        J_new = state_space_cost(smoothed_newton.mean, ys, observation_function, transition_function, Q, R, m0, P0)

        lam_param, J = jax.lax.cond(J_new < J, lambda _: (lam_param / nu, J_new), lambda _: (nu * lam_param, J), operand=None)

        return (smoothed_newton, lam_param, J), J

    LM_smoothed_trajectories, Js = jax.lax.scan(body, (current_nomminal_trajectory, lam, J1), jnp.arange(n_iter))

    return LM_smoothed_trajectories, Js


def newton_filtering_lam(observations: jnp.ndarray,
                         x0: MVNStandard,
                         transition_model: FunctionalModel,
                         observation_model: FunctionalModel,
                         linearization_method_hessian: Callable,
                         nominal_trajectory: Optional[MVNStandard] = None,
                         lam: jnp.ndarray = 1e-2,
                         information: bool = True):
    if nominal_trajectory is not None:
        are_inputs_compatible(x0, nominal_trajectory)

    # first step
    f, _ = transition_model
    F, Q, _, F_xx = linearization_method_hessian(transition_model, x0)
    P0_inv = jnp.linalg.inv(x0.cov)
    Psi0 = jnp.tensordot(-F_xx.T, jnp.linalg.inv(Q) @ (nominal_trajectory.mean[1] - f(x0.mean)),
                         axes=1)
    S0 = jnp.diag(1/(jnp.diag(P0_inv) + jnp.diag(Psi0) + jnp.diag(F.T @ jnp.linalg.inv(Q) @ F)))

    P0_Newton_LM = jnp.linalg.inv(P0_inv + Psi0 + 1/lam * S0)
    x0_Newton = MVNStandard(x0.mean, P0_Newton_LM)

    # middle steps
    def predict(F_x, cov_or_chol, b, x):
        return _predict(F_x, cov_or_chol, b, x)

    def update(H_x, cov_or_chol, c, x, y):
        return _update(H_x, cov_or_chol, c, x, y)

    def body(carry, inp):
        x = carry
        y, x_predict_nominal, x_update_nominal, x_nominal_p_1 = inp

        if x_predict_nominal is None:
            x_predict_nominal = x
        F_x, Q, b, _ = linearization_method_hessian(transition_model, x_predict_nominal)
        x = predict(F_x, Q, b, x)
        if x_update_nominal is None:
            x_update_nominal = x

        F_x_, _, _, F_xx = linearization_method_hessian(transition_model, x_update_nominal)
        H_x, R, c, H_xx = linearization_method_hessian(observation_model, x_update_nominal)

        x = update(H_x, R, c, x, y)
        x = _pseudo_update(transition_model, observation_model, F_xx, H_xx, F_x_, H_x, x_update_nominal, y, x, x_nominal_p_1, lam, information)
        return x, x

    predict_traj = none_or_shift(nominal_trajectory, -1)
    update_traj = none_or_shift(nominal_trajectory, 1)

    x, xs = jax.lax.scan(body, x0_Newton, (observations[:-1], none_or_shift(predict_traj, -1), none_or_shift(update_traj, -1), none_or_shift(update_traj, 1)))

    # last step
    F_x, Q, b, F_xx = linearization_method_hessian(transition_model,
                                                   MVNStandard(predict_traj.mean[-1], predict_traj.cov[-1]))
    x = predict(F_x, Q, b, x)
    H_x, R, c, H_xx = linearization_method_hessian(observation_model,
                                                   MVNStandard(update_traj.mean[-1], update_traj.cov[-1]))

    x = update(H_x, R, c, x, observations[-1])
    x_N = MVNStandard(update_traj.mean[-1], update_traj.cov[-1])
    x = _pseudo_update(transition_model, observation_model, 0 * F_xx, H_xx, 0*F_x, H_x, x_N, observations[-1], x, x,
                       lam,information)

    xs = none_or_concat(xs, x, -1)
    xs = none_or_concat(xs, x0_Newton, 1)

    return xs


def _predict(F, Q, b, x):
    m, P = x

    m = F @ m + b
    P = Q + F @ P @ F.T

    return MVNStandard(m, P)


def _update(H, R, c, x, y):
    m, P = x

    y_hat = H @ m + c
    y_diff = y - y_hat
    S = R + H @ P @ H.T
    chol_S = jnp.linalg.cholesky(S)
    G = P @ cho_solve((chol_S, True), H).T

    m = m + G @ y_diff
    P = P - G @ S @ G.T
    return MVNStandard(m, P)


def _pseudo_update(transition_model,
                   observation_model,
                   F_xx,
                   H_xx,
                   F_x,
                   H_x,
                   x_update,
                   y,
                   xf,
                   x_p_1,
                   lam,
                   information):

    mu_nominal, Pu_nominal = x_update
    x_f, P_f = xf
    f, Q = transition_model[0], transition_model[1].cov
    h, R = observation_model[0], observation_model[1].cov
    x_p_1_, _ = x_p_1

    Q_inv = jnp.linalg.inv(Q)
    R_inv = jnp.linalg.inv(R)
    Lambda = jnp.tensordot(-F_xx.T, jnp.linalg.inv(Q) @ (x_p_1_ - f(mu_nominal)), axes=1)
    Phi = jnp.tensordot(-H_xx.T, jnp.linalg.inv(R) @ (y - h(mu_nominal)), axes=1)
    S = jnp.diag(1 / (jnp.diag(Q_inv) + jnp.diag(Lambda) + jnp.diag(Phi) + jnp.diag(F_x.T @ Q_inv @ F_x) + jnp.diag(H_x.T @ R_inv @ H_x)))

    if information:
        P = jnp.linalg.inv(jnp.linalg.inv(P_f) + Lambda + Phi + 1/lam * S)
        temp = (Lambda + Phi + 1/lam * S) @ mu_nominal + jnp.linalg.inv(P_f) @ x_f
        x = P @ temp
    else:
        nx = Q.shape[0]
        Sigma = P_f + jnp.linalg.inv(Lambda + Phi + 1/lam * S)
        Sigma = (Sigma + Sigma.T) / 2
        chol_Sigma = jnp.linalg.cholesky(Sigma)
        K = cho_solve((chol_Sigma, True), P_f.T).T
        # K = jax.scipy.linalg.solve(Sigma.T, P_f.T).T
        # K = P_f @ jnp.linalg.inv(Sigma)
        y_diff = mu_nominal - x_f

        x = x_f + K @ y_diff
        P = P_f - K @ Sigma @ K.T

    return MVNStandard(x, P)

