from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev
from jax.experimental.host_callback import id_print
from jax.scipy.linalg import cho_solve

from parsmooth._base import MVNStandard, FunctionalModel, are_inputs_compatible
from parsmooth._utils import none_or_shift, none_or_concat


def filtering(observations: jnp.ndarray,
              x0: MVNStandard,
              transition_model: FunctionalModel,
              observation_model: FunctionalModel,
              linearization_method_hessian: Callable,
              nominal_trajectory: Optional[MVNStandard] = None,
              information: bool = True):
    if nominal_trajectory is not None:
        are_inputs_compatible(x0, nominal_trajectory)

    def predict(F_x, cov_or_chol, b, x):
        return _predict(F_x, cov_or_chol, b, x)

    def update(H_x, cov_or_chol, c, x, y):
        return _update(H_x, cov_or_chol, c, x, y)

    def body(carry, inp):
        x = carry
        y, x_predict_nominal, x_update_nominal = inp

        if x_predict_nominal is None:
            x_predict_nominal = x
        F_x, Q, b, F_xx = linearization_method_hessian(transition_model, x_predict_nominal)
        x = predict(F_x, Q, b, x)
        if x_update_nominal is None:
            x_update_nominal = x
        H_x, R, c, H_xx = linearization_method_hessian(observation_model, x_update_nominal)

        x = update(H_x, R, c, x, y)
        x = _pseudo_update(transition_model, observation_model, F_xx, H_xx, x_predict_nominal, x_update_nominal, Q,
                           R, y, x)
        return x, x

    predict_traj = none_or_shift(nominal_trajectory, -1)
    update_traj = none_or_shift(nominal_trajectory, 1)

    _, xs = jax.lax.scan(body, x0, (observations, predict_traj, update_traj))
    xs = none_or_concat(xs, x0, 1)

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


def hessian(f):
    return jacfwd(jacrev(f))


def vectens(a, b):
    """"
    a is a 3 dimensional tensor
    b is a 1 dimensional array
    """
    return jnp.sum(a * b[:, None, None], 0)


def _pseudo_update(transition_model, observation_model, F_xx, H_xx, x_predict, x_update, Q, R, y, xf):
    mp_nominal, Pp_nominal = x_predict
    mu_nominal, Pu_nominal = x_update
    x_f, P_f = xf
    f, _ = transition_model
    h, _ = observation_model

    Lambda = jnp.tensordot(-F_xx.T, Q @ (mu_nominal - f(mp_nominal)), axes=1)
    Phi = jnp.tensordot(-H_xx.T, R @ (y - h(mu_nominal)), axes=1)
    nx = Q.shape[0]
    Sigma = P_f + jnp.linalg.inv(Lambda + Phi + 1e-12*jnp.eye(nx, nx))
    Sigma = (Sigma + Sigma.T)/2
    # chol_Sigma = jnp.linalg.cholesky(Sigma)
    # K = cho_solve((chol_Sigma, True), P_f.T).T
    # K = jax.scipy.linalg.solve(Sigma.T, P_f.T).T
    # # K = P_f @ jnp.linalg.inv(Sigma)
    #
    # x = x_f + K @ (mu_nominal - x_f)
    # P = P_f - K @ Sigma @ K.T
    #

    # using information form
    P_inf = jnp.linalg.inv(jnp.linalg.inv(P_f) + Lambda + Phi)
    temp = (Lambda + Phi) @ mu_nominal + jnp.linalg.inv(P_f)  @ x_f
    x_inf = P_inf @ temp

    return MVNStandard(x_inf, P_inf)

