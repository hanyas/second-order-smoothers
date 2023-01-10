from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev
from jax.scipy.linalg import cho_solve

from parsmooth._base import MVNStandard, FunctionalModel, are_inputs_compatible, ConditionalMomentsModel
from parsmooth._utils import none_or_shift, none_or_concat, mvn_loglikelihood


def filtering(observations: jnp.ndarray,
              x0: MVNStandard,
              transition_model: Union[FunctionalModel, ConditionalMomentsModel],
              observation_model: Union[FunctionalModel, ConditionalMomentsModel],
              linearization_method: Callable,
              nominal_trajectory: Optional[MVNStandard] = None):
    if nominal_trajectory is not None:
        are_inputs_compatible(x0, nominal_trajectory)

    def predict(F_x, cov_or_chol, b, x):
        return _predict(F_x, cov_or_chol, b, x)

    def update(H_x, cov_or_chol, c, x, y):
        return _update(H_x, cov_or_chol, c, x, y)

    def body(carry, inp):
        x, ell = carry
        y, predict_ref, update_ref = inp

        if predict_ref is None:
            predict_ref = x
        F_x, Q, b = linearization_method(transition_model, predict_ref)
        x = predict(F_x, Q, b, x)
        if update_ref is None:
            update_ref = x
        H_x, R, c = linearization_method(observation_model, update_ref, )

        x, ell_inc = update(H_x, R, c, x, y)
        H_xx, F_xx = _pseudo_update(transition_model, observation_model, predict_ref, update_ref, Q, R)
        return (x, ell + ell_inc), x

    predict_traj = none_or_shift(nominal_trajectory, -1)
    update_traj = none_or_shift(nominal_trajectory, 1)

    (_, ell), xs = jax.lax.scan(body, (x0, 0.), (observations, predict_traj, update_traj))
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
    ell = mvn_loglikelihood(y_diff, chol_S)
    return MVNStandard(m, P), ell


def hessian(f):
    return jacfwd(jacrev(f))


vectens = lambda a, b: jnp.sum(a * b[:,None,None], 0)


def _pseudo_update(transition_model, observation_model, x_predict, x_update, Q, R, y, P):
    x_k_1, P_p = x_predict
    x_k, P_u = x_update
    F_xx = hessian(transition_model)(x_k_1)
    H_xx = hessian(observation_model)(x_k)
    Lambda = vectens(-F_xx, Q @ (x_k_1 - transition_model(x_k_1)))
    Phi = vectens(-H_xx, R @ (y - observation_model(x_k)))
    Sigma = P_u + Lambda + Phi
    K = P_p @ Sigma^(-1)
    x = x_k + K @ (...)
    P = P_u - K @ Sigma @ K.T
    xx = MVNStandard(x, P)
    return xx

