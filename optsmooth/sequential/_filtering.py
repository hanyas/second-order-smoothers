from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve

from optsmooth._base import MVNStandard, FunctionalModel, are_inputs_compatible
from optsmooth._utils import none_or_shift, none_or_concat, mvn_logpdf_chol


def filtering(observations: jnp.ndarray,
              init_dist: MVNStandard,
              transition_model: FunctionalModel,
              observation_model: FunctionalModel,
              linearization_method: Callable,
              nominal_trajectory: Optional[MVNStandard] = None,
              return_loglikelihood: bool = False):

    if nominal_trajectory is not None:
        are_inputs_compatible(init_dist, nominal_trajectory)

    def body(carry, inp):
        x, ell = carry
        y, predict_ref, update_ref = inp

        if predict_ref is None:
            predict_ref = x

        F_x, b, Q = linearization_method(transition_model, predict_ref)
        x = _standard_predict(F_x, b, Q, x)

        if update_ref is None:
            update_ref = x

        H_x, c, R = linearization_method(observation_model, update_ref)
        x, _ell = _standard_update(H_x, c, R, x, y)
        return (x, ell + _ell), x

    predict_traj = none_or_shift(nominal_trajectory, -1)
    update_traj = none_or_shift(nominal_trajectory, 1)

    x0 = init_dist

    (_, ell), Xs = jax.lax.scan(body, (x0, 0.), (observations, predict_traj, update_traj))
    xs = none_or_concat(Xs, x0, 1)

    if return_loglikelihood:
        return Xs, ell
    else:
        return Xs


def _standard_predict(F, b, Q, x):
    m, P = x
    m = F @ m + b
    P = Q + F @ P @ F.T
    return MVNStandard(m, P)


def _standard_update(H, c, R, x, y):
    m, P = x

    y_hat = H @ m + c
    y_diff = y - y_hat
    S = R + H @ P @ H.T
    chol_S = jnp.linalg.cholesky(S)
    G = P @ cho_solve((chol_S, True), H).T

    m = m + G @ y_diff
    P = P - G @ S @ G.T
    ell = mvn_logpdf_chol(y, y_hat, chol_S)
    return MVNStandard(m, P), ell
