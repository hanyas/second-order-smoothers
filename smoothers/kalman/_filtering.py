import jax
from jax import numpy as jnp, scipy as jsc

from smoothers import MVNStandard
from smoothers.base import LinearTransition, LinearObservation
from smoothers.utils import none_or_concat


def filtering(
    observations: jnp.ndarray,
    init_dist: MVNStandard,
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
        chol_S = jnp.linalg.cholesky(S)
        G = P @ jsc.linalg.cho_solve((chol_S, True), H).T

        y_hat = H @ m + c
        y_diff = y - y_hat
        m = m + G @ y_diff
        P = P - G @ S @ G.T
        return MVNStandard(m, P)

    def body(carry, args):
        xf = carry
        y, lt, lo = args

        F, b, Q = lt
        H, c, R = lo

        xp = _predict(F, b, Q, xf)
        xf = _update(H, c, R, xp, y)
        return xf, xf

    x0 = init_dist
    ys = observations

    lts = linear_transition
    los = linear_observation

    _, Xf = jax.lax.scan(body, x0, (ys, lts, los))
    return none_or_concat(Xf, x0, 1)
