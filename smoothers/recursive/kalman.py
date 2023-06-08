import jax
from jax import numpy as jnp

from smoothers.base import MVNStandard
from smoothers.base import LinearTransition, LinearObservation
from smoothers.utils import none_or_concat, none_or_shift


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
        G = jnp.linalg.solve(S.T, H @ P.T).T

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


def smoothing(
    linear_transition: LinearTransition,
    filter_trajectory: MVNStandard,
):
    def _smooth(F, b, Q, xf, xs):
        mf, Pf = xf
        ms, Ps = xs

        mean_diff = ms - (b + F @ mf)
        S = F @ Pf @ F.T + Q
        cov_diff = Ps - S

        gain = Pf @ jnp.linalg.solve(S, F).T
        ms = mf + gain @ mean_diff
        Ps = Pf + gain @ cov_diff @ gain.T
        return MVNStandard(ms, Ps)

    def body(carry, args):
        xs = carry
        xf, lt = args

        F, b, Q = lt
        xs = _smooth(F, b, Q, xf, xs)
        return xs, xs

    xl = jax.tree_map(lambda z: z[-1], filter_trajectory)

    Xf = none_or_shift(filter_trajectory, -1)
    lts = linear_transition

    _, Xs = jax.lax.scan(
        body,
        xl,
        (Xf, lts),
        reverse=True,
    )

    Xs = none_or_concat(Xs, xl, -1)
    return Xs
