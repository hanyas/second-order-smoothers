import jax
from jax import numpy as jnp

from smoothers import MVNStandard
from smoothers.base import LinearTransition
from smoothers.utils import none_or_shift, none_or_concat


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
