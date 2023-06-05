from typing import Optional, Callable

import jax
import jax.numpy as jnp

from smoothers.base import MVNStandard, are_inputs_compatible, FunctionalModel
from smoothers.utils import none_or_shift, none_or_concat


def smoothing(
    transition_model: FunctionalModel,
    filter_traj: MVNStandard,
    linearization_method: Callable,
    nominal_traj: Optional[MVNStandard] = None,
):
    last_state = jax.tree_map(lambda z: z[-1], filter_traj)

    if nominal_traj is not None:
        are_inputs_compatible(filter_traj, nominal_traj)

    def body(smoothed, inputs):
        filtered, ref = inputs
        if ref is None:
            ref = smoothed

        F_x, b, Q = linearization_method(transition_model, ref)
        smoothed_state = _smooth(F_x, b, Q, filtered, smoothed)

        return smoothed_state, smoothed_state

    _, smoothed_states = jax.lax.scan(
        body,
        last_state,
        [
            none_or_shift(filter_traj, -1),
            none_or_shift(nominal_traj, -1),
        ],
        reverse=True,
    )

    smoothed_states = none_or_concat(smoothed_states, last_state, -1)
    return smoothed_states


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
