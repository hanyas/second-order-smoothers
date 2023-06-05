from typing import Any, Tuple, Union, Callable

import jax
import jax.numpy as jnp

from smoothers.base import MVNStandard, FunctionalModel
from smoothers.base import are_inputs_compatible


def linearize(model: Union[FunctionalModel, Callable],
              x: Union[MVNStandard, jnp.ndarray]):
    """
    Extended linearization for a non-linear function f(x, q). If the function is linear, JAX Jacobian calculation will
    simply return the matrices without additional complexity.

    Parameters
    ----------
    model: FunctionalModel
        The function to be called on x and q
    x: MVNStandard
        x-coordinate state at which to linearize f

    Returns
    -------
    F_x, F_q, res: jnp.ndarray
        The linearization parameters
    cov_q: jnp.ndarray
        The covariance matrix.
    """
    if isinstance(model, FunctionalModel):
        f, q = model
        are_inputs_compatible(x, q)

        m_x, _ = x
        return _standard_linearize_callable(f, m_x, *q)
    else:
        return _linearize_callable_common(model, x)


def _linearize_callable_common(f, x) -> Tuple[Any, Any]:
    return f(x), jax.jacfwd(f, 0)(x)


def _standard_linearize_callable(f, x, m_q, cov_q):
    res, F_x = _linearize_callable_common(f, x)
    return F_x, res - F_x @ x + m_q, cov_q
