from typing import Tuple, Any, Union, Callable

import jax
import jax.numpy as jnp

from smoothers import FunctionalModel, MVNStandard
from smoothers.base import are_inputs_compatible


def quadratize(model: Union[FunctionalModel, Callable],
               x: Union[MVNStandard, jnp.ndarray]):
    """
    Quadratization for a nonlinear function f(x, q).
    If the function is linear, JAX Jacobian calculation will
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
        Linearization parameters
    F_xx: jnp.ndarray
        The hessian
    cov_q: jnp.ndarray
        The covariance matrix
    """
    if isinstance(model, FunctionalModel):
        f, q = model
        are_inputs_compatible(x, q)

        m_x, _ = x
        return _standard_second_order_callable(f, m_x, *q)
    else:
        return _second_order_callable_common(model, x)


def _second_order_callable_common(f, x) -> Tuple[Any, Any, Any]:
    return f(x), jax.jacfwd(f, 0)(x), jax.jacfwd(jax.jacrev(f))(x)


def _standard_second_order_callable(f, x, m_q, cov_q):
    res, F_x, F_xx = _second_order_callable_common(f, x)
    return F_xx, F_x, res - F_x @ x + m_q, cov_q
