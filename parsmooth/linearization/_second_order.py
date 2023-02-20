from typing import Tuple, Any

import jax

from parsmooth import FunctionalModel, MVNStandard
from parsmooth._base import are_inputs_compatible


def second_order(model: FunctionalModel, x: MVNStandard):
    """
    Quadratization for a non-linear function f(x, q).
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
        The linearization parameters
    cov_q: jnp.ndarray
        Either the cholesky or the full-rank modified covariance matrix
    F_xx: jnp.ndarray
        The hessian
    """
    f, q = model

    if isinstance(x, MVNStandard):
        m_x, _ = x
        return _standard_second_order_callable(f, m_x, *q)
    else:
        return _second_order_callable_common(f, x)


def _second_order_callable_common(f, x) -> Tuple[Any, Any, Any]:
    return f(x), jax.jacfwd(f, 0)(x), jax.jacfwd(jax.jacrev(f))(x)


def _standard_second_order_callable(f, x, m_q, cov_q):
    res, F_x, F_xx = _second_order_callable_common(f, x)
    return res - F_x @ x + m_q, F_x, F_xx, cov_q
