from typing import Any, Tuple

import jax
import jax.numpy as jnp

from parsmooth._base import FunctionalModel, are_inputs_compatible, MVNStandard


def linearize(model: FunctionalModel, x: MVNStandard):
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
        Either the cholesky or the full-rank modified covariance matrix
    F_xx: jnp.ndarray
        The hessian
    """
    f, q = model
    are_inputs_compatible(x, q)

    m_x, _ = x
    return _standard_linearize_callable(f, m_x, *q)


def _linearize_callable_common(f, x) -> Tuple[Any, Any, Any]:
    return f(x), jax.jacfwd(f, 0)(x), jax.jacfwd(jax.jacrev(f))(x)


def _standard_linearize_callable(f, x, m_q, cov_q):
    res, F_x, F_xx = _linearize_callable_common(f, x)
    return F_x, cov_q, res - F_x @ x + m_q, F_xx
