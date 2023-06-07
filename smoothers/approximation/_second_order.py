from typing import Tuple, Any, Union, Callable

import jax
import jax.numpy as jnp

from smoothers import FunctionalModel, MVNStandard


def quadratize(model: Union[FunctionalModel, Callable],
               x: Union[MVNStandard, jnp.ndarray]):
    """
    Quadratization for a nonlinear function f(x, q).
    If the function is linear, JAX Jacobian calculation will
    simply return the matrices without additional complexity.
    """

    if isinstance(model, FunctionalModel) and isinstance(x, MVNStandard):
        f, q = model
        m_x, _ = x
        return _quadratize_distribution(f, m_x, *q)
    elif isinstance(model, Callable) and isinstance(x, jnp.ndarray):
        return _quadratize_callable(model, x)
    else:
        raise NotImplementedError


def _quadratize_callable(f, x) -> Tuple[Any, Any, Any]:
    return f(x), jax.jacfwd(f, 0)(x), jax.jacfwd(jax.jacrev(f))(x)


def _quadratize_distribution(f, x, m_q, cov_q):
    f0, F_x, F_xx = _quadratize_callable(f, x)
    return F_xx, F_x, f0 - F_x @ x + m_q, cov_q
