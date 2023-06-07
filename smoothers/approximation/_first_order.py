from typing import Any, Tuple, Union, Callable

import jax
import jax.numpy as jnp

from smoothers.base import MVNStandard, FunctionalModel


def linearize(model: Union[FunctionalModel, Callable],
              x: Union[MVNStandard, jnp.ndarray]):
    """
    Linearization for a nonlinear function f(x, q).
    If the function is linear, JAX Jacobian calculation will
    simply return the matrices without additional complexity.
    """

    if isinstance(model, FunctionalModel) and isinstance(x, MVNStandard):
        f, q = model
        m_x, _ = x
        return _linearize_distribution(f, m_x, *q)
    elif isinstance(model, Callable) and isinstance(x, jnp.ndarray):
        return _linearize_callable(model, x)
    else:
        raise NotImplementedError


def _linearize_callable(f, x) -> Tuple[Any, Any]:
    return f(x), jax.jacfwd(f, 0)(x)


def _linearize_distribution(f, x, m_q, cov_q):
    f0, F_x = _linearize_callable(f, x)
    return F_x, f0 - F_x @ x + m_q, cov_q
