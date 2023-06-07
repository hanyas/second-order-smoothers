from typing import Any, Tuple, Union, Callable, Optional

import jax
import jax.numpy as jnp

from smoothers.base import MVNStandard, FunctionalModel


def quadratize(model: Union[FunctionalModel, Callable],
               x: Union[MVNStandard, jnp.ndarray],
               mode: Optional[Any] = "taylor_coeff"):
    """
    Quadratization for a nonlinear function f(x) + q, q ~ N(0, Q).
    If the function is linear, JAX Jacobian calculation will
    simply return the matrices without additional complexity.
    """

    if isinstance(model, FunctionalModel) and isinstance(x, MVNStandard):
        f, (_, cov_q) = model
        m_x, _ = x
        return _quadratize_distribution(f, m_x, cov_q, mode)
    elif isinstance(model, Callable) and isinstance(x, jnp.ndarray):
        return _quadratize_mean(model, x, mode)
    else:
        raise NotImplementedError


def _second_order_taylor_coeff(f, x) -> Tuple[Any, Any, Any]:
    return f(x), jax.jacfwd(f, 0)(x), jax.jacfwd(jax.jacrev(f))(x)


def _quadratize_distribution(f, x, cov_q, mode):
    f0, F_x, F_xx = _second_order_taylor_coeff(f, x)
    if mode == "taylor_coeff":
        return F_xx, F_x, f0, cov_q
    elif mode == "linear_coeff":
        return F_xx, F_x, f0 - F_x @ x, cov_q
    else:
        raise NotImplementedError


def _quadratize_mean(f, x, mode):
    f0, F_x, F_xx = _second_order_taylor_coeff(f, x)
    if mode == "taylor_coeff":
        return F_xx, F_x, f0
    elif mode == "linear_coeff":
        return F_xx, F_x, f0 - F_x @ x
    else:
        raise NotImplementedError
