from typing import Any, Tuple, Union, Callable, Optional

import jax
import jax.numpy as jnp

from newton_smoothers.base import MVNStandard, FunctionalModel


def linearize(
    model: Union[FunctionalModel, Callable],
    x: Union[MVNStandard, jnp.ndarray],
    mode: Optional[Any] = "linear_coeff",
):
    """
    Linearization for a nonlinear function f(x) + q, q ~ N(0, Q).
    If the function is linear, JAX Jacobian calculation will
    simply return the matrices without additional complexity.
    """

    if isinstance(model, FunctionalModel) and isinstance(x, MVNStandard):
        f, (_, cov_q) = model
        m_x, _ = x
        return _linearize_distribution(f, m_x, cov_q, mode)
    elif isinstance(model, Callable) and isinstance(x, jnp.ndarray):
        return _linearize_mean(model, x, mode)
    else:
        raise NotImplementedError


def _first_order_taylor_coeff(f, x) -> Tuple[Any, Any]:
    return f(x), jax.jacfwd(f, 0)(x)


def _linearize_distribution(f, x, cov_q, mode):
    f0, F_x = _first_order_taylor_coeff(f, x)
    if mode == "taylor_coeff":
        return F_x, f0, cov_q
    elif mode == "linear_coeff":
        return F_x, f0 - F_x @ x, cov_q
    else:
        raise NotImplementedError


def _linearize_mean(f, x,  mode):
    f0, F_x = _first_order_taylor_coeff(f, x)
    if mode == "taylor_coeff":
        return F_x, f0
    elif mode == "linear_coeff":
        return F_x, f0 - F_x @ x
    else:
        raise NotImplementedError
