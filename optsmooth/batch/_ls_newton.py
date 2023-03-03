from typing import Callable

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from jaxopt import BacktrackingLineSearch

from optsmooth._base import MVNStandard, FunctionalModel
from optsmooth._utils import mvn_logpdf


def _modify_direction(args):
    grad, hess, _ = args

    def cond(carry):
        lmbda = carry

        d = hess.shape[0]
        hess_reg = hess + lmbda * jnp.eye(d)
        dx = -jnp.linalg.solve(hess_reg, grad)
        df = -jnp.dot(dx, grad) - 0.5 * jnp.dot(jnp.dot(dx, hess_reg), dx)
        return df <= 0.0

    def body(carry):
        lmbda = carry
        lmbda = lmbda * 10.0
        return lmbda

    lmbda = jax.lax.while_loop(
        cond,
        body,
        1e-8,
    )

    d = hess.shape[0]
    hess_reg = hess + lmbda * jnp.eye(d)
    dx = -jnp.linalg.solve(hess_reg, grad)
    return dx


def _keep_direction(args):
    _, _, dx = args
    return dx


def _newton_step(fun: Callable, x0: jnp.ndarray):
    grad = jax.grad(fun)(x0)
    hess = jax.hessian(fun)(x0)

    d = hess.shape[0]
    dx = -jnp.linalg.solve(hess, grad)
    df = -jnp.dot(dx, grad) - 0.5 * jnp.dot(jnp.dot(dx, hess), dx)

    # modify direciton if necessary
    dx = jax.lax.cond(
        df > 0.0, _keep_direction, _modify_direction, (grad, hess, dx)
    )
    return dx


def _line_search_newton_step(fun: Callable, x0: jnp.ndarray):
    dx = _newton_step(fun, x0)

    ls = BacktrackingLineSearch(fun=fun, maxiter=100)

    alpha, _ = ls.run(
        init_stepsize=1.0,
        params=x0,
        descent_direction=dx,
    )
    xn = x0 + alpha * dx
    fn = fun(xn)
    return xn, fn


def log_posterior(
    states: jnp.ndarray,
    observations: jnp.ndarray,
    initial_dist: MVNStandard,
    transition_model: FunctionalModel,
    observation_model: FunctionalModel,
):
    xp, xn = states[:-1], states[1:]
    yn = observations

    m0, P0 = initial_dist
    f, (_, Q) = transition_model
    h, (_, R) = observation_model

    xn_mu = jax.vmap(f)(xp)
    yn_mu = jax.vmap(h)(xn)

    cost = -mvn_logpdf(states[0], m0, P0)
    cost -= jnp.sum(jax.vmap(mvn_logpdf, in_axes=(0, 0, None))(xn, xn_mu, Q))
    cost -= jnp.sum(jax.vmap(mvn_logpdf, in_axes=(0, 0, None))(yn, yn_mu, R))
    return cost


def line_search_iterated_batch_newton_smoother(
    observations: jnp.ndarray,
    initial_dist: MVNStandard,
    transition_model: FunctionalModel,
    observation_model: FunctionalModel,
    quadratization_method: Callable,
    init_nominal_mean: jnp.ndarray,
    nb_iter: int = 10,
):
    flat_nominal_mean, _unflatten = ravel_pytree(init_nominal_mean)

    def _flat_log_posterior(flat_state):
        _state = _unflatten(flat_state)
        return log_posterior(
            _state,
            observations,
            initial_dist,
            transition_model,
            observation_model,
        )

    init_cost = _flat_log_posterior(flat_nominal_mean)

    def body(carry, _):
        flat_nominal_mean = carry
        flat_nominal_mean, cost = _line_search_newton_step(
            fun=_flat_log_posterior, x0=flat_nominal_mean
        )
        return flat_nominal_mean, cost

    flat_nominal_mean, costs = jax.lax.scan(
        body, flat_nominal_mean, jnp.arange(nb_iter)
    )

    nominal_mean = _unflatten(flat_nominal_mean)
    return nominal_mean, jnp.hstack((init_cost, costs))


def _build_grad_and_hess(
    observations: jnp.ndarray,
    initial_dist: MVNStandard,
    transition_model: FunctionalModel,
    observation_model: FunctionalModel,
    quadratization_method: Callable,
    nominal_mean: jnp.ndarray,
):
    y = observations
    m0, P0 = initial_dist

    f, (_, Q) = transition_model
    h, (_, R) = observation_model

    m_curr = nominal_mean[:-1]
    m_next = nominal_mean[1:]

    f0, F_x, F_xx = jax.vmap(quadratization_method, in_axes=(None, 0))(f, m_curr)
    h0, H_x, H_xx = jax.vmap(quadratization_method, in_axes=(None, 0))(h, m_next)

    T = nominal_mean.shape[0]
    nx = Q.shape[-1]
    ny = R.shape[-1]

    def _dynamics_hessian(F_xx, m_next, m_curr):
        return -jnp.einsum("ijk,k->ij", F_xx.T, jnp.linalg.solve(Q, m_next - f(m_curr)))

    def _observation_hessian(H_xx, y, m_next):
        return -jnp.einsum("ijk,k->ij", H_xx.T, jnp.linalg.solve(R, y - h(m_next)))

    Phi = jax.vmap(_dynamics_hessian)(F_xx, m_next, m_curr)
    Gamma = jax.vmap(_observation_hessian)(H_xx, y, m_next)

    P0_inv = jnp.linalg.inv(P0)
    Q_inv = jnp.linalg.inv(Q)
    R_inv = jnp.linalg.inv(R)

    hess = jnp.zeros((T * nx, T * nx))

    # off-diagonal
    def _off_diagonal(carry, args):
        F_x, t = args
        hess = carry

        val = - Q_inv @ F_x
        hess = jax.lax.dynamic_update_slice(hess, val, ((t + 1) * nx, t * nx))
        hess = jax.lax.dynamic_update_slice(hess, val.T, (t * nx, (t + 1) * nx))
        return hess, _

    hess, _ = jax.lax.scan(_off_diagonal, hess, (F_x, jnp.arange(T - 1)))

    # first diagonal
    val = P0_inv + F_x[0].T @ Q_inv @ F_x[0] + Phi[0]
    hess = hess.at[:nx, :nx].set(val)

    # intermediate diagonal
    def _diagonal(carry, args):
        F_x, H_x, Phi, Gamma, t = args
        hess = carry

        val = Q_inv + F_x.T @ Q_inv @ F_x + H_x.T @ R_inv @ H_x + Phi + Gamma
        hess = jax.lax.dynamic_update_slice(hess, val, (t * nx, t * nx))
        return hess, _

    hess, _ = jax.lax.scan(
        _diagonal,
        hess,
        (
            F_x[1:],
            H_x[:-1],
            Phi[1:],
            Gamma[:-1],
            jnp.arange(1, T - 1)),
    )

    # last diagonal
    val = Q_inv + H_x[-1].T @ R_inv @ H_x[-1] + Gamma[-1]
    hess = hess.at[-nx:, -nx:].set(val)

    # grad vector
    grad = jnp.zeros((T * nx,))

    m = nominal_mean

    # first vector
    val = P0_inv @ (m[0] - m0) - F_x[0].T @ Q_inv @ (m[1] - f(m[0]))
    grad = grad.at[:nx].set(val)

    # intermediate vectors
    def _intermediate(carry, args):
        y, m_prev, m_curr, m_next, F_x, H_x, t = args
        grad = carry

        val = (
            Q_inv @ (m_curr - f(m_prev))
            - F_x.T @ Q_inv @ (m_next - f(m_curr))
            - H_x.T @ R_inv @ (y - h(m_curr))
        )
        grad = jax.lax.dynamic_update_slice(grad, val, (t * nx,))
        return grad, _

    m_prev = nominal_mean[:-2]
    m_curr = nominal_mean[1:-1]
    m_next = nominal_mean[2:]

    grad, _ = jax.lax.scan(
        _intermediate,
        grad,
        (
            y[:-1],
            m_prev,
            m_curr,
            m_next,
            F_x[1:],
            H_x[:-1],
            jnp.arange(1, T - 1),
        ),
    )

    # last vector
    val = Q_inv @ (m[-1] - f(m[-2])) - H_x[-1].T @ R_inv @ (y[-1] - h(m[-1]))
    grad = grad.at[-nx:].set(val)

    return grad, hess
