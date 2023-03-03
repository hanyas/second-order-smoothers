from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy as jsc
from jax.flatten_util import ravel_pytree

from jaxopt import BacktrackingLineSearch

from optsmooth._base import MVNStandard, FunctionalModel
from optsmooth._utils import mvn_logpdf


def _bfgs_hess_update(B, s, y):
    # Implements a damped BFGS, see Nocedal et al.

    def accept(args):
        B, s, y = args
        return 1.0

    def reject(args):
        B, s, y = args
        return (
            0.8
            * jnp.dot(jnp.dot(s, B), s)
            / (jnp.dot(jnp.dot(s, B), s) - jnp.dot(s, y))
        )

    a = jax.lax.cond(
        jnp.dot(s, y) >= 0.2 * jnp.dot(jnp.dot(s, B), s),
        accept,
        reject,
        (B, s, y),
    )
    r = a * y + (1.0 - a) * B @ s
    return (
        B
        + jnp.outer(r, r) / jnp.dot(s, r)
        - jnp.outer(B @ s, B @ s) / jnp.dot(s, B @ s)
    )


def _bfgs_step(grad: jnp.ndarray, approx_hess: jnp.ndarray):
    dx = -jnp.linalg.solve(approx_hess, grad)
    return dx


def _line_search_bfgs_step(
    fun: Callable,
    grad: jnp.ndarray,
    approx_hess: jnp.ndarray,
    x0: jnp.ndarray,
):
    # BFGS direction update
    dx = _bfgs_step(grad, approx_hess)

    ls = BacktrackingLineSearch(fun=fun, maxiter=100)

    alpha, _ = ls.run(
        init_stepsize=1.0,
        params=x0,
        descent_direction=dx,
    )
    xn = x0 + alpha * dx
    fn = fun(xn)

    # BFGS hessian update
    next_grad = jax.grad(fun)(xn)
    next_hess = _bfgs_hess_update(approx_hess, alpha * dx, next_grad - grad)

    return xn, next_grad, next_hess, fn


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


def line_search_iterated_batch_bfgs_smoother(
    observations: jnp.ndarray,
    initial_dist: MVNStandard,
    transition_model: FunctionalModel,
    observation_model: FunctionalModel,
    linearization_method: Callable,
    init_nominal_mean: jnp.ndarray,
    nb_iter: int = 10,
):
    flat_nominal_mean, _unflatten = ravel_pytree(init_nominal_mean)

    weights = _blkmatrix(
        init_nominal_mean,
        observations,
        initial_dist,
        transition_model,
        observation_model,
    )

    def _flat_log_posterior(flat_state):
        _state = _unflatten(flat_state)
        return log_posterior(
            _state,
            observations,
            initial_dist,
            transition_model,
            observation_model,
        )

    def _flat_residual(flat_state):
        _state = _unflatten(flat_state)
        return _residual(
            _state,
            observations,
            initial_dist,
            transition_model,
            observation_model,
        )

    def _flat_approx_log_posterior(flat_state, flat_nominal):
        r0 = _flat_residual(flat_nominal)
        J = jax.jacobian(_flat_residual)(flat_nominal)
        r = r0 + jnp.dot(J, flat_state - flat_nominal)
        return jnp.dot(r, jnp.dot(weights, r))

    init_cost = _flat_log_posterior(flat_nominal_mean)

    init_grad = jax.grad(_flat_log_posterior)(flat_nominal_mean)
    _jac = jax.jacobian(_flat_residual)(flat_nominal_mean)
    init_hess = _jac.T @ weights @ _jac

    def body(carry, _):
        flat_nominal_mean, grad, approx_hess = carry
        flat_nominal_mean, grad, approx_hess, cost = _line_search_bfgs_step(
            fun=_flat_log_posterior,
            grad=grad,
            approx_hess=approx_hess,
            x0=flat_nominal_mean,
        )
        return (flat_nominal_mean, grad, approx_hess), cost

    (flat_nominal_mean, _, _), costs = jax.lax.scan(
        body, (flat_nominal_mean, init_grad, init_hess), jnp.arange(nb_iter)
    )

    nominal_mean = _unflatten(flat_nominal_mean)
    return nominal_mean, jnp.hstack((init_cost, costs))


def _residual(
    states: jnp.ndarray,
    observations: jnp.ndarray,
    initial_dist: MVNStandard,
    transition_model: FunctionalModel,
    observation_model: FunctionalModel,
):
    m0, P0 = initial_dist
    f = transition_model.function
    h = observation_model.function

    xp, xn = states[:-1], states[1:]
    yn = observations

    T, nx = states.shape
    _, ny = observations.shape
    n = nx + ny

    r0 = states[0] - m0
    rx = xn - jax.vmap(f)(xp)
    ry = yn - jax.vmap(h)(xn)

    rxy, _ = ravel_pytree(jnp.hstack((rx, ry)))
    r = jnp.hstack((r0, rxy))
    return r


def _blkmatrix(
    states: jnp.ndarray,
    observations: jnp.ndarray,
    initial_dist: MVNStandard,
    transition_model: FunctionalModel,
    observation_model: FunctionalModel,
):
    _, P0 = initial_dist
    _, Q = transition_model.mvn
    _, R = observation_model.mvn
    T, _ = states.shape

    QR = jsc.linalg.block_diag(jnp.linalg.inv(Q), jnp.linalg.inv(R))

    blk_QR = jnp.kron(jnp.eye(T - 1), QR)
    blk_QR = jsc.linalg.block_diag(jnp.linalg.inv(P0), blk_QR)
    return blk_QR
