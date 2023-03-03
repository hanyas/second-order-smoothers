from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy as jsc
from jax.flatten_util import ravel_pytree

from optsmooth._base import MVNStandard, FunctionalModel
from optsmooth._utils import mvn_logpdf


def _gauss_newton_step(
    jac: jnp.ndarray, res: jnp.ndarray, ws: jnp.ndarray, lmbda: float
):
    J, r, W = jac, res, ws

    d = J.shape[-1]
    grad = jnp.dot(J.T @ W, r)
    hess_reg = J.T @ W @ J + lmbda * jnp.eye(d)

    dx = -jnp.linalg.solve(hess_reg, grad)
    df = -jnp.dot(dx, grad) - 0.5 * jnp.dot(jnp.dot(dx, hess_reg), dx)
    return dx, df


def _trust_region_gauss_newton_step(
    fun: Callable,
    jac: jnp.ndarray,
    res: jnp.ndarray,
    ws: jnp.ndarray,
    x0: jnp.ndarray,
    lmbda: float,
    nu: float,
):
    dx, df = _gauss_newton_step(jac, res, ws, lmbda)
    xn = x0 + dx

    f0, fn = fun(x0), fun(xn)
    rho = (f0 - fn) / df

    def accept(args):
        _, _, xn, fn, lmbda, nu = args
        lmbda = lmbda * jnp.maximum(1.0 / 3.0, 1.0 - (2.0 * rho - 1) ** 3)
        lmbda = jnp.maximum(1e-16, lmbda)
        return xn, fn, lmbda, 2.0

    def reject(args):
        x0, f0, _, _, lmbda, nu = args
        lmbda = jnp.minimum(1e16, lmbda)
        return x0, f0, lmbda * nu, 2.0 * nu

    xn, fn, lmbda, nu = jax.lax.cond(
        (rho > 0.0) & (df > 0.0),
        accept,
        reject,
        operand=(x0, f0, xn, fn, lmbda, nu),
    )
    return xn, lmbda, nu, fn


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


def trust_region_iterated_batch_gauss_newton_smoother(
    observations: jnp.ndarray,
    initial_dist: MVNStandard,
    transition_model: FunctionalModel,
    observation_model: FunctionalModel,
    linearization_method: Callable,
    init_nominal_mean: jnp.ndarray,
    lmbda: float = 1e2,
    nu: float = 2.0,
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

    def body(carry, _):
        flat_nominal_mean, lmbda, nu = carry

        res = _flat_residual(flat_nominal_mean)
        jac = jax.jacobian(_flat_residual)(flat_nominal_mean)

        flat_nominal_mean, lmbda, nu, cost = _trust_region_gauss_newton_step(
            fun=_flat_log_posterior,
            jac=jac,
            res=res,
            ws=weights,
            x0=flat_nominal_mean,
            lmbda=lmbda,
            nu=nu,
        )
        return (flat_nominal_mean, lmbda, nu), cost

    (flat_nominal_mean, lmbda, _), costs = jax.lax.scan(
        body, (flat_nominal_mean, lmbda, nu), jnp.arange(nb_iter)
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
