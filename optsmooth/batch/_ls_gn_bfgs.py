from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy as jsc

from jax.flatten_util import ravel_pytree

from jaxopt import BacktrackingLineSearch

from optsmooth._base import MVNStandard, FunctionalModel
from optsmooth._utils import mvn_logpdf


def _gn_bfgs_hess_update(S, s, yd, yc):
    aux = (jnp.outer(yc - S @ s, yd) + jnp.outer(yd, yc - S @ s)) / jnp.dot(yd, s)\
          - jnp.outer(yc - S @ s, s) @ jnp.outer(yd, yd) / jnp.dot(yd, s) ** 2
    tau = jnp.minimum(1.0, jnp.abs(jnp.dot(s, yc)) / jnp.abs(jnp.dot(s, jnp.dot(S, s))))
    return tau * S + aux


def _gn_bfgs_step(
    jac: jnp.ndarray, rsd: jnp.ndarray, ws: jnp.ndarray, bfgs_hess: jnp.ndarray
):
    J, r, W = jac, rsd, ws

    grad = jnp.dot(J.T @ W, r)
    hess = J.T @ W @ J + bfgs_hess
    dx = -jnp.linalg.solve(hess, grad)
    return dx


def _line_search_gn_bfgs_step(
    obj_fun: Callable,
    rsd_fun: Callable,
    jac: jnp.ndarray,
    rsd: jnp.ndarray,
    bfgs_hess: jnp.ndarray,
    ws: jnp.ndarray,
    x0: jnp.ndarray,
):
    # GN-BFGS direction update
    dx = _gn_bfgs_step(jac, rsd, ws, bfgs_hess)

    ls = BacktrackingLineSearch(fun=obj_fun, maxiter=100)

    alpha, _ = ls.run(
        init_stepsize=1.0,
        params=x0,
        descent_direction=dx,
    )
    xn = x0 + alpha * dx
    fn = obj_fun(xn)

    # GN-BFGS hessian update
    Jp, rp, W = jac, rsd, ws

    rn = rsd_fun(xn)
    Jn = jax.jacobian(rsd_fun)(xn)

    yd = jnp.dot(Jn.T @ W, rn) - jnp.dot(Jp.T @ W, rp)
    yc = jnp.dot(Jn.T @ W, rn) - jnp.dot(Jp.T @ W, rn)

    next_bfgs_hess = _gn_bfgs_hess_update(bfgs_hess, alpha * dx, yd, yc)
    return xn, Jn, rn, next_bfgs_hess, fn


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


def line_search_iterated_batch_gn_bfgs_smoother(
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

    init_rsd = _flat_residual(flat_nominal_mean)
    init_jac = jax.jacobian(_flat_residual)(flat_nominal_mean)
    init_bfgs_hess = 1e-6 * jnp.eye(flat_nominal_mean.shape[0])

    def body(carry, _):
        flat_nominal_mean, jac, rsd, bfgs_hess = carry

        (
            flat_nominal_mean,
            jac,
            rsd,
            bfgs_hess,
            cost,
        ) = _line_search_gn_bfgs_step(
            obj_fun=_flat_log_posterior,
            rsd_fun=_flat_residual,
            jac=jac,
            rsd=rsd,
            bfgs_hess=bfgs_hess,
            ws=weights,
            x0=flat_nominal_mean,
        )
        return (flat_nominal_mean, jac, rsd, bfgs_hess), cost

    (flat_nominal_mean, _, _, _), costs = jax.lax.scan(
        body, (flat_nominal_mean, init_jac, init_rsd, init_bfgs_hess), jnp.arange(nb_iter)
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
