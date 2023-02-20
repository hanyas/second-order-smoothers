import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve, cho_solve
from jax.flatten_util import ravel_pytree

from jaxopt import BacktrackingLineSearch

from parsmooth._base import MVNStandard, FunctionalModel


logdet = lambda x: jnp.linalg.slogdet(x)[1]


def _batch_iterated_bfgs_smoother(observations: jnp.ndarray, initial_dist: MVNStandard,
                                  transition_model: FunctionalModel, observation_model: FunctionalModel,
                                  nominal_mean: jnp.ndarray, n_iter: int = 10):

    flat_nominal_mean, unravel = ravel_pytree(nominal_mean)

    def _flattend_log_posterior(flat_state):
        _state = unravel(flat_state)
        return log_posterior(_state, observations, initial_dist,
                             transition_model, observation_model)

    T, nx = nominal_mean.shape
    hess = jnp.eye(flat_nominal_mean.shape[0])

    jac = jax.jacobian(_flattend_log_posterior)(flat_nominal_mean)
    init_cost = _flattend_log_posterior(flat_nominal_mean)

    ls = BacktrackingLineSearch(fun=_flattend_log_posterior,
                                maxiter=100, condition="strong-wolfe")

    def _bfgs_update(B, s, y):
        def accept(args):
            B, s, y = args
            return 1.0

        def reject(args):
            B, s, y = args
            return 0.8 * jnp.dot(jnp.dot(s, B), s) / (jnp.dot(jnp.dot(s, B), s) - jnp.dot(s, y))

        a = jax.lax.cond(jnp.dot(s, y) >= 0.2 * jnp.dot(jnp.dot(s, B), s), accept, reject, (B, s, y))
        r = a * y + (1.0 - a) * B @ s
        return B + jnp.outer(r, r) / jnp.dot(s, r) - jnp.outer(B @ s, B @ s) / jnp.dot(s, B @ s)

    def body(carry, _):
        nominal_mean, hess, jac, last_cost = carry

        # update direction
        search_dir = - jnp.linalg.solve(hess, jac)

        # backtracking
        def cond(carry):
            _, backtrack, new_cost = carry
            return jnp.logical_and((new_cost > last_cost), (backtrack < 50))

        def body(carry):
            alpha, backtrack, _ = carry
            alpha = 0.5 * alpha
            backtrack = backtrack + 1
            new_cost = _flattend_log_posterior(nominal_mean + alpha * search_dir)
            return alpha, backtrack, new_cost

        alpha, backtrack = 1.0, 0
        new_cost = _flattend_log_posterior(nominal_mean + alpha * search_dir)
        alpha, _, new_cost = jax.lax.while_loop(cond, body, (alpha, backtrack, new_cost))

        # alpha, state = ls.run(init_stepsize=1.0, params=nominal_mean,
        #                       descent_direction=search_dir)
        # new_cost = _flattend_log_posterior(nominal_mean + alpha * search_dir)

        smoothed_mean = nominal_mean + alpha * search_dir

        # BFGS hessian update
        new_jac = jax.jacobian(_flattend_log_posterior)(smoothed_mean)
        hess = _bfgs_update(hess, alpha * search_dir, new_jac - jac)

        return (smoothed_mean, hess, new_jac, new_cost), new_cost

    (flat_nominal_mean, _, _, _), costs = jax.lax.scan(body, (flat_nominal_mean, hess,
                                                              jac, init_cost),
                                                       jnp.arange(n_iter))

    nominal_mean = unravel(flat_nominal_mean)
    return nominal_mean, jnp.hstack((init_cost, costs))


def log_posterior(states, observations,
                  initial_dist, transition_model, observation_model):

    xp, xn = states[:-1], states[1:]
    yn = observations

    m0, P0 = initial_dist
    f, (_, Q) = transition_model
    h, (_, R) = observation_model

    _xn = jax.vmap(f)(xp)
    _yn = jax.vmap(h)(xn)

    cost = jnp.sum(jax.vmap(mvn_loglikelihood, in_axes=[0, None])(xn - _xn, Q)
                   + jax.vmap(mvn_loglikelihood, in_axes=[0, None])(yn - _yn, R))\
           + mvn_loglikelihood(states[0] - m0, P0)

    return - cost


def mvn_loglikelihood(x, cov):
    dim = cov.shape[0]
    normalizer = logdet(cov) / 2.0 + dim * jnp.log(2.0 * jnp.pi) / 2.0
    norm = jnp.dot(x, solve(cov, x))
    return - 0.5 * norm - normalizer
