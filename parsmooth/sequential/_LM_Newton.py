from typing import Callable, Optional

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg
from jax import jacfwd, jacrev
from jax.experimental.host_callback import id_print
from jax.scipy.linalg import cho_solve

from parsmooth._base import MVNStandard, FunctionalModel, are_inputs_compatible
from parsmooth._utils import none_or_shift, none_or_concat
from parsmooth.sequential._filtering_Newton import filtering as newton_filtering
from parsmooth.sequential._smoothing_Newton import smoothing as newton_smoothing


def mvn_loglikelihood(x, chol_cov):
    """multivariate normal"""
    dim = chol_cov.shape[0]
    y = jlinalg.solve_triangular(chol_cov, x, lower=True)
    normalizing_constant = (
            jnp.sum(jnp.log(jnp.abs(jnp.diag(chol_cov)))) + dim * jnp.log(2 * jnp.pi) / 2.0
    )
    norm_y = jnp.sum(y * y, -1)
    return -0.5 * norm_y - normalizing_constant

def L(predict_trajectory, update_trajectory, z, measurement_fun, dynamic_fun, chol_Q, chol_R):
    mp_nominal = predict_trajectory
    mu_nominal = update_trajectory
    cost = mvn_loglikelihood(mu_nominal - dynamic_fun(mp_nominal), chol_Q) + mvn_loglikelihood(z - measurement_fun(mu_nominal), chol_R)
    return -cost

def state_space_cost(x, ys, observation_function, transition_function, Q, R, m0, P0):
    x0 = x[0]
    predict_traj =x[:-1]
    update_traj = x[1:]
    vmapped_fun = jax.vmap(L, in_axes=[0, 0, 0, None, None, None, None])
    return jnp.sum(vmapped_fun(predict_traj, update_traj, ys, observation_function, transition_function, jnp.linalg.cholesky(Q), jnp.linalg.cholesky(R))) - mvn_loglikelihood(x0 - m0, jnp.linalg.cholesky(P0))


def LM_Newton(current_nomminal_trajectory: Optional[MVNStandard],
              cost_function: Callable,
              nu: jnp.ndarray,
              lam: jnp.ndarray, ys, observation_model, transition_model, init, extended_hessian, extended):
    transition_function, Q = transition_model
    observation_function, R = observation_model
    m0, P0 = init


    nominal_trajectory = current_nomminal_trajectory
    J1 = state_space_cost(nominal_trajectory, ys, observation_function, transition_function, Q, R, m0, P0)

    # filtered_newton = newton_filtering_lam(ys, init, transition_model, observation_model, extended_hessian, nominal_trajectory, True)
    # smoothed_newton = newton_smoothing(transition_model, filtered_newton, extended, nominal_trajectory)
    #
    # J2 =
    # pred =
    #
    # def true_fun():
    #
    #     return
    #
    # def false_fun():
    #
    #     return
    #
    new_nominal_trajectory = jax.lax.cond(pred, true_fun, false_fun)

    return new_nominal_trajectory