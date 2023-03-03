import sys
sys.path.append('..//')

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')

from parsmooth._base import MVNStandard, FunctionalModel
from parsmooth.linearization import extended, second_order
from bearing_data import get_data, make_parameters
from parsmooth.sequential._ls_newton import _iterated_recursive_newton_smoother
from parsmooth.sequential._ls_gn import _iterated_recursive_gauss_newton_smoother
from parsmooth.sequential._ls_newton import log_posterior

s1 = jnp.array([-1.5, 0.5])  # First sensor location
s2 = jnp.array([1., 1.])  # Second sensor location
r = 0.3  # Observation noise (stddev)
x0 = jnp.array([0.1, 0.2, 1, 0])  # initial true location
dt = 0.01  # discretization time step
qc = 0.1  # discretization noise
qw = 0.1  # discretization noise
T = 500
Q, R, transition_function, observation_function, _, _ = make_parameters(qc, qw, r, dt, s1, s2)
_, true_states, ys = get_data(x0, dt, r, T, s1, s2, random_state=13)

nx = 5
init = MVNStandard(mean=jnp.array([-1., -1., 0., 0., 0.]),
                   cov=jnp.eye(nx))


key = jax.random.PRNGKey(2)

initial_states =  MVNStandard(mean = 0.001*jax.random.normal(key, shape=(T + 1, 5)),
                              cov = jnp.repeat(jnp.eye(5)[None, ...], T + 1, axis=0))
transition_model = FunctionalModel(transition_function, MVNStandard(jnp.zeros((5,)), Q))
observation_model = FunctionalModel(observation_function, MVNStandard(jnp.zeros((2,)), R))


# Newton Recursive Iterated Smoother
def iterated_recursive_newton_smoother(iteration, nominal_trajectory):
    _, costs = _iterated_recursive_newton_smoother(ys, init,transition_model,
                                                                    observation_model,second_order,
                                                                    extended, nominal_trajectory
                                                                    ,n_iter=iteration)
    return costs

# Gauss-Newton recursive Iterated Smoother
def  iterated_recursive_gauss_newton_smoother(iteration, nominal_trajectory):
    _, costs = _iterated_recursive_gauss_newton_smoother(ys, init,
                                                         transition_model, observation_model,extended, nominal_trajectory,
                                                                                n_iter=iteration)

    return costs


newton_costs = iterated_recursive_newton_smoother(25, initial_states)
gauss_newton_costs = iterated_recursive_gauss_newton_smoother(25, initial_states)

plt.plot( newton_costs, "*--", label="newton cost")
plt.plot(gauss_newton_costs, ':', label="gauss-newton cost")
plt.legend()
plt.grid()
plt.ylabel("cost value")
plt.xlabel("iteration")
plt.show()