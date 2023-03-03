import sys
from functools import partial

import numpy as np

sys.path.append('..//')

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')

from parsmooth._base import MVNStandard, FunctionalModel
from parsmooth.linearization import extended, second_order
from parsmooth.sequential._ls_newton import _iterated_recursive_newton_smoother
from parsmooth.sequential._ls_gn import _iterated_recursive_gauss_newton_smoother


def dynamic_model(xk):
    return jnp.tanh(xk)


def measurement_model(xk):
    return jnp.sin(xk)


T = 60
xs = np.zeros((T+1, 1))
ys = np.zeros((T, 1))
sigma = 0.01
x0 = np.random.normal(loc=0, scale=1)
x = x0
xs[0] = x
for k in range(T):
    q = np.random.normal(loc=0, scale=sigma)
    r = np.random.normal(loc=0, scale=sigma)
    x = dynamic_model(x) + q
    y = measurement_model(x) + r
    xs[k+1] = x
    ys[k] = y

Q = jnp.array([[sigma**2]])
R = jnp.array([[sigma**2]])
initial_states = MVNStandard(jnp.repeat(jnp.array([[3.]]),T + 1, axis=0),
                                                     jnp.repeat(jnp.eye(1).reshape(1, 1, 1), T + 1, axis=0))
transition_model = FunctionalModel(partial(dynamic_model), MVNStandard(jnp.zeros((1,)), Q))
observation_model = FunctionalModel(partial(measurement_model), MVNStandard(jnp.zeros((1,)), R))

m0 = jnp.array([0.])
P0 = jnp.diag(jnp.array([0.1]))
init = MVNStandard(m0, P0)

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
                                                         transition_model, observation_model, extended, nominal_trajectory,
                                                                                n_iter=iteration)

    return costs


newton_costs = iterated_recursive_newton_smoother(25, initial_states)
gauss_newton_costs = iterated_recursive_gauss_newton_smoother(25, initial_states)


plt.figure()
plt.subplot(2, 2, 1)
xk = jnp.arange(-10,10)
plt.plot(xk, dynamic_model(xk))
plt.title("dynamic_model")
plt.subplot(2, 2, 2)
xk = jnp.arange(-10, 10)
plt.plot(xk, measurement_model(xk))
plt.title("measurement_model")
plt.subplot(2, 2, (3,4))
plt.plot(newton_costs, "*--", label="newton cost")
plt.plot(gauss_newton_costs, ':', label="gauss-newton cost")
plt.legend()
plt.grid()
plt.ylabel("cost value")
plt.xlabel("iteration")
plt.show()
