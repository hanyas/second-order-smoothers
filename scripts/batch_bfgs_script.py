import jax
import jax.numpy as jnp

from parsmooth import MVNStandard
from parsmooth import FunctionalModel
from parsmooth.methods import iterated_smoothing

from parsmooth.linearization import extended

from parsmooth.sequential._bfgs import _batch_iterated_bfgs_smoother
from parsmooth.sequential._bfgs import log_posterior

import matplotlib.pyplot as plt

from bearing_data import get_data, make_parameters

jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
# jax.config.update('jax_disable_jit', True)


s1 = jnp.array([-1.5, 0.5])  # First sensor location
s2 = jnp.array([1., 1.])  # Second sensor location
r = 0.5  # Observation noise (stddev)
x0 = jnp.array([0.1, 0.2, 1, 0])  # initial true location
dt = 0.01  # discretization time step

qc = 0.01  # discretization noise
qw = 0.1  # discretization noise

T = 500  # number of observations
nx, ny = 5, 2

_, true_states, observations = get_data(x0, dt, r, T, s1, s2, random_state=17)

Q, R, transition_function, observation_function, _, _ = make_parameters(qc, qw, r, dt, s1, s2)

transition_model = FunctionalModel(transition_function, MVNStandard(jnp.zeros((nx,)), Q))
observation_model = FunctionalModel(observation_function, MVNStandard(jnp.zeros((ny,)), R))

initial_dist = MVNStandard(mean=jnp.array([-1., -1., 0., 0., 0.]),
                           cov=jnp.eye(nx))

nominal_trajectory = MVNStandard(mean=jnp.zeros((T + 1, nx)),
                                 cov=jnp.repeat(jnp.eye(nx).reshape(1, nx, nx), T + 1, axis=0))
nominal_trajectory.mean.at[0].set(initial_dist.mean)
nominal_trajectory.cov.at[0].set(initial_dist.cov)

# BFGS Batch Iterated Smoother
bfgs_smoothed = _batch_iterated_bfgs_smoother(observations, initial_dist,
                                              transition_model, observation_model,
                                              nominal_trajectory.mean,
                                              n_iter=100)[0]

bfgs_cost = log_posterior(bfgs_smoothed, observations,
                          initial_dist, transition_model, observation_model)

# Gauss-Newton Recursive Iterated Smoother
gauss_smoothed = iterated_smoothing(observations,
                                    initial_dist, transition_model, observation_model,
                                    extended, nominal_trajectory,
                                    False, criterion=lambda i, *_: i < 10)

gauss_cost = log_posterior(gauss_smoothed.mean, observations,
                           initial_dist, transition_model, observation_model)


plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
plt.plot(gauss_smoothed.mean[:, 0], gauss_smoothed.mean[:, 1], "-*", label="Iterated Recursive Gauss-Newton Smoother")
plt.plot(true_states[:, 0], true_states[:, 1], "*", label="True")
plt.title("Gauss-Newton")
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(bfgs_smoothed[:, 0], bfgs_smoothed[:, 1], "-*", label="Iterated Batch BFGS Smoother")
plt.plot(true_states[:, 0], true_states[:, 1], "*", label="True")
plt.title("Newton")
plt.grid()
plt.legend()
plt.show()
