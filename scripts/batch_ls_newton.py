import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

from optsmooth import MVNStandard
from optsmooth import FunctionalModel
from optsmooth.linearization import second_order
from optsmooth.batch._ls_newton import line_search_iterated_batch_newton_smoother

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


_, true_states, observations = get_data(x0, dt, r, T, s1, s2, random_state=42)
Q, R, trns_fcn, obs_fcn, _, _ = make_parameters(qc, qw, r, dt, s1, s2)

trns_mdl = FunctionalModel(trns_fcn, MVNStandard(jnp.zeros((nx,)), Q))
obs_mdl = FunctionalModel(obs_fcn, MVNStandard(jnp.zeros((ny,)), R))

init_dist = MVNStandard(mean=jnp.array([-1., -1., 0., 0., 0.]),
                        cov=jnp.eye(nx))

init_nominal_mean = jnp.zeros((T + 1, nx))
init_nominal_mean.at[0].set(init_dist.mean)

smoothed_traj, costs = line_search_iterated_batch_newton_smoother(observations, init_dist,
                                                                  trns_mdl, obs_mdl,
                                                                  second_order, init_nominal_mean,
                                                                  nb_iter=50)

plt.figure(figsize=(7, 7))
plt.plot(smoothed_traj[:, 0], smoothed_traj[:, 1], "-*", label="Iterated Batch Newton Smoother")
plt.plot(true_states[:, 0], true_states[:, 1], "*", label="True")
plt.title("Newton")
plt.grid()
plt.legend()
plt.show()
