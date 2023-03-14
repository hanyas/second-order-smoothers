import jax
import jax.numpy as jnp

from optsmooth import MVNStandard
from optsmooth import FunctionalModel
from optsmooth.linearization import extended, second_order
from optsmooth.sequential._ls_gauss_newton import (
    line_search_iterated_recursive_gauss_newton_smoother
)

import matplotlib.pyplot as plt

from bearing_data import get_data, make_parameters

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
# jax.config.update('jax_disable_jit', True)


s1 = jnp.array([-1.5, 0.5])  # First sensor location
s2 = jnp.array([1.0, 1.0])  # Second sensor location
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

init_dist = MVNStandard(
    mean=jnp.array([-1.0, -1.0, 0.0, 0.0, 0.0]), cov=jnp.eye(nx)
)

nominal_traj = MVNStandard(
    mean=jnp.zeros((T + 1, nx)),
    cov=jnp.repeat(jnp.eye(nx).reshape(1, nx, nx), T + 1, axis=0),
)
nominal_traj.mean.at[0].set(init_dist.mean)
nominal_traj.cov.at[0].set(init_dist.cov)

smoothed_traj, costs = line_search_iterated_recursive_gauss_newton_smoother(
    observations,
    init_dist,
    trns_mdl,
    obs_mdl,
    extended,
    second_order,
    nominal_traj,
    nb_iter=25,
)

plt.figure(figsize=(7, 7))
plt.plot(
    smoothed_traj.mean[:, 0],
    smoothed_traj.mean[:, 1],
    "-*",
    label="Iterated Recursive Gauss-Newton Smoother",
)
plt.plot(true_states[:, 0], true_states[:, 1], "*", label="True")
plt.title("Gauss-Newton")
plt.grid()
plt.legend()
plt.show()
