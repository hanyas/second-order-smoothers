import jax
import jax.numpy as jnp

from smoothers.base import MVNStandard, FunctionalModel
from smoothers import line_search_iterated_batch_newton_smoother

from bearing_data import get_data, make_parameters

import matplotlib.pyplot as plt

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
Q, R, trans_fcn, obsrv_fcn, _, _ = make_parameters(qc, qw, r, dt, s1, s2)

trans_mdl = FunctionalModel(trans_fcn, MVNStandard(jnp.zeros((nx,)), Q))
obsrv_mdl = FunctionalModel(obsrv_fcn, MVNStandard(jnp.zeros((ny,)), R))

init_dist = MVNStandard(
    mean=jnp.array([-1.0, -1.0, 0.0, 0.0, 0.0]), cov=jnp.eye(nx)
)

init_nominal = jnp.zeros((T + 1, nx))
init_nominal.at[0].set(init_dist.mean)

smoothed, costs = line_search_iterated_batch_newton_smoother(
    init_nominal, observations, init_dist, trans_mdl, obsrv_mdl, nb_iter=50
)

plt.figure(figsize=(7, 7))
plt.plot(
    smoothed[:, 0],
    smoothed[:, 1],
    "-*",
    label="Newton",
)
plt.plot(true_states[:, 0], true_states[:, 1], "*", label="True")
plt.title("Iterated Batch Newton Smoother with Line Search")
plt.grid()
plt.legend()
plt.show()
