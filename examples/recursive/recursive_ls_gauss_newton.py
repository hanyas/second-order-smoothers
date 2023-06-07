import jax
import jax.numpy as jnp

from smoothers import MVNStandard
from smoothers import FunctionalModel
from smoothers.linearization import extended
from smoothers import line_search_iterated_recursive_gauss_newton_smoother

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

init_nominal = MVNStandard(
    mean=jnp.zeros((T + 1, nx)),
    cov=jnp.repeat(jnp.eye(nx).reshape(1, nx, nx), T + 1, axis=0),
)
init_nominal.mean.at[0].set(init_dist.mean)
init_nominal.cov.at[0].set(init_dist.cov)

smoothed, costs = line_search_iterated_recursive_gauss_newton_smoother(
    init_nominal,
    observations,
    init_dist,
    trans_mdl,
    obsrv_mdl,
    extended,
    nb_iter=25,
)

plt.figure(figsize=(7, 7))
plt.plot(
    smoothed.mean[:, 0],
    smoothed.mean[:, 1],
    "-*",
    label="Gauss-Newton",
)
plt.plot(true_states[:, 0], true_states[:, 1], "*", label="True")
plt.title("Iterated Recursive Gauss-Newton Smoother with Line Search")
plt.grid()
plt.legend()
plt.show()
