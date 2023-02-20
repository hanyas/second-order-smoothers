import jax
import jax.numpy as jnp
import numpy as np
import pytest

from parsmooth._base import MVNStandard, FunctionalModel
from parsmooth.linearization import extended, second_order
from parsmooth.methods import iterated_smoothing
from parsmooth.sequential._filtering import filtering
from parsmooth.sequential._filtering_newton import filtering as newton_filtering
from tests.bearings.bearings_utils import make_parameters


@pytest.fixture(scope="session", autouse=True)
def config():
    jax.config.update("jax_enable_x64", False)
    jax.config.update('jax_disable_jit', False)
    jax.config.update("jax_debug_nans", False)


# @pytest.mark.skip("Skip on continuous integration")
def test_bearings():
    s1 = jnp.array([-1.5, 0.5])  # First sensor location
    s2 = jnp.array([1., 1.])  # Second sensor location
    r = 0.5  # Observation noise (stddev)
    dt = 0.01  # discretization time step
    qc = 0.01  # discretization noise
    qw = 0.1  # discretization noise

    ys = np.load("./bearings/ys.npy")
    with np.load("./bearings//ieks.npz") as loaded:
        expected_mean, expected_cov = loaded["arr_0"], loaded["arr_1"]

    Q, R, observation_function, transition_function = make_parameters(qc, qw, r, dt, s1, s2)

    m0 = jnp.array([-1., -1., 0., 0., 0.])
    P0 = jnp.eye(5)

    init = MVNStandard(m0, P0)
    transition_model = FunctionalModel(transition_function, MVNStandard(jnp.zeros((5,)), Q))

    observation_model = FunctionalModel(observation_function, MVNStandard(jnp.zeros((2,)), R))

    filtered_states = filtering(ys, init, transition_model, observation_model, extended, None)
    filtered_states_newton = newton_filtering(ys, init, transition_model, observation_model, extended_hessian, None)
    f, _ = transition_model
    # print("hessian", jax.jacfwd(jax.jacrev(f))(m0))
    np.testing.assert_array_almost_equal(filtered_states.mean, filtered_states_newton.mean, decimal=3)  # noqa


    # np.testing.assert_array_almost_equal(filtered_states.mean[1:], expected_mean, decimal=3)  # noqa
    # np.testing.assert_array_almost_equal(filtered_states_newton.mean[1:], expected_mean, decimal=3)  # noqa
