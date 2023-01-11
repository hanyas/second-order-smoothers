from functools import partial

import jax
import numpy as np
import pytest
from jax.scipy.linalg import solve

from parsmooth._base import FunctionalModel, MVNStandard
from parsmooth.linearization import extended, extended_hessian
from parsmooth.sequential._filtering import filtering
from parsmooth.sequential._filtering_Newton import filtering as newton_filtering
from tests._lgssm import get_data, transition_function as lgssm_f, observation_function as lgssm_h
from tests._test_utils import get_system


@pytest.fixture(scope="session", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    jax.config.update('jax_disable_jit', False)
    jax.config.update("jax_debug_nans", False)


@pytest.mark.parametrize("dim_x", [1, 3])
@pytest.mark.parametrize("dim_y", [2, 3])
@pytest.mark.parametrize("seed", [0, 42])
def test_filter_no_info(dim_x, dim_y, seed):
    np.random.seed(seed)
    T = 3

    x0, _, F, Q, _, b, _ = get_system(dim_x, dim_x)
    _, _, H, R, cholR, c, _ = get_system(dim_x, dim_y)

    R = 1e12 * R
    true_states, observations = get_data(x0.mean, F, H, R, Q, b, c, T)

    transition_model = FunctionalModel(partial(lgssm_f, A=F), MVNStandard(b, Q))
    observation_model = FunctionalModel(partial(lgssm_h, H=H), MVNStandard(c, R))

    fun = lambda x: F @ x + b
    expected_mean = [x0.mean]

    for t in range(T):
        expected_mean.append(fun(expected_mean[-1]))

    filtered_states = filtering(observations, x0, transition_model, observation_model, extended, None)
    # filtered_states_newton = newton_filtering(observations, x0, transition_model, observation_model, extended_hessian, None)

    np.testing.assert_allclose(filtered_states.mean, expected_mean, atol=1e-3, rtol=1e-3)
    # np.testing.assert_allclose(filtered_states_newton.mean, expected_mean, atol=1e-3, rtol=1e-3)


# Run this test after including the if (lax.cond) part
# @pytest.mark.parametrize("dim", [1, 3])
# @pytest.mark.parametrize("seed", [0, 42])
# def test_filter_infinite_info(dim, seed):
#     np.random.seed(seed)
#     T = 3
#
#     x0, chol_x0, _, Q, cholQ, b, _ = get_system(dim, dim)
#     F = np.eye(dim)
#
#     _, _, _, R, cholR, c, _ = get_system(dim, dim)
#     H = np.eye(dim)
#
#     R = 1e-6 * R
#     cholR = 1e-3 * cholR
#     true_states, observations = get_data(x0.mean, F, H, R, Q, b, c, T, chol_R=cholR)
#
#
#     transition_model = FunctionalModel(partial(lgssm_f, A=F), MVNStandard(b, Q))
#     observation_model = FunctionalModel(partial(lgssm_h, H=H), MVNStandard(c, R))
#
#     expected_mean = np.stack([y - c for y in observations], axis=0)
#
#     filtered_states = filtering(observations, x0, transition_model, observation_model, extended, None)
#     filtered_states_newton = newton_filtering(observations, x0, transition_model, observation_model, extended_hessian, None)
#
#     np.testing.assert_allclose(filtered_states.mean[1:], expected_mean, atol=1e-3, rtol=1e-3)
#     np.testing.assert_allclose(filtered_states_newton.mean[1:], expected_mean, atol=1e-3, rtol=1e-3)


# @pytest.mark.parametrize("dim_x", [1, 3])
# @pytest.mark.parametrize("dim_y", [2, 3])
# @pytest.mark.parametrize("seed", [0, 42])
# def test_all_filters_agree(dim_x, dim_y, seed):
#     np.random.seed(seed)
#     T = 4
#
#     x0, chol_x0, F, Q, cholQ, b, _ = get_system(dim_x, dim_x)
#     _, _, H, R, cholR, c, _ = get_system(dim_x, dim_y)
#
#     true_states, observations = get_data(x0.mean, F, H, R, Q, b, c, T)
#
#     sqrt_transition_model = FunctionalModel(partial(lgssm_f, A=F), MVNSqrt(b, cholQ))
#     sqrt_observation_model = FunctionalModel(partial(lgssm_h, H=H), MVNSqrt(c, cholR))
#
#     transition_model = FunctionalModel(partial(lgssm_f, A=F), MVNStandard(b, Q))
#     observation_model = FunctionalModel(partial(lgssm_h, H=H), MVNStandard(c, R))
#
#     res = []
#     for method in LIST_LINEARIZATIONS:
#         filtered_states = filtering(observations, x0, transition_model, observation_model, method, None)
#         sqrt_filtered_states = filtering(observations, chol_x0, sqrt_transition_model, sqrt_observation_model, method,
#                                          None)
#         res.append(filtered_states)
#         res.append(sqrt_filtered_states)
#
#     for res_1, res_2 in zip(res[:-1], res[1:]):
#         np.testing.assert_array_almost_equal(res_1.mean, res_2.mean, decimal=3)
#
#
# @pytest.mark.parametrize("dim_x", [1, 3])
# @pytest.mark.parametrize("dim_y", [2, 3])
# @pytest.mark.parametrize("seed", [0, 42])
# def test_all_filters_with_nominal_traj(dim_x, dim_y, seed):
#     np.random.seed(seed)
#     T = 5
#     m_nominal = np.random.randn(T + 1, dim_x)
#     P_nominal = np.repeat(np.eye(dim_x, dim_x)[None, ...], T + 1, axis=0)
#     cholP_nominal = P_nominal
#
#     x_nominal = MVNStandard(m_nominal, P_nominal)
#     x_nominal_sqrt = MVNSqrt(m_nominal, cholP_nominal)
#
#     x0, chol_x0, F, Q, cholQ, b, _ = get_system(dim_x, dim_x)
#     _, _, H, R, cholR, c, _ = get_system(dim_x, dim_y)
#
#     true_states, observations = get_data(x0.mean, F, H, R, Q, b, c, T)
#
#     sqrt_transition_model = FunctionalModel(partial(lgssm_f, A=F), MVNSqrt(b, cholQ))
#     sqrt_observation_model = FunctionalModel(partial(lgssm_h, H=H), MVNSqrt(c, cholR))
#
#     transition_model = FunctionalModel(partial(lgssm_f, A=F), MVNStandard(b, Q))
#     observation_model = FunctionalModel(partial(lgssm_h, H=H), MVNStandard(c, R))
#
#     for method in LIST_LINEARIZATIONS:
#         filtered_states = filtering(observations, x0, transition_model, observation_model, method,
#                                     None)
#         filtered_states_nominal = filtering(observations, x0, transition_model, observation_model, method,
#                                             x_nominal)
#         sqrt_filtered_states = filtering(observations, chol_x0, sqrt_transition_model, sqrt_observation_model, method,
#                                          None)
#         sqrt_filtered_states_nominal = filtering(observations, chol_x0, sqrt_transition_model, sqrt_observation_model,
#                                                  method, x_nominal_sqrt)
#
#         np.testing.assert_allclose(filtered_states_nominal.mean, filtered_states.mean, atol=1e-3)
#         np.testing.assert_allclose(filtered_states_nominal.mean, sqrt_filtered_states_nominal.mean, atol=1e-3)
#
#         np.testing.assert_allclose(sqrt_filtered_states.mean, sqrt_filtered_states_nominal.mean, atol=1e-3)
