from typing import Callable

import jax
from jax import numpy as jnp
from jax.scipy.stats import multivariate_normal as mvn

from smoothers import MVNStandard, FunctionalModel
from smoothers.base import LinearTransition, LinearObservation
from smoothers.utils import none_or_shift


def log_posterior_cost(
    states: jnp.ndarray,
    observations: jnp.ndarray,
    init_dist: MVNStandard,
    transition_model: FunctionalModel,
    observation_model: FunctionalModel,
):
    xp, xn = states[:-1], states[1:]
    yn = observations

    m0, P0 = init_dist
    f, (_, Q) = transition_model
    h, (_, R) = observation_model

    xn_mu = jax.vmap(f)(xp)
    yn_mu = jax.vmap(h)(xn)

    cost = -mvn.logpdf(states[0], m0, P0)
    cost -= jnp.sum(jax.vmap(mvn.logpdf, in_axes=(0, 0, None))(xn, xn_mu, Q))
    cost -= jnp.sum(jax.vmap(mvn.logpdf, in_axes=(0, 0, None))(yn, yn_mu, R))
    return cost


def approx_log_posterior_cost(
    states: jnp.ndarray,
    observations: jnp.ndarray,
    init_dist: MVNStandard,
    linear_transition: LinearTransition,
    linear_observation: LinearObservation,
):
    xp, xn = states[:-1], states[1:]
    yn = observations

    m0, P0 = init_dist
    F_x, b, Q = linear_transition
    H_x, c, R = linear_observation

    xn_mu = jnp.einsum("nij,nj->ni", F_x, xp) + b
    yn_mu = jnp.einsum("nij,nj->ni", H_x, xn) + c

    cost = -mvn.logpdf(states[0], m0, P0)
    cost -= jnp.sum(jax.vmap(mvn.logpdf)(xn, xn_mu, Q))
    cost -= jnp.sum(jax.vmap(mvn.logpdf)(yn, yn_mu, R))
    return cost


def linearize_state_space_model(
    transition_model: FunctionalModel,
    observation_model: FunctionalModel,
    linearization_method: Callable,
    nominal_trajectory: MVNStandard,
):
    curr_nominal = none_or_shift(nominal_trajectory, -1)
    next_nominal = none_or_shift(nominal_trajectory, 1)

    Fs, bs, Qs = jax.vmap(linearization_method, in_axes=(None, 0))(
        transition_model, curr_nominal
    )
    Hs, cs, Rs = jax.vmap(linearization_method, in_axes=(None, 0))(
        observation_model, next_nominal
    )

    return LinearTransition(Fs, bs, Qs), LinearObservation(Hs, cs, Rs)
