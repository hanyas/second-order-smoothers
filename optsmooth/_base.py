import itertools
from typing import NamedTuple, Callable

import jax.numpy as jnp


class MVNStandard(NamedTuple):
    mean: jnp.ndarray
    cov: jnp.ndarray


class LinearTransition(NamedTuple):
    F_x: jnp.ndarray
    b: jnp.ndarray
    Q: jnp.ndarray


class LinearObservation(NamedTuple):
    H_x: jnp.ndarray
    c: jnp.ndarray
    R: jnp.ndarray


class QuadraticTransition(NamedTuple):
    F_xx: jnp.ndarray
    F_x: jnp.ndarray
    b: jnp.ndarray
    Q: jnp.ndarray


class QuadraticObservation(NamedTuple):
    H_xx: jnp.ndarray
    H_x: jnp.ndarray
    c: jnp.ndarray
    R: jnp.ndarray


class FunctionalModel(NamedTuple):
    function: Callable
    mvn: MVNStandard


def are_inputs_compatible(*y):
    a, b = itertools.tee(map(type, y))
    _ = next(b, None)
    ok = sum(map(lambda u: u[0] == u[1], zip(a, b)))
    if not ok:
        raise TypeError(f"All inputs should have the same type. {y} was given")
