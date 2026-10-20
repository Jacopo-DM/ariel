"""Tests for the Euler angle conversion (speed-up S2)."""

import itertools
import math

import numpy as np
import quaternion as library
from numpy.typing import NDArray

from ariel.body_phenotypes import quaternions

TRIPLES = 2000
RIGHT_ANGLES = [k * math.pi / 4 for k in range(-8, 9)]


def reference(angles: tuple[float, float, float]) -> NDArray[np.float64]:
    """Convert with the library, then roll by one, as the modules did before.

    Parameters
    ----------
    angles
        The three Euler angles in radians.

    Returns
    -------
    NDArray[np.float64]
        The components with the scalar part last.
    """
    quat = library.from_euler_angles(list(angles))  # type: ignore[no-untyped-call]  # numpy-quaternion has no type hints
    return np.roll(library.as_float_array(quat), shift=-1)  # type: ignore[no-untyped-call]  # numpy-quaternion has no type hints


def test_the_conversion_gives_the_same_bits_as_the_library() -> None:
    """Random angles and every multiple of 45 degrees match exactly."""
    rng = np.random.default_rng(0)
    random = [
        tuple(float(x) for x in rng.uniform(-7, 7, 3)) for _ in range(TRIPLES)
    ]
    grid = list(itertools.product(RIGHT_ANGLES, repeat=3))
    for angles in [*random, *grid]:
        triple = (angles[0], angles[1], angles[2])
        assert np.array_equal(
            quaternions.from_euler_angles(triple), reference(triple)
        )


def test_the_identity_rotation_has_a_unit_scalar_part_last() -> None:
    """No rotation is (0, 0, 0, 1): the scalar part is the last component."""
    assert np.array_equal(
        quaternions.from_euler_angles([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0, 1.0]
    )
