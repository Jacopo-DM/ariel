"""Euler angles to the quaternions that the robot modules give to MuJoCo.

The module used to wrap ``numpy-quaternion``, which builds the quaternion with
NumPy calls on 0-d arrays (14 us) and was followed by a ``numpy.roll`` (8 us)
at every call site. The same four products in plain ``math`` take 1 us and give
the same bits (DESIGN.md Decisions, speed-up S2).
"""

# Standard library
import math
from collections.abc import Sequence

# Third-party libraries
import numpy as np
from numpy.typing import NDArray


def from_euler_angles(angles: Sequence[float]) -> NDArray[np.float64]:
    """Build the quaternion of three Euler angles, as the modules store it.

    The rotation is ``exp(alpha*z/2) * exp(beta*y/2) * exp(gamma*z/2)``
    (the convention of ``numpy-quaternion``).

    Parameters
    ----------
    angles
        The three Euler angles ``(alpha, beta, gamma)`` in radians.

    Returns
    -------
    NDArray[np.float64]
        The components in the order the modules hand to MuJoCo: the vector
        part first and the scalar part last, that is ``(x, y, z, w)``.
    """
    alpha, beta, gamma = angles
    cos_half_beta = math.cos(beta / 2)
    sin_half_beta = math.sin(beta / 2)
    return np.array([
        -sin_half_beta * math.sin((alpha - gamma) / 2),
        sin_half_beta * math.cos((alpha - gamma) / 2),
        cos_half_beta * math.sin((alpha + gamma) / 2),
        cos_half_beta * math.cos((alpha + gamma) / 2),
    ])
