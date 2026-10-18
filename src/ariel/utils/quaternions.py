"""Typed wrappers around the untyped numpy-quaternion functions.

``quaternion`` ships no type hints, so every call to it is an untyped call.
The robot modules reach it only through these two functions, which keeps the
untyped calls, and their ignores, in one place.
"""

# Standard library
from collections.abc import Sequence
from typing import cast

# Third-party libraries
import numpy as np
import quaternion as _quaternion
from numpy.typing import NDArray


def from_euler_angles(angles: Sequence[float]) -> NDArray[np.void]:
    """Build a quaternion from three Euler angles.

    Parameters
    ----------
    angles
        The three Euler angles in radians, as ``quaternion`` reads them.

    Returns
    -------
    NDArray[np.void]
        The quaternion as a ``quaternion`` array.
    """
    return cast(
        "NDArray[np.void]",
        _quaternion.from_euler_angles(list(angles)),  # type: ignore[no-untyped-call]  # numpy-quaternion has no type hints
    )


def as_float_array(quaternions: NDArray[np.void]) -> NDArray[np.float64]:
    """Convert quaternions to a float array with the components last.

    Parameters
    ----------
    quaternions
        The quaternions to convert.

    Returns
    -------
    NDArray[np.float64]
        The components in (w, x, y, z) order, one row per quaternion.
    """
    return cast(
        "NDArray[np.float64]",
        _quaternion.as_float_array(quaternions),  # type: ignore[no-untyped-call]  # numpy-quaternion has no type hints
    )
