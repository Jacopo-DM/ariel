"""Heightmap generation functions for simulation environments."""

import typing
from typing import TYPE_CHECKING

# Third-party libraries
import cv2
import numpy as np

# Local libraries
from ariel.parameters import (
    ND_FLOAT_PRECISION,
    FloatArray,
)
from ariel.utils.noise_gen import NormMethod, PerlinNoise

if TYPE_CHECKING:
    from numpy.typing import NDArray


def flat_heightmap(
    dims: tuple[int, int],
) -> FloatArray:
    """Return a flat heightmap of zeros.

    Returns
    -------
    FloatArray
        The heightmap.
    """
    return np.zeros(shape=dims).astype(ND_FLOAT_PRECISION)


def rugged_heightmap(
    dims: tuple[int, int],
    scale_of_noise: int,
    normalize: NormMethod,
) -> FloatArray:
    # Create noise generator
    """Return a rugged heightmap from noise at `scale_of_noise`.

    Returns
    -------
    FloatArray
        The heightmap.
    """
    pnoise = PerlinNoise()

    # Generate a grid of noise
    nrow, ncol = dims
    return pnoise.as_grid(
        ncol,
        nrow,
        scale=scale_of_noise,
        normalize=normalize,
    )


def smooth_edges_heightmap(
    dims: tuple[int, int],
    edge_width: float,
) -> FloatArray:
    # Generate square grid for processing
    """Return a mask that smooths the heightmap edges over `edge_width`.

    Returns
    -------
    FloatArray
        The mask; 0 at the edge and 1 inside.
    """
    size = np.max(dims)
    y, x = np.mgrid[0:size, 0:size].astype(ND_FLOAT_PRECISION)
    x /= size
    y /= size

    # Distance to the nearest border (0 at border, up to 0.5 in center)
    # Using 1-D projections to calculate the boundary distance
    dist_to_border = np.minimum(
        np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y)
    ).astype(ND_FLOAT_PRECISION)

    # Unpack parameters
    w = edge_width

    # Define normalized transition value 't'
    # If edge_width <= 0: hard step at border
    # If edge_width > 0: linear ramp from 0 to 1 over width w
    if w <= 0.0:
        t = (dist_to_border > 0.0).astype(ND_FLOAT_PRECISION)
    else:
        t = np.piecewise(
            dist_to_border,
            [
                dist_to_border <= 0.0,
                (dist_to_border > 0.0) & (dist_to_border < w),
                dist_to_border >= w,
            ],
            [0.0, lambda d: d / w, 1.0],
        )

    # Smoothstep interpolation: 3t^2 - 2t^3
    mask = (t * t * (3.0 - 2.0 * t)).astype(ND_FLOAT_PRECISION)

    # Downsample/resize if the requested dims are not square
    if dims[0] != dims[1]:
        mask = cv2.resize(
            mask,
            dsize=(dims[1], dims[0]),
            interpolation=cv2.INTER_CUBIC,
        )

    return typing.cast("NDArray[np.float64]", mask.astype(ND_FLOAT_PRECISION))


def amphitheater_heightmap(
    dims: tuple[int, int],
    ring_inner_radius: float,
    ring_outer_radius: float,
    cone_height: float,
) -> FloatArray:
    # Generate grid
    """Return a heightmap with a ring-shaped amphitheater.

    Returns
    -------
    FloatArray
        The heightmap.
    """
    size = np.max(dims)
    y, x = np.mgrid[0:size, 0:size].astype(ND_FLOAT_PRECISION)
    x /= size
    y /= size

    # Radial distance from center
    r = np.array(
        np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2),
    ).astype(ND_FLOAT_PRECISION)

    # Unpack parameters
    r0 = ring_inner_radius
    r1 = ring_outer_radius
    d = cone_height

    # Piecewise slope: flat -> conical rise -> plateau
    heightmap = np.piecewise(
        r,
        [r <= r0, (r > r0) & (r <= r1), r > r1],
        [0.0, lambda r: d * (r - r0) / (r1 - r0), d],
    )

    # Downsample if dims is not square
    if dims[0] != dims[1]:
        heightmap = cv2.resize(
            heightmap,
            dsize=(dims[1], dims[0]),
            interpolation=cv2.INTER_CUBIC,
        )
    return heightmap.astype(ND_FLOAT_PRECISION)


def crater_heightmap(
    dims: tuple[int, int],
    crater_depth: float,
    crater_radius: float,
) -> FloatArray:
    # Generate grid
    """Return a heightmap with a crater of given depth and radius.

    Returns
    -------
    FloatArray
        The heightmap.
    """
    size = np.max(dims)
    y, x = np.mgrid[0:size, 0:size].astype(ND_FLOAT_PRECISION)
    x /= size
    y /= size

    # Elliptical cone shape
    a = crater_radius
    b = crater_radius

    # Base conical height
    r = np.sqrt(((x - 0.5) / a) ** 2 + ((y - 0.5) / b) ** 2)
    heightmap = crater_depth * r

    # Downsample if dims is not square
    if dims[0] != dims[1]:
        heightmap = cv2.resize(
            heightmap,
            dsize=(dims[1], dims[0]),
            interpolation=cv2.INTER_CUBIC,
        )
    return typing.cast(
        "NDArray[np.float64]", heightmap.astype(ND_FLOAT_PRECISION)
    )
