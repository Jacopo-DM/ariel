"""Test: the Perlin noise generator."""

from __future__ import annotations

# Standard library
# Third-party libraries
import numpy as np
import pytest

# Local libraries
from ariel.simulation.environments.noise_gen import PerlinNoise

SPIDER_HINGES = 4
SPIDER_MODULES = 5
BRANCHING_BRICKS = 2

# ---------------------------------------------------------------------------
# PerlinNoise
# ---------------------------------------------------------------------------


def test_perlin_noise_initialization() -> None:
    """Simply instantiate the PerlinNoise class."""
    _ = PerlinNoise()


def test_perlin_noise_seeded_reproducible() -> None:
    """Two PerlinNoise instances with the same seed produce identical grids."""
    n1 = PerlinNoise(seed=42)
    n2 = PerlinNoise(seed=42)
    g1 = n1.as_grid(32, 32, scale=8.0)
    g2 = n2.as_grid(32, 32, scale=8.0)
    assert np.allclose(g1, g2)


def test_perlin_noise_different_seeds_differ() -> None:
    """Two PerlinNoise instances with different seeds give different grids."""
    n1 = PerlinNoise(seed=1)
    n2 = PerlinNoise(seed=2)
    g1 = n1.as_grid(32, 32, scale=8.0)
    g2 = n2.as_grid(32, 32, scale=8.0)
    assert not np.allclose(g1, g2)


def test_perlin_noise_output_shape() -> None:
    """as_grid returns an array of shape (height, width)."""
    noise = PerlinNoise(seed=0)
    grid = noise.as_grid(width=64, height=32, scale=16.0)
    assert grid.shape == (32, 64)


def test_perlin_noise_normalize_linear_range() -> None:
    """Linear normalization clips output to approximately [0, 1]."""
    noise = PerlinNoise(seed=7)
    grid = noise.as_grid(64, 64, scale=16.0, normalize="linear")
    assert grid.min() >= 0.0 - 1e-6
    assert grid.max() <= 1.0 + 1e-6


def test_perlin_noise_normalize_clip_range() -> None:
    """Clip normalization keeps all values in [0, 1]."""
    noise = PerlinNoise(seed=7)
    grid = noise.as_grid(64, 64, scale=16.0, normalize="clip")
    assert grid.min() >= 0.0
    assert grid.max() <= 1.0


def test_perlin_noise_invalid_width_raises() -> None:
    """as_grid raises ValueError for non-positive width."""
    noise = PerlinNoise()
    with pytest.raises(ValueError, match="must be > 0"):
        noise.as_grid(width=0, height=32, scale=8.0)


def test_perlin_noise_invalid_height_raises() -> None:
    """as_grid raises ValueError for non-positive height."""
    noise = PerlinNoise()
    with pytest.raises(ValueError, match="must be > 0"):
        noise.as_grid(width=32, height=0, scale=8.0)


def test_perlin_noise_invalid_scale_raises() -> None:
    """as_grid raises ValueError for non-positive scale."""
    noise = PerlinNoise()
    with pytest.raises(ValueError, match="must be > 0"):
        noise.as_grid(width=32, height=32, scale=0.0)
