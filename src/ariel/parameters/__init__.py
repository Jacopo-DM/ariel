# ruff: file-ignore[non-empty-init-module]  # package entry point: runs setup code before the re-exports (see CODEMAP.md)
"""Centralized location for all configuration settings in ARIEL."""

# Third-party libraries
from pydantic_settings import BaseSettings

# Local libraries
from ariel.parameters.ariel_modules import ArielModulesConfig
from ariel.parameters.ariel_types import (
    ND_FLOAT_PRECISION,
    ND_INT_PRECISION,
    Dimension,
    FloatArray,
    IntArray,
    Position,
    Rotation,
)
from ariel.parameters.base_model import FrozenModel, MutableModel
from ariel.parameters.mujoco_params import MujocoConfig

__all__ = [
    "ND_FLOAT_PRECISION",
    "ND_INT_PRECISION",
    "ArielConfig",
    "ArielModulesConfig",
    "Dimension",
    "FloatArray",
    "FrozenModel",
    "IntArray",
    "MujocoConfig",
    "MutableModel",
    "Position",
    "Rotation",
]


class ArielConfig(BaseSettings):
    """Global ARIEL settings."""

    verbose: bool = True
