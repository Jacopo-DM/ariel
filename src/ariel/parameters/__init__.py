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
from ariel.parameters.module_specs import BrickSpec, CoreSpec, HingeSpec
from ariel.parameters.mujoco_params import MujocoConfig
from ariel.parameters.quantities import (
    ANGLES,
    KILOGRAM_FORCE,
    STANDARD_GRAVITY,
    kilograms,
    metres,
    quantity_of,
    radians,
    to_unit,
)
from ariel.parameters.servos import DSSM15, ServoRating, ServoSettings

__all__ = [
    "ANGLES",
    "DSSM15",
    "KILOGRAM_FORCE",
    "ND_FLOAT_PRECISION",
    "ND_INT_PRECISION",
    "STANDARD_GRAVITY",
    "ArielConfig",
    "ArielModulesConfig",
    "BrickSpec",
    "CoreSpec",
    "Dimension",
    "FloatArray",
    "FrozenModel",
    "HingeSpec",
    "IntArray",
    "MujocoConfig",
    "MutableModel",
    "Position",
    "Rotation",
    "ServoRating",
    "ServoSettings",
    "kilograms",
    "metres",
    "quantity_of",
    "radians",
    "to_unit",
]


class ArielConfig(BaseSettings):
    """Global ARIEL settings."""

    verbose: bool = True
