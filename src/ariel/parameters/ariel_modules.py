"""Configuration of the RoboGen-lite modules and the servo of the hinge."""

# Third-party libraries
from pydantic import Field
from pydantic_settings import BaseSettings

# Local libraries
from ariel.parameters.servos import DSSM15

# Type Aliases
type WeightType = float
type DimensionType = tuple[float, float, float]


class ArielModulesConfig(BaseSettings):
    # --- Brick Config --- #
    # Module weights (kg)
    """Masses, dimensions and limits of the RoboGen-lite modules."""

    BRICK_MASS: WeightType = 0.055  # 55 grams

    # Module dimensions (length, width, height) in meters
    BRICK_DIMENSIONS: DimensionType = (0.05, 0.05, 0.05)
    # ------------------------------ #

    # --- Hinge Config --- #
    ACTUATOR: DSSM15 = Field(default_factory=DSSM15)
    # ------------------------------ #
