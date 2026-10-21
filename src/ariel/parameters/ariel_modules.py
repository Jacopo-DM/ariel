"""Configuration of the RoboGen-lite modules and the servo of the hinge."""

# Third-party libraries
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Local libraries
from ariel.parameters.module_specs import BrickSpec, CoreSpec, HingeSpec
from ariel.parameters.servos import DSSM15


class ArielModulesConfig(BaseSettings):
    """Measured sizes and masses of the modules, and the hinge's servo.

    An environment variable overrides one value, written with its unit and
    named by the path to it: ``BRICK__MASS="0.1 kg"``. A bare number is
    refused.
    """

    model_config = SettingsConfigDict(env_nested_delimiter="__")

    brick: BrickSpec = Field(default_factory=BrickSpec)
    core: CoreSpec = Field(default_factory=CoreSpec)
    hinge: HingeSpec = Field(default_factory=HingeSpec)
    servo: DSSM15 = Field(default_factory=DSSM15)
