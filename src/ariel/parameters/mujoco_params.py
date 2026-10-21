"""Settings of the MuJoCo worlds that are not physics."""

# Third-party libraries
from pydantic_settings import BaseSettings


class MujocoConfig(BaseSettings):
    """Names that worlds share; physics settings live in ``Seeds``."""

    floor_name: str = "floor"
