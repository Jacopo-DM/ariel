"""Settings of the MuJoCo worlds that are not physics.

Notes
-----
    *

References
----------
    [1]

Todo
----
    [ ]

"""

# Standard library
from pathlib import Path

# Third-party libraries
import numpy as np
from pydantic_settings import BaseSettings

# Local libraries
# Global constants
# Global functions
# Warning Control
# Type Checking
# Type Aliases
# Type Aliases
type WeightType = float
type DimensionType = tuple[float, float, float]

# --- DATA SETUP --- #
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__"
DATA.mkdir(exist_ok=True)

# --- RANDOM GENERATOR SETUP --- #
SEED = 42
RNG = np.random.default_rng(SEED)


class MujocoConfig(BaseSettings):
    """Names that worlds share; physics settings live in ``Seeds``."""

    floor_name: str = "floor"
