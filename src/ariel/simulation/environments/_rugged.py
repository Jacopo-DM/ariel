"""MuJoCo world: rugged heightmap terrain."""

# Standard library
from dataclasses import dataclass  # ruff: ignore[banned-api]  # behaviour first: a world builds its MjSpec after init; the fields are its settings

# Local libraries
from ariel.parameters import Dimension
from ariel.simulation.environments._compound_world import CompoundWorld
from ariel.simulation.environments.heightmap_functions import (
    rugged_heightmap,
    smooth_edges_heightmap,
)
from ariel.utils import NormMethod


@dataclass
class RuggedTerrainWorld(CompoundWorld):
    """A rugged terrain world (CompoundWorld)."""

    name: str = "rugged-world"

    floor_size: Dimension = (10, 10, 1)  # meters (width, height, depth)
    checker_floor: bool = False

    # Rugged heightmap parameters
    dims: tuple[int, int] = (100, 100)
    scale_of_noise: int = 4
    normalize: NormMethod = "none"

    # Whether to load precompiled XML (if it exists)
    load_precompiled: bool = True

    def __post_init__(self) -> None:
        # Rugged part of heightmap
        rugged_part = rugged_heightmap(
            self.dims,
            self.scale_of_noise,
            self.normalize,
        )

        # Smooth edges of world
        smooth_part = smooth_edges_heightmap(self.dims, edge_width=0.2)

        # Combine parts
        self.floor_heightmap = rugged_part * smooth_part

        # Initialize base class
        super().__init__(
            name=self.name,
            floor_size=self.floor_size,
            checker_floor=self.checker_floor,
            floor_heightmap=self.floor_heightmap,
            load_precompiled=self.load_precompiled,
        )
