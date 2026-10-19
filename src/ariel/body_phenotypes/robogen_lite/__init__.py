"""ARIEL-robots."""

# Local libraries
from ariel.body_phenotypes.robogen_lite.config import (
    ALLOWED_FACES,
    ALLOWED_ROTATIONS,
    IDX_OF_CORE,
    NUM_OF_FACES,
    NUM_OF_ROTATIONS,
    NUM_OF_TYPES_OF_MODULES,
    ModuleFaces,
    ModuleRotationsIdx,
    ModuleType,
)
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.cppn_neat.genome import Genome
from ariel.body_phenotypes.robogen_lite.cppn_neat.id_manager import IdManager
from ariel.body_phenotypes.robogen_lite.decoders.cppn_best_first import (
    MorphologyDecoderBestFirst,
)
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule

__all__ = [
    "ALLOWED_FACES",
    "ALLOWED_ROTATIONS",
    "IDX_OF_CORE",
    "NUM_OF_FACES",
    "NUM_OF_ROTATIONS",
    "NUM_OF_TYPES_OF_MODULES",
    "CoreModule",
    "Genome",
    "IdManager",
    "ModuleFaces",
    "ModuleRotationsIdx",
    "ModuleType",
    "MorphologyDecoderBestFirst",
    "construct_mjspec_from_graph",
]
