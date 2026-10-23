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
    ModuleRotationsTheta,
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
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule
from ariel.body_phenotypes.robogen_lite.overlap import (
    Overlap,
    find_overlaps,
    module_sizes,
    repair_overlaps,
)
from ariel.body_phenotypes.robogen_lite.rest_pose import RestBoxes, rest_boxes

__all__ = [
    "ALLOWED_FACES",
    "ALLOWED_ROTATIONS",
    "IDX_OF_CORE",
    "NUM_OF_FACES",
    "NUM_OF_ROTATIONS",
    "NUM_OF_TYPES_OF_MODULES",
    "BrickModule",
    "CoreModule",
    "Genome",
    "HingeModule",
    "IdManager",
    "ModuleFaces",
    "ModuleRotationsIdx",
    "ModuleRotationsTheta",
    "ModuleType",
    "MorphologyDecoderBestFirst",
    "Overlap",
    "RestBoxes",
    "construct_mjspec_from_graph",
    "find_overlaps",
    "module_sizes",
    "repair_overlaps",
    "rest_boxes",
]
