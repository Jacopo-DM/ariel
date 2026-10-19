"""Phenotype descriptions of different robot systems (based on 'MjSpec')."""

# Local libraries
from ariel.body_phenotypes.morphological_descriptor import MorphologicalMeasures
from ariel.body_phenotypes.robogen_lite import (
    ALLOWED_FACES,
    ALLOWED_ROTATIONS,
    IDX_OF_CORE,
    NUM_OF_FACES,
    NUM_OF_ROTATIONS,
    NUM_OF_TYPES_OF_MODULES,
    CoreModule,
    Genome,
    IdManager,
    ModuleFaces,
    ModuleRotationsIdx,
    ModuleType,
    MorphologyDecoderBestFirst,
    construct_mjspec_from_graph,
)

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
    "MorphologicalMeasures",
    "MorphologyDecoderBestFirst",
    "construct_mjspec_from_graph",
]
