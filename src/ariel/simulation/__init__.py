"""ARIEL simulation package."""

# Local libraries
from ariel.simulation.controllers import (
    CPG,
    AdaptiveHopfCPG,
    Controller,
    HopfCPG,
    InputRoute,
    LearnableTegotaeCPG,
    NaCPG,
    PhaseCPG,
    SwingStanceCPG,
    TegotaeCPG,
    create_fully_connected_adjacency,
)
from ariel.simulation.environments import (
    AmphitheatreTerrainWorld,
    BaseWorld,
    CompoundWorld,
    CraterTerrainWorld,
    OlympicArena,
    RuggedTerrainWorld,
    RuggedTiltedWorld,
    SimpleFlatWorld,
    SimpleTiltedWorld,
)
from ariel.simulation.environments.noise_gen import NormMethod, PerlinNoise
from ariel.simulation.mujoco_ops import (
    euler_to_quat_conversion,
    has_self_collision,
    mjspec_deep_copy,
)
from ariel.simulation.parallel import process_pool
from ariel.simulation.tracker import Tracker

__all__ = [
    "CPG",
    "AdaptiveHopfCPG",
    "AmphitheatreTerrainWorld",
    "BaseWorld",
    "CompoundWorld",
    "Controller",
    "CraterTerrainWorld",
    "HopfCPG",
    "InputRoute",
    "LearnableTegotaeCPG",
    "NaCPG",
    "NormMethod",
    "OlympicArena",
    "PerlinNoise",
    "PhaseCPG",
    "RuggedTerrainWorld",
    "RuggedTiltedWorld",
    "SimpleFlatWorld",
    "SimpleTiltedWorld",
    "SwingStanceCPG",
    "TegotaeCPG",
    "Tracker",
    "create_fully_connected_adjacency",
    "euler_to_quat_conversion",
    "has_self_collision",
    "mjspec_deep_copy",
    "process_pool",
]
