"""Types of robotic controllers."""

# Local libraries
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.controllers.cpg import (
    CPG,
    AdaptiveHopfCPG,
    HopfCPG,
    LearnableTegotaeCPG,
    NaCPG,
    PhaseCPG,
    SwingStanceCPG,
    TegotaeCPG,
    create_fully_connected_adjacency,
)

__all__ = [
    "CPG",
    "AdaptiveHopfCPG",
    "Controller",
    "HopfCPG",
    "LearnableTegotaeCPG",
    "NaCPG",
    "PhaseCPG",
    "SwingStanceCPG",
    "TegotaeCPG",
    "create_fully_connected_adjacency",
]
