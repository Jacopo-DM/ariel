"""Types of robotic controllers."""

# Local libraries
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.controllers.na_cpg import (
    NaCPG,
    create_fully_connected_adjacency,
)

__all__ = ["Controller", "NaCPG", "create_fully_connected_adjacency"]
