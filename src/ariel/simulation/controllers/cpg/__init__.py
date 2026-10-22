"""Central pattern generators: networks of coupled oscillators, one per joint.

Every class here shares the interface of ``CPG``: parameter groups that can be
set as one flat vector, ``forward(time)`` that follows the simulation clock,
angle limits and an optional history. The functional notebooks
``tests/functional/cpg_properties.ipynb`` and ``cpg_locomotion.ipynb`` compare
them.
"""

# Local libraries
from ariel.simulation.controllers.cpg._base import (
    CPG,
    BetaParams,
    InputRoute,
    create_fully_connected_adjacency,
    scaled_beta,
)
from ariel.simulation.controllers.cpg.hopf import AdaptiveHopfCPG, HopfCPG
from ariel.simulation.controllers.cpg.na_cpg import NaCPG
from ariel.simulation.controllers.cpg.phase import PhaseCPG
from ariel.simulation.controllers.cpg.swing_stance import SwingStanceCPG
from ariel.simulation.controllers.cpg.tegotae import (
    LearnableTegotaeCPG,
    TegotaeCPG,
)

__all__ = [
    "CPG",
    "AdaptiveHopfCPG",
    "BetaParams",
    "HopfCPG",
    "InputRoute",
    "LearnableTegotaeCPG",
    "NaCPG",
    "PhaseCPG",
    "SwingStanceCPG",
    "TegotaeCPG",
    "create_fully_connected_adjacency",
    "scaled_beta",
]
