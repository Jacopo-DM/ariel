"""Body genotype after Karl Sims (1994): a graph of module templates."""

from .genome import SimsConnection, SimsGenome, SimsNode
from .unfold import max_copy_limit, unfold

__all__ = [
    "SimsConnection",
    "SimsGenome",
    "SimsNode",
    "max_copy_limit",
    "unfold",
]
