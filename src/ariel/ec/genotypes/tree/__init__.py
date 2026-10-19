"""Tree genotype package for ARIEL.

Provides a JSON-serializable tree genome and helpers.
"""

from .crossover import crossover_subtree, subtree_swap
from .io import load_genome, save_genome
from .mutations import (
    mutate_hoist,
    mutate_replace_node,
    mutate_shrink,
    mutate_subtree_replacement,
)
from .structure import (
    add_node,
    get_top_ancestor,
    get_tree_depth,
    prune_invalid_edges,
    random_tree,
    remove_subtree,
    validate_tree_depth,
)
from .tree_genome import TreeGenome

__all__ = [
    "TreeGenome",
    "add_node",
    "crossover_subtree",
    "get_top_ancestor",
    "get_tree_depth",
    "load_genome",
    "mutate_hoist",
    "mutate_replace_node",
    "mutate_shrink",
    "mutate_subtree_replacement",
    "prune_invalid_edges",
    "random_tree",
    "remove_subtree",
    "save_genome",
    "subtree_swap",
    "validate_tree_depth",
]
