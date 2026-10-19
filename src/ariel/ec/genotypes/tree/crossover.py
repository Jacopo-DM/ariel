# ruff: file-ignore[suspicious-non-cryptographic-random-usage]  # pseudo-random numbers drive the search, no security use
"""Subtree crossover for tree genomes."""

from __future__ import annotations

import copy
import random
from typing import TYPE_CHECKING

import networkx as nx

from ariel.body_phenotypes import IDX_OF_CORE

from .structure import prune_invalid_edges, reassign_ids, remove_subtree
from .validation import validate_genome_dict

if TYPE_CHECKING:
    from .tree_genome import TreeEdge, TreeGenome, TreeNode


def subtree_swap(
    a: TreeGenome, b: TreeGenome, a_node: int, b_node: int
) -> None:
    """Swap subtrees rooted at *a_node* and *b_node* between two genomes.

    The subtrees are detached from their original parents and then inserted into
    the opposite genome.  To avoid identifier collisions we remap all node IDs
    in each subtree before reinsertion.  Only internal edges (both parent and
    child in the subtree) are carried across; any external connections are
    discarded.  After the exchange the two genomes are pruned and validated.
    """

    def extract_subtree(
        genome: TreeGenome, root: int
    ) -> tuple[dict[int, TreeNode], list[TreeEdge]]:
        g = genome.to_networkx()
        subnodes = {root, *nx.descendants(g, root)}
        internal_edges = [
            e
            for e in genome.edges
            if e.parent in subnodes and e.child in subnodes
        ]
        nodes = {n: genome.nodes[n] for n in subnodes}
        return nodes, internal_edges

    a_nodes, a_edges = extract_subtree(a, a_node)
    b_nodes, b_edges = extract_subtree(b, b_node)

    # remove the chosen branches from their parents
    remove_subtree(a, a_node)
    remove_subtree(b, b_node)

    # determine occupied ids after removal
    a_existing = set(a.nodes.keys())
    b_existing = set(b.nodes.keys())

    # reassign identifiers before inserting into the other genome
    b_nodes_new, b_edges_new, _ = reassign_ids(b_nodes, b_edges, a_existing)
    a_nodes_new, a_edges_new, _ = reassign_ids(a_nodes, a_edges, b_existing)

    a.nodes.update(b_nodes_new)
    a.edges.extend(b_edges_new)
    b.nodes.update(a_nodes_new)
    b.edges.extend(a_edges_new)

    # clean up any now-invalid connections and perform final checks
    prune_invalid_edges(a)
    prune_invalid_edges(b)


def crossover_subtree(
    a: TreeGenome, b: TreeGenome
) -> tuple[TreeGenome, TreeGenome]:
    """Perform standard GP subtree crossover on tree genomes.

    Two offspring are produced by exchanging randomly chosen subtrees from each
    parent.
    Unlike one-point crossover on linear genomes, this selects any node
    (including leaves)
    and swaps the entire subtree rooted at that node. Returns a pair ``(child1,
    child2)``.
    Parents are left unmodified.

    Returns
    -------
    tuple[TreeGenome, TreeGenome]
        The two children.
    """
    child1 = copy.deepcopy(a)
    child2 = copy.deepcopy(b)

    def pick_noncore(gen: TreeGenome) -> int | None:
        candidates = [nid for nid in gen.nodes if nid != IDX_OF_CORE]
        return random.choice(candidates) if candidates else None

    n1 = pick_noncore(child1)
    n2 = pick_noncore(child2)
    if n1 is None or n2 is None:
        return child1, child2

    # Standard GP: swap subtrees directly at selected nodes (no ancestor
    # expansion)
    subtree_swap(child1, child2, n1, n2)
    # Validate after crossover

    try:
        validate_genome_dict(child1.to_dict())
        validate_genome_dict(child2.to_dict())
    except ValueError:
        # If validation fails, return copies of parents
        return copy.deepcopy(a), copy.deepcopy(b)
    return child1, child2
