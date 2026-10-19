# ruff: file-ignore[suspicious-non-cryptographic-random-usage]  # pseudo-random numbers drive the search, no security use
"""Structure helpers for tree genomes: nodes, subtrees, ids, depth."""

from __future__ import annotations

import random

import networkx as nx

from ariel.body_phenotypes import (
    ALLOWED_FACES,
    ALLOWED_ROTATIONS,
    IDX_OF_CORE,
    ModuleType,
)

from .tree_genome import TreeEdge, TreeGenome, TreeNode


def add_node(
    genome: TreeGenome,
    parent: int,
    face: str,
    node_id: int,
    mtype: str,
    rotation: str,
) -> None:
    # add node and edge if allowed; caller should ensure face is free
    """Add node `node_id` and an edge from `parent` on `face`."""
    genome.nodes[node_id] = TreeNode(type=mtype, rotation=rotation)
    genome.edges.append(TreeEdge(parent=parent, child=node_id, face=face))


def remove_subtree(genome: TreeGenome, node_id: int) -> None:
    # remove node and all descendants
    """Remove node `node_id` and all of its descendants."""
    g = genome.to_networkx()
    if node_id not in g:
        return
    descendants = list(nx.descendants(g, node_id))
    descendants.append(node_id)
    for d in descendants:
        genome.nodes.pop(d, None)
    genome.edges = [
        e
        for e in genome.edges
        if e.child not in descendants and e.parent not in descendants
    ]


def reassign_ids(
    nodes: dict[int, TreeNode],
    edges: list[TreeEdge],
    existing_ids: set[int],
) -> tuple[dict[int, TreeNode], list[TreeEdge], dict[int, int]]:
    """Return copies of *nodes* and *edges* with fresh identifiers.

    IDs are remapped to the smallest integers greater than any value in
    *existing_ids*.  A mapping dict is also returned but only callers that
    need to translate external references will use it.

    Returns
    -------
    tuple[dict[int, TreeNode], list[TreeEdge], dict[int, int]]
        The new nodes, the new edges and the map from old to new ids.
    """
    mapping: dict[int, int] = {}
    next_id = max(existing_ids, default=-1) + 1
    for old in sorted(nodes):
        mapping[old] = next_id
        next_id += 1
    new_nodes = {
        mapping[old]: attr.model_copy(deep=True) for old, attr in nodes.items()
    }
    # model_copy keeps any additional keys (face/rotation/etc.)
    new_edges = [
        e.model_copy(
            update={"parent": mapping[e.parent], "child": mapping[e.child]},
            deep=True,
        )
        for e in edges
    ]
    return new_nodes, new_edges, mapping


def get_top_ancestor(genome: TreeGenome, node_id: int) -> int:
    """Return the highest ancestor of ``node_id`` (stop at core or no parent).

    This walks predecessors in the directed graph until reaching a node whose
    parent is the core (IDX_OF_CORE) or there is no parent. Useful for
    selecting a subtree root that contains a full branch rather than a leaf.

    Returns
    -------
    int
        The id of the highest ancestor.
    """
    g = genome.to_networkx()
    current = node_id
    # climb while there is a single parent that is not the core
    while True:
        preds = list(g.predecessors(current))
        if not preds:
            break
        parent = preds[0]
        # stop if parent is core
        if parent == IDX_OF_CORE:
            break
        current = parent
    return current


def random_tree(max_modules: int) -> TreeGenome:
    """Generate a random valid tree genome with up to ``max_modules`` nodes.

    Starts with only the core and iteratively adds a random module on a free
    face until the budget is exhausted or no free faces remain.

    Returns
    -------
    TreeGenome
        The new tree genome.
    """
    g = TreeGenome()
    g.nodes = {IDX_OF_CORE: TreeNode(type="CORE", rotation="DEG_0")}
    g.edges = []

    next_id = 1
    while next_id <= max_modules:
        # collect available (parent,face) positions
        free: list[tuple[int, str]] = []
        for pid, pdata in g.nodes.items():
            ptype = ModuleType[pdata.type]
            allowed_faces = [f.name for f in ALLOWED_FACES[ptype]]
            used = {e.face for e in g.edges if e.parent == pid}
            free.extend(
                (pid, face) for face in allowed_faces if face not in used
            )
        parent, face = random.choice(free)
        # choose random module type/rotation
        types = [
            t for t in ModuleType if t not in {ModuleType.CORE, ModuleType.NONE}
        ]
        mtype = random.choice(types).name
        rotations = [r.name for r in ALLOWED_ROTATIONS[ModuleType[mtype]]]
        rot = random.choice(rotations) if rotations else "DEG_0"
        add_node(g, parent, face, next_id, mtype, rot)
        next_id += 1
    return g


def get_tree_depth(genome: TreeGenome) -> int:
    """Calculate the maximum depth from core to any leaf node.

    Returns
    -------
    int
        The depth, counted in edges from the core.
    """
    if not genome.nodes:
        return 0

    g = genome.to_networkx()

    max_depth = 0
    stack = [(IDX_OF_CORE, 0)]  # (node, depth)

    while stack:
        node, depth = stack.pop()
        max_depth = max(max_depth, depth)

        stack.extend((child, depth + 1) for child in g.successors(node))

    return max_depth


def validate_tree_depth(genome: TreeGenome, max_depth: int) -> bool:
    """Check if tree depth is within the specified limit.

    Returns
    -------
    bool
        True if the depth is within the limit.
    """
    return get_tree_depth(genome) <= max_depth


def prune_invalid_edges(genome: TreeGenome) -> None:
    """Remove any edges that violate face constraints or are duplicated.

    This is a safety check that filters out edges where the parent type does
    not allow the specified face, removes duplicate parent-face pairs, and
    drops any edges pointing at nonexistent nodes.  Child subtrees flagged for
    deletion are removed only after the scan to avoid modifying the edge list
    during iteration.
    """
    seen = set()  # (parent, face) tuples
    valid_edges: list[TreeEdge] = []
    to_prune: list[int] = []

    # iterate over a snapshot of edges to safely modify genome later
    for e in list(genome.edges):
        parent_id = e.parent
        child_id = e.child
        # drop edges whose parent or child no longer exist
        if parent_id not in genome.nodes or child_id not in genome.nodes:
            continue
        parent_type = ModuleType[genome.nodes[parent_id].type]
        allowed = [f.name for f in ALLOWED_FACES[parent_type]]
        edge_key = (parent_id, e.face)

        # Check 1: face must be allowed for parent type
        if e.face not in allowed:
            to_prune.append(child_id)
            continue

        # Check 2: face must not already be occupied (no duplicates)
        if edge_key in seen:
            to_prune.append(child_id)
            continue

        seen.add(edge_key)
        valid_edges.append(e)

    # remove any subtrees flagged for pruning
    for cid in set(to_prune):
        remove_subtree(genome, cid)

    # keep only edges that still reference existing nodes after subtree pruning
    genome.edges = [
        e
        for e in valid_edges
        if e.parent in genome.nodes and e.child in genome.nodes
    ]
