# ruff: file-ignore[suspicious-non-cryptographic-random-usage]  # pseudo-random numbers drive the search, no security use
"""Mutation operators for tree genomes."""

from __future__ import annotations

import copy
import random

import networkx as nx

from ariel.body_phenotypes import (
    ALLOWED_FACES,
    ALLOWED_ROTATIONS,
    IDX_OF_CORE,
    ModuleType,
)

from .structure import (
    prune_invalid_edges,
    random_tree,
    reassign_ids,
    remove_subtree,
)
from .tree_genome import TreeEdge, TreeGenome, TreeNode
from .validation import validate_genome_dict


def mutate_replace_node(genome: TreeGenome) -> None:
    """Point mutation: replace a randomly selected non-core node.

    The node's type and rotation are chosen uniformly at random from the
    allowed options (excluding CORE/NONE).  Any children attached to faces not
    permitted by the new type are deleted along with their subtrees.  The
    genome is validated at the end; invalid mutations are simply no-ops.
    """
    # transactional backup: restore if mutation cannot be validated
    old_nodes = copy.deepcopy(genome.nodes)
    old_edges = copy.deepcopy(genome.edges)

    candidates = [nid for nid in genome.nodes if nid != IDX_OF_CORE]
    if not candidates:
        return

    nid = random.choice(candidates)
    old_type = genome.nodes[nid].type

    # choose a new type different from current; exclude CORE/NONE
    types = [
        t for t in ModuleType if t not in {ModuleType.CORE, ModuleType.NONE}
    ]
    types = [t for t in types if t.name != old_type]
    new_type = random.choice(types).name

    # choose a rotation allowed for the new type
    rotations = [r.name for r in ALLOWED_ROTATIONS[ModuleType[new_type]]]
    new_rot = (
        random.choice(rotations) if rotations else genome.nodes[nid].rotation
    )

    genome.nodes[nid] = TreeNode(type=new_type, rotation=new_rot)

    # drop any children on now-disallowed faces
    allowed = [f.name for f in ALLOWED_FACES[ModuleType[new_type]]]
    for e in list(genome.edges):
        if e.parent == nid and e.face not in allowed:
            remove_subtree(genome, e.child)

    # validate
    try:
        prune_invalid_edges(genome)
        validate_genome_dict(genome.to_dict())
    except ValueError:
        genome.nodes = old_nodes
        genome.edges = old_edges


def mutate_subtree_replacement(  # ruff: ignore[too-many-locals]  # long routine kept as written; splitting it is a refactor outside this goal
    genome: TreeGenome, max_modules: int = 10
) -> None:
    """
    Apply standard GP subtree mutation: replace a subtree with a new one.

    Selects a random non-core node, removes its entire subtree, and replaces it
    with a new randomly generated subtree. This is the primary mutation operator
    in canonical GP (Koza, 1992).
    """
    # transactional backup: restore if mutation cannot be validated
    old_nodes = copy.deepcopy(genome.nodes)
    old_edges = copy.deepcopy(genome.edges)

    candidates = [nid for nid in genome.nodes if nid != IDX_OF_CORE]
    if not candidates:
        return

    # Select node to replace
    node_id = random.choice(candidates)

    # Find parent and face connection
    parent_id = None
    parent_face = None
    for e in genome.edges:
        if e.child == node_id:
            parent_id = e.parent
            parent_face = e.face
            break

    # Remove the old subtree
    remove_subtree(genome, node_id)

    # Generate new random subtree and attach it
    if parent_id is not None and parent_face is not None:
        # Create a small random subtree (1-3 nodes typically)
        subtree_size = random.randint(1, min(3, max_modules))
        new_subtree = random_tree(subtree_size)

        # The generated subtree has its own core. We graft a single branch under
        # the original parent/face to preserve tree connectivity.
        g_sub = new_subtree.to_networkx()
        core_children = [
            e.child for e in new_subtree.edges if e.parent == IDX_OF_CORE
        ]

        root_child = random.choice(core_children)
        branch_nodes = {root_child, *nx.descendants(g_sub, root_child)}
        branch_node_dict = {
            n: new_subtree.nodes[n].model_copy(deep=True) for n in branch_nodes
        }
        branch_edges = [
            copy.deepcopy(e)
            for e in new_subtree.edges
            if e.parent in branch_nodes and e.child in branch_nodes
        ]

        existing_ids = set(genome.nodes.keys())
        new_nodes, new_edges, mapping = reassign_ids(
            branch_node_dict, branch_edges, existing_ids
        )

        genome.nodes.update(new_nodes)
        genome.edges.extend(new_edges)
        genome.edges.append(
            TreeEdge(
                parent=parent_id,
                child=mapping[root_child],
                face=parent_face,
            )
        )

    # Clean up and validate
    prune_invalid_edges(genome)
    try:
        validate_genome_dict(genome.to_dict())
    except ValueError:
        genome.nodes = old_nodes
        genome.edges = old_edges


def mutate_shrink(genome: TreeGenome) -> None:
    """
    Shrink mutation: replace a node and its subtree with a single new leaf node.

    Selects a random non-core node, removes its entire subtree, and replaces it
    with a single new randomly chosen module type. This reduces tree size and
    is part of standard GP mutation operators.
    """
    # transactional backup: restore if mutation cannot be validated
    old_nodes = copy.deepcopy(genome.nodes)
    old_edges = copy.deepcopy(genome.edges)

    candidates = [nid for nid in genome.nodes if nid != IDX_OF_CORE]
    if not candidates:
        return

    node_id = random.choice(candidates)

    # Find parent connection
    parent_id = None
    parent_face = None
    for e in genome.edges:
        if e.child == node_id:
            parent_id = e.parent
            parent_face = e.face
            break

    if parent_id is None or parent_face is None:
        return  # Shouldn't happen for valid trees

    # Remove the entire subtree
    remove_subtree(genome, node_id)

    # Replace with a single new node
    types = [
        t for t in ModuleType if t not in {ModuleType.CORE, ModuleType.NONE}
    ]
    new_type = random.choice(types).name
    rotations = [r.name for r in ALLOWED_ROTATIONS[ModuleType[new_type]]]
    new_rot = random.choice(rotations) if rotations else "DEG_0"

    genome.nodes[node_id] = TreeNode(type=new_type, rotation=new_rot)
    genome.edges.append(
        TreeEdge(parent=parent_id, child=node_id, face=parent_face)
    )

    # Clean up and validate
    prune_invalid_edges(genome)
    try:
        validate_genome_dict(genome.to_dict())
    except ValueError:
        genome.nodes = old_nodes
        genome.edges = old_edges


def mutate_hoist(genome: TreeGenome) -> None:
    """Hoist mutation: replace a parent node with one of its children.

    Selects a random node that has children, then replaces it with one of its
    children, effectively "hoisting" the child up in the tree. This changes
    tree structure without changing size and is part of standard GP operators.
    """
    # transactional backup: restore if mutation cannot be validated
    old_nodes = copy.deepcopy(genome.nodes)
    old_edges = copy.deepcopy(genome.edges)

    # Find nodes that have children
    nodes_with_children = {e.parent for e in genome.edges}

    candidates = [nid for nid in nodes_with_children if nid != IDX_OF_CORE]
    if not candidates:
        return

    parent_id = random.choice(candidates)

    # Get all children of this parent
    children = [e.child for e in genome.edges if e.parent == parent_id]

    # Choose one child to hoist
    child_to_hoist = random.choice(children)

    # Find the parent's parent and face
    grandparent_id = None
    parent_face = None
    for e in genome.edges:
        if e.child == parent_id:
            grandparent_id = e.parent
            parent_face = e.face
            break

    if grandparent_id is None or parent_face is None:
        return  # Can't hoist root

    # Remove the parent and all its other children (except the one being
    # hoisted)
    for child in children:
        if child != child_to_hoist:
            remove_subtree(genome, child)

    # Remove the parent node and its edges
    if parent_id in genome.nodes:
        del genome.nodes[parent_id]
    genome.edges = [
        e for e in genome.edges if parent_id not in {e.child, e.parent}
    ]

    # Connect the hoisted child directly to the grandparent
    genome.edges = [
        e
        for e in genome.edges
        if not (e.parent == grandparent_id and e.face == parent_face)
    ]
    genome.edges.append(
        TreeEdge(parent=grandparent_id, child=child_to_hoist, face=parent_face)
    )

    # Clean up and validate
    prune_invalid_edges(genome)
    try:
        validate_genome_dict(genome.to_dict())
    except ValueError:
        genome.nodes = old_nodes
        genome.edges = old_edges
