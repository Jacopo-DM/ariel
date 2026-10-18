"""Validation utilities for tree genomes.

Functions here check required fields and enforce allowed faces/rotations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from ariel.body_phenotypes.robogen_lite.config import (
    ALLOWED_FACES,
    IDX_OF_CORE,
    ModuleFaces,
    ModuleRotationsIdx,
    ModuleType,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


class _RawNode(BaseModel):
    """A node of a genome dict; the type and rotation are checked later."""

    type: str = ""
    rotation: str = ""


class _RawEdge(BaseModel):
    """An edge of a genome dict; the face is checked later."""

    parent: int
    child: int
    face: str


class _RawGenome(BaseModel):
    """A genome dict with string or integer node ids."""

    nodes: dict[int, _RawNode] = Field(default_factory=dict)
    edges: list[_RawEdge] = Field(default_factory=list)


def is_single_connected_tree(genome: Mapping[str, object]) -> bool:
    """Verify that the genome forms a single connected tree with core as root.

    In this domain, the core (IDX_OF_CORE) must be the root and all nodes
    must be reachable from it.

    Returns
    -------
    bool
        True if the genome is one connected tree rooted at the core.
    """
    raw = _RawGenome.model_validate(genome)

    if not raw.nodes:
        return True  # Empty genome is trivially connected

    # node ids are ints even when the dict came from JSON with string keys
    node_ids = set(raw.nodes)

    # Core must exist and be the root (no incoming edges)
    if IDX_OF_CORE not in node_ids:
        return False

    # Build adjacency list and check core has no parents
    adj_list: dict[int, list[int]] = {nid: [] for nid in node_ids}
    has_parent = dict.fromkeys(node_ids, False)

    for e in raw.edges:
        if e.parent in adj_list and e.child in adj_list:
            adj_list[e.parent].append(e.child)
            has_parent[e.child] = True

    # Core must have no parent
    if has_parent[IDX_OF_CORE]:
        return False

    # Check connectivity: all nodes reachable from core
    visited = set()
    stack = [IDX_OF_CORE]

    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        stack.extend(child for child in adj_list[node] if child not in visited)

    # All nodes must be visited
    return len(visited) == len(node_ids)


def validate_genome_dict(genome: Mapping[str, object]) -> None:  # ruff: ignore[complex-structure]
    """Check a genome dict for consistency; raise ValueError if invalid.

    Raises
    ------
    ValueError
        If the genome has no core, a bad type, rotation or face, a missing
        parent, a repeated face, or is not one connected tree.
    """
    raw = _RawGenome.model_validate(genome)

    # nodes must include core at IDX_OF_CORE
    if IDX_OF_CORE not in raw.nodes:
        msg = f"Genome must contain core node with index {IDX_OF_CORE}"
        raise ValueError(msg)

    # Validate node fields
    for nid, node in raw.nodes.items():
        if node.type not in ModuleType.__members__:
            msg = f"Node {nid} has invalid type '{node.type}'"
            raise ValueError(msg)
        if node.rotation not in ModuleRotationsIdx.__members__:
            msg = f"Node {nid} has invalid rotation '{node.rotation}'"
            raise ValueError(msg)

    # Validate edges
    occupied: dict[tuple[int, str], int] = {}  # (parent, face) -> child
    for e in raw.edges:
        # face must be valid
        if e.face not in ModuleFaces.__members__:
            msg = (
                f"Edge parent={e.parent} child={e.child} "
                f"has invalid face '{e.face}'"
            )
            raise ValueError(msg)
        # skip edges whose parent node no longer exists (shouldn't happen
        # normally, but may after swaps); treat as invalid structure rather than
        # raising KeyError.
        if e.parent not in raw.nodes:
            msg = f"Edge has missing parent node {e.parent}"
            raise ValueError(msg)
        # parent type must allow that face
        ptype = raw.nodes[e.parent].type
        allowed = [f.name for f in ALLOWED_FACES[ModuleType[ptype]]]
        if e.face not in allowed:
            msg = f"Face '{e.face}' not allowed for parent type '{ptype}'"
            raise ValueError(msg)
        # ensure a face is not occupied twice
        key = (e.parent, e.face)
        if key in occupied:
            msg = f"Parent {e.parent} already has child at face '{e.face}'"
            raise ValueError(msg)
        occupied[key] = e.child

    # Validate connectivity: must form a single connected tree with core as root
    if not is_single_connected_tree(genome):
        msg = "Genome does not form a single connected tree with core as root"
        raise ValueError(msg)
