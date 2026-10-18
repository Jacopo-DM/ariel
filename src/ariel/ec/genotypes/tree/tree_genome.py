"""TreeGenome: minimal tree-based genome representation and conversion.

This module provides a simple JSON-serializable tree genome that converts
to a NetworkX DiGraph compatible with `construct_mjspec_from_graph`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from ariel.parameters import MutableModel

if TYPE_CHECKING:
    from collections.abc import Mapping


class TreeNode(MutableModel):
    """A module of the body: its type and rotation (``Enum`` member names)."""

    type: str
    rotation: str


class TreeEdge(MutableModel):
    """A connection from a parent module to a child on one parent face.

    Extra keys that come with an edge are kept and written back unchanged.
    """

    model_config = ConfigDict(extra="allow")

    parent: int
    child: int
    face: str  # ``ModuleFaces`` member name


class _SerializedNode(BaseModel):
    """A node as stored in a dict or JSON file; unknown keys are dropped."""

    id: int | None = None
    type: str
    rotation: str


class _SerializedGenome(BaseModel):
    """A genome as stored: nodes keyed by id, or listed with an ``id`` each."""

    nodes: dict[int, _SerializedNode] | list[_SerializedNode] = Field(
        default_factory=dict
    )
    edges: list[TreeEdge] = Field(default_factory=list)


class TreeGenome(MutableModel):
    """Tree genome: module nodes and the edges that connect them."""

    nodes: dict[int, TreeNode] = Field(default_factory=dict)
    edges: list[TreeEdge] = Field(default_factory=list)

    def to_networkx(self) -> nx.DiGraph[int]:
        """Return the genome as a networkx DiGraph.

        Returns
        -------
        nx.DiGraph[int]
            The genome as a directed graph.
        """
        g: nx.DiGraph[int] = nx.DiGraph()
        for nid, nobj in self.nodes.items():
            g.add_node(nid, type=nobj.type, rotation=nobj.rotation)
        for e in self.edges:
            g.add_edge(e.parent, e.child, face=e.face)
        return g

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the genome as a JSON-serialisable dict.

        Returns
        -------
        dict[str, JsonValue]
            The genome as a JSON-serialisable dict.
        """
        return {
            "nodes": {str(k): v.model_dump() for k, v in self.nodes.items()},
            "edges": [e.model_dump() for e in self.edges],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> TreeGenome:
        """Create a genome from a dict as returned by `to_dict`.

        Returns
        -------
        TreeGenome
            The new genome.

        Raises
        ------
        ValueError
            If a listed node has no id.
        """
        raw = _SerializedGenome.model_validate(data)
        nodes: dict[int, TreeNode] = {}
        # accept both string-keyed mapping and list
        if isinstance(raw.nodes, dict):
            for nid, entry in raw.nodes.items():
                nodes[nid] = TreeNode(type=entry.type, rotation=entry.rotation)
        else:
            for entry in raw.nodes:
                if entry.id is None:
                    msg = "Every listed node needs an 'id'"
                    raise ValueError(msg)
                nodes[entry.id] = TreeNode(
                    type=entry.type, rotation=entry.rotation
                )
        return cls(nodes=nodes, edges=raw.edges)

    def save_json(self, path: str) -> None:
        """Write the genome to `path` as JSON."""
        with Path(path).open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load_json(cls, path: str) -> TreeGenome:
        """Read a genome from the JSON file at `path`.

        Returns
        -------
        TreeGenome
            The genome read from the file.
        """
        with Path(path).open(encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_dict(data)
