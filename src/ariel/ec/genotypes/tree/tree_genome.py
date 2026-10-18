"""TreeGenome: minimal tree-based genome representation and conversion.

This module provides a simple JSON-serializable tree genome that converts
to a NetworkX DiGraph compatible with `construct_mjspec_from_graph`.
"""

from __future__ import annotations

import json
import typing
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx


@dataclass
class TreeGenome:
    # nodes: mapping id -> dict with keys "type" and "rotation" (Enum names)
    """Tree genome: module nodes and the edges that connect them."""

    nodes: dict[int, dict[str, str]] = field(default_factory=dict)
    # edges: list of {"parent": int, "child": int, "face": ModuleFaces.name}
    edges: list[dict[str, typing.Any]] = field(default_factory=list)

    def to_networkx(self) -> nx.DiGraph[int]:
        """Return the genome as a networkx DiGraph.

        Returns
        -------
        nx.DiGraph[int]
            The genome as a directed graph.
        """
        g: nx.DiGraph[int] = nx.DiGraph()
        for nid, nobj in self.nodes.items():
            g.add_node(nid, type=nobj["type"], rotation=nobj["rotation"])
        for e in self.edges:
            g.add_edge(e["parent"], e["child"], face=e["face"])
        return g

    def to_dict(self) -> dict[str, typing.Any]:
        """Return the genome as a JSON-serialisable dict.

        Returns
        -------
        dict[str, typing.Any]
            The genome as a JSON-serialisable dict.
        """
        return {
            "nodes": {str(k): v for k, v in self.nodes.items()},
            "edges": self.edges,
        }

    @classmethod
    def from_dict(cls, data: dict[str, typing.Any]) -> TreeGenome:
        """Create a genome from a dict as returned by `to_dict`.

        Returns
        -------
        TreeGenome
            The new genome.
        """
        nodes_raw = data.get("nodes", {})
        # accept both string-keyed mapping and list
        nodes: dict[int, dict[str, str]] = {}
        if isinstance(nodes_raw, dict):
            for k, v in nodes_raw.items():
                nodes[int(k)] = {"type": v["type"], "rotation": v["rotation"]}
        elif isinstance(nodes_raw, list):
            for entry in nodes_raw:
                nid = int(entry["id"])
                nodes[nid] = {
                    "type": entry["type"],
                    "rotation": entry["rotation"],
                }
        edges = data.get("edges", [])
        return cls(nodes=nodes, edges=edges)

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
