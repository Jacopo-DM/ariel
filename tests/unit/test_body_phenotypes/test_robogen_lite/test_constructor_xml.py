"""Test: the XML that construct_mjspec_from_graph gives for fixed graphs.

The hashes pin the bodies as built today (no repair), so a later change to the
constructor that alters a body shows up here first.
"""

from __future__ import annotations

import hashlib

import networkx as nx
import pytest

from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)

type Spec = tuple[dict[int, tuple[str, str]], list[tuple[int, int, str]]]

CORE_ONLY: Spec = ({0: ("CORE", "DEG_0")}, [])
SNAKE: Spec = (
    {
        0: ("CORE", "DEG_0"),
        1: ("BRICK", "DEG_0"),
        2: ("HINGE", "DEG_90"),
        3: ("BRICK", "DEG_0"),
    },
    [(0, 1, "FRONT"), (1, 2, "FRONT"), (2, 3, "FRONT")],
)
CROSS: Spec = (
    {
        0: ("CORE", "DEG_0"),
        1: ("BRICK", "DEG_0"),
        2: ("BRICK", "DEG_90"),
        3: ("HINGE", "DEG_45"),
        4: ("BRICK", "DEG_45"),
    },
    [(0, 1, "FRONT"), (0, 2, "BACK"), (0, 3, "RIGHT"), (0, 4, "LEFT")],
)
OVERLAPPING: Spec = (
    {
        0: ("CORE", "DEG_0"),
        1: ("BRICK", "DEG_45"),
        2: ("BRICK", "DEG_45"),
        3: ("HINGE", "DEG_45"),
    },
    [(0, 1, "FRONT"), (1, 2, "TOP"), (2, 3, "LEFT")],
)

XML_HASHES = {
    "core only": (
        "140cab4a30584af450e181e3ffa4cb5ae08adac719ea88e7431b71def3fb8fb4"
    ),
    "snake": (
        "84d0bb1f3fd7b39f5db8568578e3d655a8572de4307d2770fb7dd3b17a1d0060"
    ),
    "cross": (
        "4e09681f88d2efbcf3e57b32a8c147ebbc1db84d8f9291a6daf99e6336470bb6"
    ),
    "overlapping": (
        "4f4a32dca5f381a92af923312ea2da5071f512bcf43b3914518a5dc3789dc6f1"
    ),
}
SPECS = {
    "core only": CORE_ONLY,
    "snake": SNAKE,
    "cross": CROSS,
    "overlapping": OVERLAPPING,
}


def _graph(spec: Spec) -> nx.DiGraph[int]:
    graph: nx.DiGraph[int] = nx.DiGraph()
    for node, (kind, rotation) in spec[0].items():
        graph.add_node(node, type=kind, rotation=rotation)
    for parent, child, face in spec[1]:
        graph.add_edge(parent, child, face=face)
    return graph


@pytest.mark.parametrize("name", list(SPECS))
def test_the_body_xml_is_pinned(name: str) -> None:
    xml = construct_mjspec_from_graph(_graph(SPECS[name])).spec.to_xml()
    assert hashlib.sha256(xml.encode()).hexdigest() == XML_HASHES[name]
