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

# sha256 of each body's XML; taken again on 2026-10-21, when the modules got
# their measured sizes and masses (physical-checks PC3), and on 2026-10-22,
# when every hinge became a DSS-M15 (servo-dynamics SD2; the core alone has
# no hinge and kept its hash)
XML_HASHES = {
    "core only": (
        "869d7890024e371dff9f612d53f095e7bedbf470bf403fbe3440194c41a5f028"
    ),
    "snake": (
        "1018ba9f0d365b27431d3410787f15e2cf352583edea0289b14dc603e25cc40e"
    ),
    "cross": (
        "c849a6dea836d98fc5c3eb1ad618177643a93f2ef72d8dc5678e4f982c23846c"
    ),
    "overlapping": (
        "2a937011198c5d491e07eb447acb69d4c74426925bb1c2873e804ecc97b67cc0"
    ),
}
SPECS = {
    "core only": CORE_ONLY,
    "snake": SNAKE,
    "cross": CROSS,
    "overlapping": OVERLAPPING,
}


def graph_from(spec: Spec) -> nx.DiGraph[int]:
    graph: nx.DiGraph[int] = nx.DiGraph()
    for node, (kind, rotation) in spec[0].items():
        graph.add_node(node, type=kind, rotation=rotation)
    for parent, child, face in spec[1]:
        graph.add_edge(parent, child, face=face)
    return graph


@pytest.mark.parametrize("name", list(SPECS))
def test_the_body_xml_is_pinned(name: str) -> None:
    xml = construct_mjspec_from_graph(graph_from(SPECS[name])).spec.to_xml()
    assert hashlib.sha256(xml.encode()).hexdigest() == XML_HASHES[name]
