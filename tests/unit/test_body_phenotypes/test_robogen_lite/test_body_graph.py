"""Test: every built body keeps the graph it was built from."""

from __future__ import annotations

# Standard library
from typing import TYPE_CHECKING

# Third-party libraries
import mujoco
import networkx as nx
import pytest

# Local libraries
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots import (
    gecko,
    john_set,
    random_20,
    spider,
    spider_with_blocks,
)

if TYPE_CHECKING:
    from collections.abc import Callable

JOHN = (
    "baby_a",
    "baby_b",
    "gecko",
    "linkin_modified",
    "snake",
    "turtle",
    "iguana",
    "spider_8",
    "spider_12",
    "spider_16",
    "centipede_3",
    "centipede_4",
    "centipede_5",
)
PREBUILT: dict[str, Callable[[], CoreModule]] = {
    **{f"john_set.{name}": getattr(john_set, name) for name in JOHN},
    "gecko": gecko.gecko,
    "spider": spider.spider,
    "random_spider20": random_20.random_spider20,
    "body_spider": spider_with_blocks.body_spider,
    "body_spider45": spider_with_blocks.body_spider45,
}


def _graph(robot: CoreModule) -> nx.DiGraph[int]:
    assert robot.graph is not None
    return robot.graph


def _geom_names(robot: CoreModule) -> list[str]:
    model = robot.spec.compile()
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        for geom in range(model.ngeom)
    ]


def _line() -> nx.DiGraph[int]:
    graph: nx.DiGraph[int] = nx.DiGraph()
    graph.add_node(0, type="CORE", rotation="DEG_0")
    graph.add_node(1, type="HINGE", rotation="DEG_90")
    graph.add_node(2, type="BRICK", rotation="DEG_0")
    graph.add_edge(0, 1, face="BACK")
    graph.add_edge(1, 2, face="FRONT")
    return graph


def test_a_new_core_has_no_graph() -> None:
    assert CoreModule(index=0).graph is None


def test_the_constructor_keeps_a_copy_with_prefixes() -> None:
    graph = _line()
    robot = construct_mjspec_from_graph(graph)
    kept = _graph(robot)
    assert kept is not graph
    assert "prefix" not in graph.nodes[1]
    assert [kept.nodes[n]["prefix"] for n in (0, 1, 2)] == [
        "",
        "0-1-1-",
        "1-2-0-",
    ]
    assert kept.nodes[1]["type"] == "HINGE"
    assert kept.edges[0, 1]["face"] == "BACK"


def test_the_repair_keeps_the_pruned_graph() -> None:
    graph: nx.DiGraph[int] = nx.DiGraph()
    for node, kind in enumerate(["CORE", "BRICK", "BRICK", "HINGE"]):
        graph.add_node(node, type=kind, rotation="DEG_45" if node else "DEG_0")
    graph.add_edge(0, 1, face="FRONT")
    graph.add_edge(1, 2, face="TOP")
    graph.add_edge(2, 3, face="LEFT")
    robot = construct_mjspec_from_graph(graph, repair_overlaps=True)
    assert set(_graph(robot).nodes) < set(graph.nodes)
    assert set(graph.nodes) == {0, 1, 2, 3}


@pytest.mark.parametrize("name", sorted(PREBUILT))
def test_a_prebuilt_robot_records_every_module(name: str) -> None:
    robot = PREBUILT[name]()
    graph = _graph(robot)
    names = _geom_names(robot)
    assert nx.is_arborescence(graph)
    assert graph.nodes[0] == {"type": "CORE", "rotation": 0, "prefix": ""}
    for node in graph.nodes - {0}:
        prefix = graph.nodes[node]["prefix"]
        assert prefix
        assert any(geom.startswith(prefix) for geom in names)
        assert graph.nodes[node]["type"] in {"HINGE", "BRICK"}


def test_a_turned_hinge_records_its_degrees() -> None:
    graph = _graph(gecko.gecko())
    rotations = sorted({graph.nodes[node]["rotation"] for node in graph.nodes})
    assert rotations == [-90, -45, 0, 45, 90]
