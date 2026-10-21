"""Test: the morphological measures of a robot graph."""

from __future__ import annotations

# Standard library
# Third-party libraries
import networkx as nx
import pytest

# Local libraries
from ariel.body_phenotypes import (
    CoreModule,
    MorphologicalMeasures,
    construct_mjspec_from_graph,
)

SPIDER_HINGES = 4
SPIDER_MODULES = 5
BRANCHING_BRICKS = 2

# ---------------------------------------------------------------------------
# MorphologicalMeasures
# ---------------------------------------------------------------------------


def _make_single_core_graph() -> nx.DiGraph[int]:
    """Minimal graph: only a CORE node.

    Returns
    -------
    nx.DiGraph[int]
        The graph.
    """
    g: nx.DiGraph[int] = nx.DiGraph()
    g.add_node(0, type="CORE", rotation="DEG_0")
    return g


def _make_spider_graph() -> nx.DiGraph[int]:
    """Spider-like graph: core + 4 hinge arms.

    Returns
    -------
    nx.DiGraph[int]
        The graph.
    """
    g: nx.DiGraph[int] = nx.DiGraph()
    g.add_node(0, type="CORE", rotation="DEG_0")
    for i, face in enumerate(["FRONT", "BACK", "LEFT", "RIGHT"], start=1):
        g.add_node(i, type="HINGE", rotation="DEG_0")
        g.add_edge(0, i, face=face)
    return g


def _make_branching_graph() -> nx.DiGraph[int]:
    """Core with two bricks attached and hinges on the bricks.

    Returns
    -------
    nx.DiGraph[int]
        The graph.
    """
    g: nx.DiGraph[int] = nx.DiGraph()
    g.add_node(0, type="CORE", rotation="DEG_0")
    g.add_node(1, type="BRICK", rotation="DEG_0")
    g.add_node(2, type="BRICK", rotation="DEG_0")
    g.add_node(3, type="HINGE", rotation="DEG_0")
    g.add_edge(0, 1, face="FRONT")
    g.add_edge(0, 2, face="BACK")
    g.add_edge(1, 3, face="FRONT")
    return g


def _measure(graph: nx.DiGraph[int]) -> MorphologicalMeasures:
    return MorphologicalMeasures(construct_mjspec_from_graph(graph))


def test_morphological_empty_graph_raises() -> None:
    """MorphologicalMeasures raises ValueError for an empty graph."""
    robot = construct_mjspec_from_graph(_make_single_core_graph())
    robot.graph = nx.DiGraph()
    with pytest.raises(ValueError, match="empty"):
        MorphologicalMeasures(robot)


def test_a_robot_without_a_graph_raises() -> None:
    """A core made by hand, with no builder, cannot be measured."""
    with pytest.raises(ValueError, match="no graph"):
        MorphologicalMeasures(CoreModule(index=0))


def test_morphological_single_core_num_modules() -> None:
    """A graph with only a core has num_modules == 1."""
    m = _measure(_make_single_core_graph())
    assert m.num_modules == 1


def test_morphological_single_core_no_bricks_or_hinges() -> None:
    """A single-core graph has no bricks or active hinges."""
    m = _measure(_make_single_core_graph())
    assert m.num_bricks == 0
    assert m.num_active_hinges == 0


def test_morphological_spider_graph_module_counts() -> None:
    """Spider graph has 4 hinges and 0 bricks."""
    m = _measure(_make_spider_graph())
    assert m.num_active_hinges == SPIDER_HINGES
    assert m.num_bricks == 0
    assert m.num_modules == SPIDER_MODULES


def test_morphological_branching_graph_brick_count() -> None:
    """Branching graph has 2 bricks and 1 hinge."""
    m = _measure(_make_branching_graph())
    assert m.num_bricks == BRANCHING_BRICKS
    assert m.num_active_hinges == 1


def test_morphological_symmetry_in_range() -> None:
    """All symmetry scores are in [0, 1]."""
    m = _measure(_make_spider_graph())
    assert 0.0 <= m.xy_symmetry <= 1.0
    assert 0.0 <= m.xz_symmetry <= 1.0
    assert 0.0 <= m.yz_symmetry <= 1.0
    assert 0.0 <= m.symmetry <= 1.0


def test_morphological_bounding_box_positive() -> None:
    """Bounding box dimensions are positive lengths in metres."""
    m = _measure(_make_spider_graph())
    assert 0.0 < m.bounding_box_depth < 1.0
    assert 0.0 < m.bounding_box_width < 1.0
    assert 0.0 < m.bounding_box_height < 1.0


def test_morphological_coverage_in_range() -> None:
    """Coverage (C) is in (0, 1]."""
    m = _measure(_make_spider_graph())
    assert 0.0 < m.C <= 1.0


def test_morphological_joints_ratio_in_range() -> None:
    """Joint ratio J is in [0, 1]."""
    m = _measure(_make_spider_graph())
    assert 0.0 <= m.J <= 1.0


def test_morphological_limbs_in_range() -> None:
    """Limbs ratio L is in [0, 1]."""
    m = _measure(_make_spider_graph())
    assert 0.0 <= m.L <= 1.0


def test_morphological_proportion_in_range() -> None:
    """Proportion P is in (0, 1]."""
    m = _measure(_make_spider_graph())
    assert 0.0 < m.P <= 1.0


def test_morphological_is_2d_flag() -> None:
    """A spider with its legs in the core's plane is flat."""
    m = _measure(_make_spider_graph())
    assert m.is_2d


def test_morphological_paper_aliases_consistent() -> None:
    """Paper-letter aliases agree with their named property counterparts."""
    m = _measure(_make_branching_graph())
    assert pytest.approx(m.branching) == m.B
    assert pytest.approx(m.limbs) == m.L
    assert pytest.approx(m.length_of_limbs) == m.E
    assert pytest.approx(m.coverage) == m.C
    assert pytest.approx(m.joints) == m.J
    assert pytest.approx(m.symmetry) == m.S
    assert pytest.approx(m.module_diversity) == m.D
    assert pytest.approx(m.proportion) == m.P
