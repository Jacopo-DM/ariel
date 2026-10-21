"""Test: morphological measures on robots of different shapes."""

from __future__ import annotations

import inspect

import networkx as nx
import pytest

from ariel.body_phenotypes import (
    MorphologicalMeasures,
    construct_mjspec_from_graph,
)

HALF = 0.5
THREE_MODULES = 3
FOUR_MODULES = 4
CORE_FACES = ("FRONT", "BACK", "RIGHT", "LEFT", "TOP", "BOTTOM")
PROPERTIES = [
    name
    for name, _ in inspect.getmembers(
        MorphologicalMeasures, lambda obj: isinstance(obj, property)
    )
]


def _graph(
    types: dict[int, str], edges: list[tuple[int, int, str]]
) -> nx.DiGraph[int]:
    graph: nx.DiGraph[int] = nx.DiGraph()
    for node, kind in types.items():
        graph.add_node(node, type=kind, rotation="DEG_0")
    for parent, child, face in edges:
        graph.add_edge(parent, child, face=face)
    return graph


def _core_only() -> nx.DiGraph[int]:
    return _graph({0: "CORE"}, [])


def _measure(graph: nx.DiGraph[int]) -> MorphologicalMeasures:
    return MorphologicalMeasures(construct_mjspec_from_graph(graph))


def _measure_graph(graph: nx.DiGraph[int]) -> MorphologicalMeasures:
    # A graph no builder accepts, set on a built core to test the checks.
    robot = construct_mjspec_from_graph(_core_only())
    robot.graph = graph
    return MorphologicalMeasures(robot)


def _filled() -> nx.DiGraph[int]:
    # Core with a brick on every face; the first brick has all faces used.
    types = {0: "CORE"}
    edges: list[tuple[int, int, str]] = []
    for index, face in enumerate(CORE_FACES, start=1):
        types[index] = "BRICK"
        edges.append((0, index, face))
    next_id = len(CORE_FACES) + 1
    for face in ("FRONT", "RIGHT", "LEFT", "TOP", "BOTTOM"):
        types[next_id] = "HINGE"
        edges.append((1, next_id, face))
        types[next_id + 1] = "BRICK"
        edges.append((next_id, next_id + 1, "FRONT"))
        next_id += 2
    return _graph(types, edges)


@pytest.mark.parametrize(
    "graph",
    [_core_only(), _filled()],
    ids=["core-only", "filled"],
)
def test_every_property_can_be_read(graph: nx.DiGraph[int]) -> None:
    """All measures evaluate without error on very different robots."""
    measures = _measure(graph)
    for name in PROPERTIES:
        assert getattr(measures, name) is not None


def test_two_roots_are_rejected() -> None:
    """A graph with two roots has no single core."""
    graph = _graph({0: "CORE", 1: "BRICK"}, [])
    with pytest.raises(ValueError, match="exactly one root"):
        _measure_graph(graph)


def test_core_only_has_zero_ratios() -> None:
    """Measures with a zero denominator return 0."""
    measures = _measure(_core_only())
    assert measures.filled_core_and_bricks_proportion == pytest.approx(0.0)
    assert measures.limbs == pytest.approx(0.0)
    assert measures.joints == pytest.approx(0.0)
    assert measures.double_neighbour_brick_and_active_hinge_proportion == (
        pytest.approx(0.0)
    )
    assert measures.brick_cluster_ratio == pytest.approx(0.0)
    assert measures.module_diversity == pytest.approx(0.0)


def test_unknown_module_type_has_no_allowed_faces() -> None:
    """A module of an unknown type exposes no faces and counts as unfilled."""
    graph = _graph({0: "CORE", 1: "NONE"}, [(0, 1, "FRONT")])
    measures = _measure(graph)
    assert measures.num_bricks == 0
    assert measures.num_active_hinges == 0


def test_filled_robot_reports_filled_modules() -> None:
    """The core and the first brick have every allowed face used."""
    measures = _measure(_filled())
    assert measures.core_is_filled
    assert 1 in measures.filled_bricks
    assert measures.num_filled_bricks == len(measures.filled_bricks)
    assert measures.num_filled_active_hinges == len(
        measures.filled_active_hinges
    )
    assert measures.num_filled_modules >= 1
    assert measures.filled_core_and_bricks_proportion > 0.0
    assert measures.num_double_neighbour_bricks >= 0
    assert measures.num_double_neighbour_active_hinges >= 0


def test_brick_cluster_ratio_counts_brick_to_brick_edges() -> None:
    """Two bricks in a row form a cluster; brick to hinge does not."""
    cluster = _graph(
        {0: "CORE", 1: "BRICK", 2: "BRICK"},
        [(0, 1, "FRONT"), (1, 2, "FRONT")],
    )
    mixed = _graph(
        {0: "CORE", 1: "BRICK", 2: "HINGE"},
        [(0, 1, "FRONT"), (1, 2, "FRONT")],
    )
    # Two brick-involved edges (core-brick, brick-brick), one is brick-brick.
    assert _measure(cluster).brick_cluster_ratio == (pytest.approx(0.5))
    assert _measure_graph(mixed).brick_cluster_ratio == (pytest.approx(0.0))
    assert _measure_graph(mixed).module_diversity == pytest.approx(1.0)


def test_size_is_the_coverage() -> None:
    """Size and coverage are one measure: the share of the box filled."""
    measures = _measure(_graph({0: "CORE", 1: "BRICK"}, [(0, 1, "FRONT")]))
    assert measures.size == pytest.approx(measures.coverage)
    assert 0.0 < measures.coverage < 1.0


def test_a_module_on_top_of_the_core_is_not_flat() -> None:
    """A brick on TOP reaches above the core; one on FRONT does not."""
    on_top = _graph({0: "CORE", 1: "BRICK"}, [(0, 1, "TOP")])
    in_front = _graph({0: "CORE", 1: "BRICK"}, [(0, 1, "FRONT")])
    assert not _measure(on_top).is_2d
    assert _measure(in_front).is_2d


def test_an_edge_with_an_unknown_face_is_rejected() -> None:
    """The error names the edge and the bad face."""
    graph = _graph({0: "CORE", 1: "BRICK"}, [(0, 1, "SIDEWAYS")])
    with pytest.raises(ValueError, match=r"Edge 0 -> 1.*'SIDEWAYS'"):
        _measure_graph(graph)


def test_an_edge_without_a_face_is_rejected() -> None:
    """A faceless edge is an error even when the module is placed already."""
    graph = _graph(
        {0: "CORE", 1: "BRICK", 2: "BRICK"},
        [(0, 1, "FRONT"), (0, 2, "RIGHT")],
    )
    graph.add_edge(1, 2)
    with pytest.raises(ValueError, match=r"Edge 1 -> 2.*None"):
        _measure_graph(graph)


def test_a_module_with_two_parents_is_counted_once() -> None:
    """A child reached from two parents is one module."""
    graph = _graph(
        {0: "CORE", 1: "BRICK", 2: "BRICK", 3: "BRICK"},
        [
            (0, 1, "FRONT"),
            (0, 2, "RIGHT"),
            (1, 3, "RIGHT"),
            (2, 3, "FRONT"),
        ],
    )
    assert _measure_graph(graph).num_modules == FOUR_MODULES


def test_edges_between_non_bricks_do_not_count_for_clusters() -> None:
    """Core to hinge to hinge involves no brick."""
    graph = _graph(
        {0: "CORE", 1: "HINGE", 2: "HINGE"},
        [(0, 1, "FRONT"), (1, 2, "FRONT")],
    )
    assert _measure(graph).brick_cluster_ratio == (pytest.approx(0.0))


def test_a_module_with_two_valid_parents_is_counted_once() -> None:
    """A child with a second parent edge is still one module."""
    graph = _graph(
        {0: "CORE", 1: "BRICK", 2: "BRICK"},
        [(0, 1, "FRONT"), (0, 2, "RIGHT")],
    )
    graph.add_edge(1, 2, face="TOP")
    measures = _measure_graph(graph)
    assert measures.num_modules == THREE_MODULES


def _arms(right: list[str], left: list[str]) -> nx.DiGraph[int]:
    """Build a core with a chain of modules out of its RIGHT and LEFT faces.

    Returns
    -------
    nx.DiGraph[int]
        The robot graph; each chain continues through FRONT faces.
    """
    types = {0: "CORE"}
    edges: list[tuple[int, int, str]] = []
    next_id = 1
    for face, chain in (("RIGHT", right), ("LEFT", left)):
        previous, via = 0, face
        for kind in chain:
            types[next_id] = kind
            edges.append((previous, next_id, via))
            previous, via = next_id, "FRONT"
            next_id += 1
    return _graph(types, edges)


@pytest.mark.parametrize("arm_length", [1, 2, 3])
def test_matching_arms_left_and_right_score_one_on_that_plane(
    arm_length: int,
) -> None:
    """Everything sits in the core's plane, so the other two planes are 0."""
    arm = ["BRICK"] * arm_length
    measures = _measure(_arms(arm, arm))
    assert measures.yz_symmetry == pytest.approx(1.0)
    assert measures.xz_symmetry == pytest.approx(0.0)
    assert measures.xy_symmetry == pytest.approx(0.0)
    assert measures.symmetry == pytest.approx(1.0)


def test_a_brick_on_one_side_only_scores_zero() -> None:
    assert _measure(_arms(["BRICK"], [])).yz_symmetry == pytest.approx(0.0)


def test_a_hinge_does_not_mirror_a_brick() -> None:
    assert _measure(_arms(["BRICK"], ["HINGE"])).yz_symmetry == (
        pytest.approx(0.0)
    )


def test_a_mismatch_at_the_end_of_the_arms_halves_the_score() -> None:
    measures = _measure(_arms(["BRICK", "BRICK"], ["BRICK", "HINGE"]))
    assert measures.yz_symmetry == pytest.approx(HALF)


def test_a_longer_arm_on_one_side_lowers_the_score() -> None:
    """Three modules right, one left: two of the four have a partner."""
    measures = _measure(_arms(["BRICK"] * 3, ["BRICK"]))
    assert measures.yz_symmetry == pytest.approx(HALF)


def test_a_four_legged_spider_is_symmetric_both_ways() -> None:
    graph = _graph(
        {0: "CORE", 1: "HINGE", 2: "HINGE", 3: "HINGE", 4: "HINGE"},
        [(0, 1, "FRONT"), (0, 2, "BACK"), (0, 3, "LEFT"), (0, 4, "RIGHT")],
    )
    measures = _measure(graph)
    assert measures.yz_symmetry == pytest.approx(1.0)
    assert measures.xz_symmetry == pytest.approx(1.0)
    assert measures.xy_symmetry == pytest.approx(0.0)
