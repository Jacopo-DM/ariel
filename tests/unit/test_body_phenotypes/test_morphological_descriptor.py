"""Test: morphological measures on graphs of different shapes."""

from __future__ import annotations

import inspect

import networkx as nx
import pytest

from ariel.body_phenotypes.morphological_descriptor import MorphologicalMeasures

CROSS_ARM = 2
THREE_MODULES = 3
FOUR_MODULES = 4
CROSS_MODULES = 1 + 6 * CROSS_ARM
SPAN = 2 * CROSS_ARM + 1
FILLED_BRICK_FACES = 5
CORE_FACES = ("FRONT", "BACK", "RIGHT", "LEFT", "TOP", "BOTTOM")
OPPOSITE = {
    "FRONT": "BACK",
    "BACK": "FRONT",
    "RIGHT": "LEFT",
    "LEFT": "RIGHT",
    "TOP": "BOTTOM",
    "BOTTOM": "TOP",
}
PROPERTIES = [
    name
    for name, member in inspect.getmembers(
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


def _cross(arm_types: dict[str, tuple[str, str]]) -> nx.DiGraph[int]:
    # A core with a two-module arm on each of the six faces.
    types = {0: "CORE"}
    edges: list[tuple[int, int, str]] = []
    next_id = 1
    for face in CORE_FACES:
        first, second = arm_types.get(face, ("BRICK", "BRICK"))
        types[next_id] = first
        types[next_id + 1] = second
        edges.extend([(0, next_id, face), (next_id, next_id + 1, face)])
        next_id += 2
    return _graph(types, edges)


def _core_only() -> nx.DiGraph[int]:
    return _graph({0: "CORE"}, [])


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
    [_core_only(), _cross({}), _filled()],
    ids=["core-only", "cross", "filled"],
)
def test_every_property_can_be_read(graph: nx.DiGraph[int]) -> None:
    """All measures evaluate without error on very different robots."""
    measures = MorphologicalMeasures(graph)
    for name in PROPERTIES:
        assert getattr(measures, name) is not None


def test_two_roots_are_rejected() -> None:
    """A graph with two roots has no single core."""
    graph = _graph({0: "CORE", 1: "BRICK"}, [])
    with pytest.raises(ValueError, match="exactly one root"):
        MorphologicalMeasures(graph)


def test_core_only_has_zero_ratios() -> None:
    """Measures with a zero denominator return 0."""
    measures = MorphologicalMeasures(_core_only())
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
    measures = MorphologicalMeasures(graph)
    assert measures.num_bricks == 0
    assert measures.num_active_hinges == 0


def test_cross_is_equally_symmetric_in_every_plane() -> None:
    """Identical arms on opposite faces score the same in all three planes."""
    measures = MorphologicalMeasures(_cross({}))
    assert measures.num_modules == CROSS_MODULES
    assert measures.bounding_box_depth == SPAN
    assert measures.xy_symmetry > 0.0
    assert measures.xy_symmetry == pytest.approx(measures.xz_symmetry)
    assert measures.xz_symmetry == pytest.approx(measures.yz_symmetry)
    assert measures.symmetry == pytest.approx(measures.xy_symmetry)
    assert measures.proportion_2d == pytest.approx(1.0)


def test_cross_with_different_arms_is_less_symmetric() -> None:
    """Different module types on opposite faces lower the symmetry."""
    mixed = _cross({
        "TOP": ("HINGE", "HINGE"),
        "LEFT": ("HINGE", "HINGE"),
        "BACK": ("HINGE", "HINGE"),
    })
    full = MorphologicalMeasures(_cross({}))
    measures = MorphologicalMeasures(mixed)
    assert measures.xy_symmetry < full.xy_symmetry
    assert measures.xz_symmetry < full.xz_symmetry
    assert measures.yz_symmetry < full.yz_symmetry


def test_filled_robot_reports_filled_modules() -> None:
    """The core and the first brick have every allowed face used."""
    measures = MorphologicalMeasures(_filled())
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
    assert MorphologicalMeasures(cluster).brick_cluster_ratio == (
        pytest.approx(0.5)
    )
    assert MorphologicalMeasures(mixed).brick_cluster_ratio == (
        pytest.approx(0.0)
    )
    assert MorphologicalMeasures(mixed).module_diversity == pytest.approx(1.0)


def test_size_is_modules_over_volume() -> None:
    """Size is the number of modules over the bounding-box volume."""
    measures = MorphologicalMeasures(_cross({}))
    assert measures.size == pytest.approx(
        measures.num_modules / measures.bounding_box_volume
    )


def test_rotated_modules_are_not_planar() -> None:
    """A 45 degree rotation leaves the 2D family."""
    graph = _graph({0: "CORE", 1: "HINGE"}, [(0, 1, "FRONT")])
    graph.nodes[1]["rotation"] = "DEG_45"
    assert not MorphologicalMeasures(graph).is_2d


def test_an_edge_with_an_unknown_face_is_rejected() -> None:
    """The error names the edge and the bad face."""
    graph = _graph({0: "CORE", 1: "BRICK"}, [(0, 1, "SIDEWAYS")])
    with pytest.raises(ValueError, match=r"Edge 0 -> 1.*'SIDEWAYS'"):
        MorphologicalMeasures(graph)


def test_an_edge_without_a_face_is_rejected() -> None:
    """A faceless edge is an error even when the module is placed already."""
    graph = _graph(
        {0: "CORE", 1: "BRICK", 2: "BRICK"},
        [(0, 1, "FRONT"), (0, 2, "RIGHT")],
    )
    graph.add_edge(1, 2)
    with pytest.raises(ValueError, match=r"Edge 1 -> 2.*None"):
        MorphologicalMeasures(graph)


def test_a_module_with_two_parents_is_placed_once() -> None:
    """A child reached from two parents keeps its first position."""
    graph = _graph(
        {0: "CORE", 1: "BRICK", 2: "BRICK", 3: "BRICK"},
        [
            (0, 1, "FRONT"),
            (0, 2, "RIGHT"),
            (1, 3, "RIGHT"),
            (2, 3, "FRONT"),
        ],
    )
    assert MorphologicalMeasures(graph).num_modules == FOUR_MODULES


def test_edges_between_non_bricks_do_not_count_for_clusters() -> None:
    """Core to hinge to hinge involves no brick."""
    graph = _graph(
        {0: "CORE", 1: "HINGE", 2: "HINGE"},
        [(0, 1, "FRONT"), (1, 2, "FRONT")],
    )
    assert MorphologicalMeasures(graph).brick_cluster_ratio == (
        pytest.approx(0.0)
    )


def test_a_module_with_two_valid_parents_is_placed_once() -> None:
    """The first edge that reaches a module decides where it sits."""
    graph = _graph(
        {0: "CORE", 1: "BRICK", 2: "BRICK"},
        [(0, 1, "FRONT"), (0, 2, "RIGHT")],
    )
    graph.add_edge(1, 2, face="TOP")
    measures = MorphologicalMeasures(graph)
    assert measures.num_modules == THREE_MODULES


def _arms(
    right: list[str],
    left: list[str],
    faces: tuple[str, str] = ("RIGHT", "LEFT"),
) -> nx.DiGraph[int]:
    """Build a core with a chain of modules on two opposite faces.

    Returns
    -------
    nx.DiGraph[int]
        The robot graph.
    """
    types = {0: "CORE"}
    edges: list[tuple[int, int, str]] = []
    next_id = 1
    for face, chain in zip(faces, (right, left), strict=True):
        previous = 0
        for kind in chain:
            types[next_id] = kind
            edges.append((previous, next_id, face))
            previous = next_id
            next_id += 1
    return _graph(types, edges)


@pytest.mark.parametrize("arm_length", [1, 2, 3])
def test_a_mirror_symmetric_cross_scores_one_in_every_plane(
    arm_length: int,
) -> None:
    """The outermost layer counts, so wide robots reach 1.0 too."""
    types = {0: "CORE"}
    edges: list[tuple[int, int, str]] = []
    next_id = 1
    for face in CORE_FACES:
        previous = 0
        for _ in range(arm_length):
            types[next_id] = "BRICK"
            edges.append((previous, next_id, face))
            previous = next_id
            next_id += 1
    measures = MorphologicalMeasures(_graph(types, edges))
    assert measures.xy_symmetry == pytest.approx(1.0)
    assert measures.xz_symmetry == pytest.approx(1.0)
    assert measures.yz_symmetry == pytest.approx(1.0)


@pytest.mark.parametrize("arm_length", [1, 2, 3])
def test_a_symmetric_line_is_symmetric_across_its_length_only(
    arm_length: int,
) -> None:
    """A flat line has everything on the other two planes, which count 0.0."""
    arm = ["BRICK"] * arm_length
    measures = MorphologicalMeasures(_arms(arm, arm))
    assert measures.xz_symmetry == pytest.approx(1.0)
    assert measures.xy_symmetry == pytest.approx(0.0)
    assert measures.yz_symmetry == pytest.approx(0.0)


def test_a_mismatch_in_the_outer_layer_lowers_the_score() -> None:
    """The outer pair (brick against hinge) is compared and does not match."""
    measures = MorphologicalMeasures(
        _arms(["BRICK", "BRICK"], ["BRICK", "HINGE"])
    )
    assert measures.xz_symmetry == pytest.approx(0.5)


def test_an_arm_without_a_mirror_scores_zero() -> None:
    """Modules on one side only have nothing to match."""
    measures = MorphologicalMeasures(_arms(["BRICK"] * 3, []))
    assert measures.xz_symmetry == pytest.approx(0.0)


def test_a_core_off_the_middle_compares_the_layers_both_sides_reach() -> None:
    """Three modules right, one left: only the first layer has a pair."""
    measures = MorphologicalMeasures(_arms(["BRICK"] * 3, ["BRICK"]))
    assert measures.xz_symmetry == pytest.approx(2 / 4)
