"""Test: finding modules of a robot body that sit inside each other."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import networkx as nx
import numpy as np
import pytest

from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

from ariel.body_phenotypes.robogen_lite.overlap import (
    TOLERANCE,
    Overlap,
    box_overlaps,
    find_overlaps,
    module_sizes,
    owner_of,
    pair_tolerance,
)

CUBE = 0.05  # half size of a brick
THIN = 0.01  # half size of a thin plate
UNDER = 0.99 * TOLERANCE  # share of the thinnest side
OVER = 1.01 * TOLERANCE
CLEAN: tuple[dict[int, tuple[str, str]], list[tuple[int, int, str]]] = (
    {
        0: ("CORE", "DEG_0"),
        1: ("BRICK", "DEG_0"),
        2: ("HINGE", "DEG_90"),
        3: ("BRICK", "DEG_0"),
    },
    [(0, 1, "FRONT"), (1, 2, "FRONT"), (2, 3, "FRONT")],
)
OVERLAPPING: tuple[dict[int, tuple[str, str]], list[tuple[int, int, str]]] = (
    {
        0: ("CORE", "DEG_0"),
        1: ("BRICK", "DEG_45"),
        2: ("BRICK", "DEG_45"),
        3: ("HINGE", "DEG_45"),
    },
    [(0, 1, "FRONT"), (1, 2, "TOP"), (2, 3, "LEFT")],
)


def _body_graph(
    spec: tuple[dict[int, tuple[str, str]], list[tuple[int, int, str]]],
) -> nx.DiGraph[int]:
    graph: nx.DiGraph[int] = nx.DiGraph()
    for node, (kind, rotation) in spec[0].items():
        graph.add_node(node, type=kind, rotation=rotation)
    for parent, child, face in spec[1]:
        graph.add_edge(parent, child, face=face)
    return graph


def _siblings() -> nx.DiGraph[int]:
    # nodes 1 and 2 hang from the core and are not neighbours
    graph: nx.DiGraph[int] = nx.DiGraph()
    graph.add_edges_from([(0, 1), (0, 2)])
    return graph


def _pose(x: float) -> NDArray[np.float64]:
    pose = np.eye(4)
    pose[0, 3] = x
    return pose


def _two_boxes(
    first: float, second: float, depth: float, *, scale: float = 1.0
) -> tuple[list[int], NDArray[np.float64], NDArray[np.float64]]:
    # Box 1 at the origin, box 2 pushed into it by `depth` along x.
    halves = scale * np.array([[first] * 3, [second] * 3])
    gap = scale * (first + second) - depth
    return [1, 2], halves, np.stack([_pose(0.0), _pose(gap)])


def test_the_tolerance_follows_the_smaller_module() -> None:
    big, small = (8e-3, 0.2), (1e-6, 0.02)
    assert pair_tolerance(big, small) == pytest.approx(TOLERANCE * 0.02)
    assert pair_tolerance(small, big) == pytest.approx(TOLERANCE * 0.02)
    equal = pair_tolerance((1.0, 0.5), (1.0, 0.7))
    assert equal == pytest.approx(TOLERANCE * 0.5)  # the first one on a tie


def test_module_sizes_add_the_geoms_of_a_module() -> None:
    owners = [1, 2, 2]
    halves = np.array([[0.05] * 3, [0.025, 0.03, 0.025], [0.025, 0.02, 0.025]])
    sizes = module_sizes(owners, halves)
    assert sizes[1] == pytest.approx((0.1**3, 0.1))
    assert sizes[2][0] == pytest.approx(0.05 * 0.06 * 0.05 + 0.05 * 0.04 * 0.05)
    assert sizes[2][1] == pytest.approx(0.04)  # the thinnest geom side


@pytest.mark.parametrize("scale", [1.0, 2.0, 0.1])
def test_a_depth_just_under_one_percent_is_not_an_overlap(
    scale: float,
) -> None:
    owners, halves, poses = _two_boxes(
        CUBE, CUBE, UNDER * 2 * CUBE * scale, scale=scale
    )
    assert box_overlaps(_siblings(), owners, halves, poses) == []


@pytest.mark.parametrize("scale", [1.0, 2.0, 0.1])
def test_a_depth_just_over_one_percent_is_an_overlap(scale: float) -> None:
    owners, halves, poses = _two_boxes(
        CUBE, CUBE, OVER * 2 * CUBE * scale, scale=scale
    )
    found = box_overlaps(_siblings(), owners, halves, poses)
    assert [(o.first, o.second) for o in found] == [(1, 2)]
    assert found[0].depth == pytest.approx(OVER * 2 * CUBE * scale, rel=1e-3)
    assert found[0].tolerance == pytest.approx(TOLERANCE * 2 * CUBE * scale)


def test_touching_boxes_are_not_an_overlap() -> None:
    owners, halves, poses = _two_boxes(CUBE, CUBE, 0.0)
    assert box_overlaps(_siblings(), owners, halves, poses) == []


def test_the_thinner_module_sets_the_tolerance() -> None:
    # A thin plate pushed 0.5 mm into a big cube: 0.5 mm is 2.5% of the
    # plate's thinnest side (0.02 m) but only 0.5% of the cube's (0.1 m).
    owners, halves, poses = _two_boxes(0.05, THIN, 5e-4)
    found = box_overlaps(_siblings(), owners, halves, poses)
    assert [(o.first, o.second) for o in found] == [(1, 2)]
    assert found[0].tolerance == pytest.approx(TOLERANCE * 2 * THIN)


@pytest.mark.parametrize("edge", [(1, 2), (2, 1)])
def test_parent_and_child_are_never_compared(edge: tuple[int, int]) -> None:
    graph: nx.DiGraph[int] = nx.DiGraph()
    graph.add_edges_from([(0, edge[0]), edge])
    owners, halves, poses = _two_boxes(CUBE, CUBE, CUBE)
    assert box_overlaps(graph, owners, halves, poses) == []


def test_two_geoms_of_one_module_never_overlap_each_other() -> None:
    halves = np.array([[CUBE] * 3, [CUBE] * 3])
    poses = np.stack([_pose(0.0), _pose(0.02)])
    assert box_overlaps(_siblings(), [1, 1], halves, poses) == []


def test_a_pair_reports_its_deepest_contact() -> None:
    # Module 2 has two boxes: one pushed 3 mm and one 8 mm into module 1.
    halves = np.array([[CUBE] * 3] * 3)
    poses = np.stack([
        _pose(0.0),
        _pose(2 * CUBE - 0.003),
        _pose(2 * CUBE - 0.008),
    ])
    found = box_overlaps(_siblings(), [1, 2, 2], halves, poses)
    assert len(found) == 1
    assert found[0].depth == pytest.approx(0.008, rel=1e-3)


def test_overlaps_come_out_in_node_order() -> None:
    graph: nx.DiGraph[int] = nx.DiGraph()
    graph.add_edges_from([(0, 1), (0, 2), (0, 3)])
    halves = np.array([[CUBE] * 3] * 3)
    poses = np.stack([_pose(0.0), _pose(0.02), _pose(0.04)])
    found = box_overlaps(graph, [3, 1, 2], halves, poses)
    assert [(o.first, o.second) for o in found] == [(1, 2), (1, 3), (2, 3)]


def test_overlap_is_a_frozen_record() -> None:
    overlap = Overlap(first=1, second=2, depth=0.1, tolerance=0.01)
    with pytest.raises(ValueError, match="frozen"):
        overlap.depth = 0.2  # type: ignore[misc]  # frozen on purpose


def test_a_clean_body_has_no_overlap() -> None:
    graph = _body_graph(CLEAN)
    assert find_overlaps(graph, construct_mjspec_from_graph(graph)) == []


def test_a_body_with_modules_inside_each_other_is_reported() -> None:
    graph = _body_graph(OVERLAPPING)
    found = find_overlaps(graph, construct_mjspec_from_graph(graph))
    assert [(o.first, o.second) for o in found] == [(0, 2), (0, 3)]
    assert all(o.depth > o.tolerance for o in found)
    # the brick on the core's FRONT face is a neighbour, so it is not listed
    assert all(1 not in {o.first, o.second} for o in found)


def test_owner_of_reads_the_node_from_the_geom_name() -> None:
    graph = _body_graph(CLEAN)
    model = construct_mjspec_from_graph(graph).spec.compile()
    owners = [owner_of(model, geom) for geom in range(model.ngeom)]
    assert owners == [0, 1, 2, 2, 3]  # core, brick, stator, rotor, brick


def test_a_geom_without_a_name_belongs_to_the_core() -> None:
    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[1, 1, 1])
    assert owner_of(spec.compile(), 0) == 0


def test_a_geom_that_is_not_a_box_is_rejected() -> None:
    graph = _body_graph(CLEAN)
    core = construct_mjspec_from_graph(graph)
    core.spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.1, 0, 0]
    )
    with pytest.raises(ValueError, match="box geoms"):
        find_overlaps(graph, core)
