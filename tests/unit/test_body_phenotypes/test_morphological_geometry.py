"""Test: the size measures describe the built robot, in metres."""

from __future__ import annotations

# Standard library
import itertools
import random
from functools import cache
from typing import TYPE_CHECKING

# Third-party libraries
import mujoco
import networkx as nx
import numpy as np
import pytest

# Local libraries
from ariel.body_phenotypes import (
    CoreModule,
    MorphologicalMeasures,
    construct_mjspec_from_graph,
)
from ariel.ec.genotypes.tree import random_tree
from tests.unit.test_body_phenotypes.test_robogen_lite.prebuilt_bodies import (
    PREBUILT,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

MM = 1e-3
CORE_SIDE = 0.2  # metres
SEED = 42
BODIES = 60
MODULES = 20
CORNERS = np.array(list(itertools.product((-1, 1), repeat=3)))


def _graph(
    types: dict[int, str], edges: list[tuple[int, int, str]]
) -> nx.DiGraph[int]:
    graph: nx.DiGraph[int] = nx.DiGraph()
    for node, kind in types.items():
        graph.add_node(node, type=kind, rotation="DEG_0")
    for parent, child, face in edges:
        graph.add_edge(parent, child, face=face)
    return graph


@cache
def _random_bodies() -> list[CoreModule]:
    random.seed(SEED)
    return [
        construct_mjspec_from_graph(random_tree(MODULES).to_networkx())
        for _ in range(BODIES)
    ]


def _extent(robot: CoreModule) -> NDArray[np.float64]:
    """Measure the robot from the corners of every geom, in the core frame.

    Returns
    -------
    NDArray[np.float64]
        Extent along x, y and z.
    """
    model = robot.spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    core = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "core")
    turn = data.xmat[core].reshape(3, 3)
    points = [
        data.geom_xpos[geom]
        + (
            data.geom_xmat[geom].reshape(3, 3)
            @ (CORNERS * model.geom_size[geom]).T
        ).T
        for geom in range(model.ngeom)
    ]
    local = (np.vstack(points) - data.xpos[core]) @ turn
    return np.asarray(local.max(axis=0) - local.min(axis=0), dtype=np.float64)


def _check_box(robot: CoreModule) -> None:
    measures = MorphologicalMeasures(robot)
    width, depth, height = _extent(robot)
    assert measures.bounding_box_depth == pytest.approx(depth, abs=MM)
    assert measures.bounding_box_width == pytest.approx(width, abs=MM)
    assert measures.bounding_box_height == pytest.approx(height, abs=MM)
    assert 0.0 < measures.coverage <= 1.0


def test_a_lone_core_is_one_core_wide() -> None:
    measures = MorphologicalMeasures(
        construct_mjspec_from_graph(_graph({0: "CORE"}, []))
    )
    assert measures.bounding_box_depth == pytest.approx(CORE_SIDE)
    assert measures.bounding_box_width == pytest.approx(CORE_SIDE)
    assert measures.coverage == pytest.approx(1.0)


def test_a_brick_in_front_adds_depth_only() -> None:
    measures = MorphologicalMeasures(
        construct_mjspec_from_graph(
            _graph({0: "CORE", 1: "BRICK"}, [(0, 1, "FRONT")])
        )
    )
    assert measures.bounding_box_depth > CORE_SIDE + MM
    assert measures.bounding_box_width == pytest.approx(CORE_SIDE)


def test_a_folded_limb_keeps_all_its_modules_and_its_length() -> None:
    """Core, hinge on BACK, brick on the hinge's FRONT: a line going back."""
    robot = construct_mjspec_from_graph(
        _graph(
            {0: "CORE", 1: "HINGE", 2: "BRICK"},
            [(0, 1, "BACK"), (1, 2, "FRONT")],
        )
    )
    measures = MorphologicalMeasures(robot)
    assert measures.num_modules == len({0, 1, 2})
    assert measures.bounding_box_depth == pytest.approx(_extent(robot)[1])
    assert measures.bounding_box_depth > CORE_SIDE + 2 * 0.05


@pytest.mark.parametrize("index", range(BODIES))
def test_a_random_body_measures_its_real_box(index: int) -> None:
    _check_box(_random_bodies()[index])


@pytest.mark.parametrize("name", sorted(PREBUILT))
def test_a_prebuilt_robot_measures_its_real_box(name: str) -> None:
    _check_box(PREBUILT[name]())


def test_the_repair_measures_the_pruned_body() -> None:
    random.seed(SEED)
    graphs = [random_tree(MODULES).to_networkx() for _ in range(BODIES)]
    pruned = 0
    for graph in graphs:
        robot = construct_mjspec_from_graph(graph, repair_overlaps=True)
        assert robot.graph is not None
        measures = MorphologicalMeasures(robot)
        assert measures.num_modules == robot.graph.number_of_nodes()
        pruned += measures.num_modules < graph.number_of_nodes()
    assert pruned > 0
