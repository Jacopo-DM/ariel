"""Test: the boxes of a built body at its rest pose."""

from __future__ import annotations

# Standard library
import math
import random

# Third-party libraries
import numpy as np
import pytest

# Local libraries
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots import john_set
from ariel.body_phenotypes.robogen_lite.rest_pose import (
    owner_by_prefix,
    rest_boxes,
)
from ariel.ec.genotypes.tree import random_tree
from tests.unit.test_body_phenotypes.test_robogen_lite.prebuilt_bodies import (
    PREBUILT,
)

TENTH = 10
PREFIXES = [("m10", TENTH), ("m1", 1)]


def test_the_longest_prefix_wins() -> None:
    assert owner_by_prefix("m10stator", PREFIXES) == TENTH
    assert owner_by_prefix("m1stator", PREFIXES) == 1


def test_a_name_no_prefix_matches_belongs_to_the_core() -> None:
    assert owner_by_prefix("core", PREFIXES) == 0


def test_a_body_without_a_graph_is_rejected() -> None:
    with pytest.raises(ValueError, match="no graph"):
        rest_boxes(CoreModule(index=0))


@pytest.mark.parametrize("name", sorted(PREBUILT))
def test_every_module_of_a_prebuilt_robot_owns_geoms(name: str) -> None:
    robot = PREBUILT[name]()
    boxes = rest_boxes(robot)
    assert robot.graph is not None
    assert set(boxes.owners) == set(robot.graph.nodes)
    assert boxes.halves.shape == (len(boxes.owners), 3)
    assert boxes.poses.shape == (len(boxes.owners), 4, 4)


def test_every_module_of_a_random_body_owns_geoms() -> None:
    random.seed(42)
    for _ in range(20):
        robot = construct_mjspec_from_graph(random_tree(20).to_networkx())
        assert robot.graph is not None
        assert set(rest_boxes(robot).owners) == set(robot.graph.nodes)


def test_poses_are_in_the_core_frame() -> None:
    plain = rest_boxes(john_set.spider_8())
    turned_core = john_set.spider_8()
    turned_core.spec.body("core").quat = np.array([
        math.cos(math.pi / 8),
        0,
        0,
        math.sin(math.pi / 8),
    ])
    turned = rest_boxes(turned_core)
    assert plain.owners == turned.owners
    np.testing.assert_allclose(turned.poses, plain.poses, atol=1e-9)
