"""Test: every module compiles to the measured sizes and masses."""

import mujoco
import networkx as nx
import pytest

from ariel.body_phenotypes.robogen_lite.config import ModuleFaces
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule
from ariel.body_phenotypes.robogen_lite.modules.module import half_extents
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.parts import (
    build,
    hinge,
)
from ariel.parameters import ArielModulesConfig, kilograms, metres
from ariel.simulators import compile_model

SPECS = ArielModulesConfig()
SHRINK = 0.99  # the fork's box is drawn 1% smaller against z-fighting
DEEP = 8  # a hinge 8 steps from the core


def _compiled(spec: mujoco.MjSpec) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = compile_model(spec)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _mass(model: mujoco.MjModel, body: str) -> float:
    return float(model.body(body).mass[0])


def test_a_brick_is_the_measured_brick() -> None:
    model, _ = _compiled(BrickModule(index=1).spec)
    assert _mass(model, "brick") == pytest.approx(kilograms(SPECS.brick.mass))
    assert model.geom("brick").size.tolist() == pytest.approx(
        half_extents(SPECS.brick.size)
    )
    assert model.site("brick-front").pos[1] == pytest.approx(0.075)


def test_the_core_is_the_measured_core() -> None:
    model, _ = _compiled(CoreModule(index=0).spec)
    assert _mass(model, "core") == pytest.approx(kilograms(SPECS.core.mass))
    assert model.geom("core").size.tolist() == pytest.approx(
        half_extents(SPECS.core.size)
    )


def test_a_hinge_is_a_fork_and_a_cradle() -> None:
    """The fork holds the axis; the cradle with the servo turns about it."""
    model, data = _compiled(HingeModule(index=1).spec)
    width = metres(SPECS.hinge.width)
    axis = metres(SPECS.hinge.axis_from_parent)
    length = metres(SPECS.hinge.length)
    assert _mass(model, "hinge") == 0  # no mass counted twice
    assert _mass(model, "stator") == pytest.approx(
        kilograms(SPECS.hinge.parent_mass)
    )
    assert _mass(model, "rotor") == pytest.approx(
        kilograms(SPECS.hinge.child_mass(1))
    )
    assert model.geom("rotor").size.tolist() == pytest.approx([
        width / 2,
        (length - axis) / 2,
        width / 2,
    ])
    assert model.geom("stator").size.tolist() == pytest.approx([
        width / 2 * SHRINK,
        axis / 2 * SHRINK,
        width / 2 * SHRINK,
    ])
    assert data.xanchor[model.joint("servo").id][1] == pytest.approx(axis)
    assert data.site("hinge-front").xpos[1] == pytest.approx(length)


def test_the_servo_cable_follows_the_route_in_a_graph() -> None:
    """A hinge 8 steps out carries 7 more steps of cable than the first."""
    graph: nx.DiGraph[int] = nx.DiGraph()
    graph.add_node(0, type="CORE", rotation="DEG_0")
    for node in range(1, DEEP + 1):
        graph.add_node(node, type="HINGE", rotation="DEG_0")
        graph.add_edge(node - 1, node, face="FRONT")
    model, _ = _compiled(construct_mjspec_from_graph(graph).spec)
    first, last = (
        float(model.body(f"{node - 1}-{node}-0-rotor").mass[0])
        for node in (1, DEEP)
    )
    assert first == pytest.approx(kilograms(SPECS.hinge.child_mass(1)))
    assert last == pytest.approx(kilograms(SPECS.hinge.child_mass(DEEP)))


def test_the_servo_cable_follows_the_route_in_a_table() -> None:
    """parts.build counts the steps from the core the same way."""
    parts = tuple(
        hinge(
            node,
            "core" if node == 1 else f"hinge_{node - 2}",
            ModuleFaces.FRONT,
        )
        for node in range(1, DEEP + 1)
    )
    model, _ = _compiled(build(parts).spec)
    assert float(model.body(f"hinge_{DEEP - 1}rotor").mass[0]) == pytest.approx(
        kilograms(SPECS.hinge.child_mass(DEEP))
    )
