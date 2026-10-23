"""Test: a hinge at -45 degrees is the mirror image of one at +45."""

from __future__ import annotations

# Third-party libraries
import mujoco
import networkx as nx
import numpy as np
import numpy.typing as npt
import pytest

# Local libraries
from ariel.body_phenotypes.robogen_lite.config import ModuleRotationsTheta
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.simulators import compile_model

MIRROR = np.array([-1.0, 1.0, 1.0])  # through the plane x = 0
TOLERANCE = 1e-6  # metres
MISMATCH = 0.01  # metres; a wrong mirror is centimetres off
BENDS = [0.0, 0.5, -0.8]  # joint angles in radians
DEG_MINUS_45 = -45


def _leg_end(face: str, rotation: str, bend: float) -> npt.NDArray[np.float64]:
    """Return where the brick after a tilted hinge sits, core at the origin.

    Returns
    -------
    npt.NDArray[np.float64]
        Position of the last geom relative to the core.
    """
    graph: nx.DiGraph[int] = nx.DiGraph()
    graph.add_node(0, type="CORE", rotation="DEG_0")
    graph.add_node(1, type="HINGE", rotation=rotation)
    graph.add_node(2, type="BRICK", rotation="DEG_0")
    graph.add_edge(0, 1, face=face)
    graph.add_edge(1, 2, face="FRONT")
    model = compile_model(construct_mjspec_from_graph(graph).spec)
    data = mujoco.MjData(model)
    data.qpos[:] = bend
    mujoco.mj_forward(model, data)
    return np.asarray(data.geom_xpos[-1] - data.geom_xpos[0])


def test_minus_45_is_an_angle_of_the_builder() -> None:
    """The builder's angle table holds -45 degrees."""
    assert ModuleRotationsTheta.DEG_MINUS_45.value == DEG_MINUS_45


@pytest.mark.parametrize("bend", BENDS)
def test_minus_45_hinge_mirrors_plus_45(bend: float) -> None:
    """A -45 hinge on LEFT bent by -a mirrors a +45 hinge on RIGHT bent by a."""
    right = _leg_end("RIGHT", "DEG_45", bend)
    left = _leg_end("LEFT", "DEG_MINUS_45", -bend)
    assert np.allclose(left, right * MIRROR, atol=TOLERANCE)


def test_plus_45_hinge_does_not_mirror_when_bent() -> None:
    """Without -45, the left copy of a bent 45-degree hinge is off."""
    bend = BENDS[1]
    target = _leg_end("RIGHT", "DEG_45", bend) * MIRROR
    errors = [
        np.abs(_leg_end("LEFT", "DEG_45", sign * bend) - target).max()
        for sign in (1, -1)
    ]
    assert min(errors) > MISMATCH
