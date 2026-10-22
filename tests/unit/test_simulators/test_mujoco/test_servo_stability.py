"""Test: every prebuilt robot stays stable with DSS-M15 hinges (SD3).

Each robot stands in a flat world for 10 s while its hinges chase random
targets, a new set every half second: the state must stay finite. The
fastest hinge speed is only reported (goal servo-dynamics, acceptance 7).
"""

import mujoco
import numpy as np
import pytest

from ariel.simulation.environments import SimpleFlatWorld
from tests.unit.test_body_phenotypes.test_robogen_lite.prebuilt_bodies import (
    PREBUILT,
)

SECONDS = 10.0
HOLD = 0.5  # s between new targets
SEED = 42


def _fastest_hinge(name: str) -> float:
    """Run one robot with random targets and return its fastest hinge speed.

    Parameters
    ----------
    name : str
        A prebuilt robot.

    Returns
    -------
    float
        The largest hinge speed seen, in rad / s.
    """
    world = SimpleFlatWorld()
    world.spawn(PREBUILT[name]().spec)
    model = world.compile()
    data = mujoco.MjData(model)
    rng = np.random.default_rng(SEED)
    dofs = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    low, high = model.actuator_ctrlrange.T
    hold = round(HOLD / model.opt.timestep)
    fastest = 0.0
    for step in range(round(SECONDS / model.opt.timestep)):
        if step % hold == 0:
            data.ctrl[:] = rng.uniform(low, high)
        mujoco.mj_step(model, data)
        fastest = max(fastest, float(np.abs(data.qvel[dofs]).max()))
    assert np.isfinite(data.qpos).all()
    assert np.isfinite(data.qvel).all()
    return fastest


@pytest.mark.parametrize("name", sorted(PREBUILT))
def test_random_targets_keep_a_robot_finite(
    name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Ten seconds of random targets leave every robot's state finite."""
    fastest = _fastest_hinge(name)
    with capsys.disabled():
        print(f"\n{name}: fastest hinge {fastest:.2f} rad/s")  # ruff: ignore[print]  # the reported number of acceptance 7
    assert fastest > 0
