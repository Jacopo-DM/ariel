"""Test: a hinge moves like the DSS-M15 (goal servo-dynamics).

Every test runs the simulation and measures what the hinge does
(``qpos``, ``qvel``, ``actuator_force``), never reads a setting back: the
rig is a core fixed in place, one hinge on its front face and a brick as the
limb, in a flat world with ARIEL's own timestep and integrator, contacts
off.
"""

import ast
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from ariel.body_phenotypes.robogen_lite.config import ModuleFaces
from ariel.body_phenotypes.robogen_lite.modules import hinge as hinge_module
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.parts import (
    brick,
    build,
    hinge,
)
from ariel.parameters import ArielModulesConfig, FloatArray
from ariel.simulation.environments import SimpleFlatWorld

MODULES = ArielModulesConfig()
SERVO, SETTINGS = MODULES.servo, MODULES.servo_settings
STALL = float(SETTINGS.stall_torque(SERVO).value)  # N m
TOP_SPEED = float(SETTINGS.no_load_speed(SERVO).value)  # rad / s
DAMPING = float(SETTINGS.damping(SERVO).value)  # N m s / rad
HOLDING = float(SETTINGS.holding_error.value)  # rad
HALF_TRAVEL = float(SERVO.angle_range.value) / 2  # rad

LIMB = "robot1_brick_0brick"
SPEED_MARGIN = 1.005  # the integrator may overshoot a limit by a hair
ASSIST = 0.3  # N m pushing the hinge the way it already turns
LOAD = 3.0  # kg, pressing on the limb
GRAVITY = 9.80665  # m / s2
SWEEP_TIME = (0.54, 0.65)  # s, -90 to +90 degrees under its own power
STEP = math.radians(30)
SETTLE_TIME = 0.3  # s
OVERSHOOT = 0.05
SETTLED = math.radians(1)


def _rig(
    *,
    gravity: bool = False,
    horizontal_axis: bool = False,
    blocked: bool = False,
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Build the core-hinge-brick rig with the core fixed in the world.

    Parameters
    ----------
    gravity : bool
        Keep the world's gravity; off by default, so only the servo moves.
    horizontal_axis : bool
        Turn the hinge so its axis is horizontal and a positive turn lifts
        the limb; otherwise the axis is vertical.
    blocked : bool
        Weld the limb to the world, so the hinge cannot move.

    Returns
    -------
    tuple[mujoco.MjModel, mujoco.MjData]
        The compiled rig at rest.
    """
    rotation = 90 if horizontal_axis else 0
    robot = build((
        hinge(1, "core", ModuleFaces.FRONT, rotation),
        brick(2, "hinge_0", ModuleFaces.FRONT),
    ))
    world = SimpleFlatWorld()
    world.spawn(
        robot.spec, position=(0, 0, 0.5), correct_collision_with_floor=False
    )
    for joint in list(world.spec.joints):
        if joint.type == mujoco.mjtJoint.mjJNT_FREE:
            world.spec.delete(joint)
    # The servo is under test, not the shape: past about 83 degrees the
    # limb would hit the core's side
    world.spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    if not gravity:
        world.spec.option.gravity = np.zeros(3)
    if blocked:
        world.spec.add_equality(
            type=mujoco.mjtEq.mjEQ_WELD,
            name1=LIMB,
            objtype=mujoco.mjtObj.mjOBJ_BODY,
        )
    model = world.compile()
    return model, mujoco.MjData(model)


def _run(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target: float,
    seconds: float,
    *,
    start: float = 0.0,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Command ``target`` and record the hinge every step.

    Parameters
    ----------
    model : mujoco.MjModel
        The rig.
    data : mujoco.MjData
        Its state; ``qfrc_applied`` and ``xfrc_applied`` are kept.
    target : float
        The commanded angle, in radians.
    seconds : float
        How long to run.
    start : float
        The hinge's angle at the start, in radians.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Angle, speed and actuator torque after each step.
    """
    data.qpos[0] = start
    mujoco.mj_forward(model, data)
    data.ctrl[0] = target
    steps = round(seconds / model.opt.timestep)
    record = np.zeros((steps, 3), dtype=np.float64)
    for step in range(steps):
        mujoco.mj_step(model, data)
        record[step] = (data.qpos[0], data.qvel[0], data.actuator_force[0])
    return record[:, 0], record[:, 1], record[:, 2]


def test_a_blocked_hinge_never_exceeds_the_stall_torque() -> None:
    """Held still and told to turn 90 degrees, it pushes at most 1.32 N m."""
    model, data = _rig(blocked=True)
    _, _, torque = _run(model, data, HALF_TRAVEL, 2.0)
    assert np.abs(torque).max() <= STALL * (1 + 1e-9)
    assert np.abs(torque).max() >= STALL * 0.99  # it does reach the cap


@pytest.mark.parametrize("gravity", [False, True], ids=["level", "uphill"])
def test_its_own_power_never_beats_the_top_speed(*, gravity: bool) -> None:
    """A full sweep never goes faster than 5.8 rad/s, level or lifting."""
    model, data = _rig(gravity=gravity, horizontal_axis=gravity)
    _, speed, _ = _run(model, data, HALF_TRAVEL, 1.0, start=-HALF_TRAVEL)
    assert np.abs(speed).max() <= TOP_SPEED * SPEED_MARGIN
    assert np.abs(speed).max() >= TOP_SPEED * 0.9  # and it gets close


def test_a_full_sweep_takes_as_long_as_the_datasheet_says() -> None:
    """-90 to +90 degrees takes 0.54 to 0.65 s (0.18 s per 60 degrees)."""
    model, data = _rig()
    angle, _, _ = _run(model, data, HALF_TRAVEL, 1.0, start=-HALF_TRAVEL)
    arrived = np.flatnonzero(np.abs(angle - HALF_TRAVEL) < SETTLED)
    seconds = (arrived[0] + 1) * model.opt.timestep
    assert SWEEP_TIME[0] <= seconds <= SWEEP_TIME[1]


def test_an_assisting_torque_follows_the_motor_line() -> None:
    """0.3 N m pushing along gives 5.8 + 0.3 / 0.228 = 7.1 rad/s."""
    model, data = _rig()
    data.qfrc_applied[0] = ASSIST
    _, speed, _ = _run(model, data, HALF_TRAVEL, 0.5, start=-HALF_TRAVEL)
    expected = (STALL + ASSIST) / DAMPING
    assert np.abs(speed).max() == pytest.approx(expected, rel=0.05)


def test_a_heavy_load_cannot_push_it_past_its_range() -> None:
    """3 kg pressing on the limb moves it at most 1 degree past 90."""
    model, data = _rig(gravity=True, horizontal_axis=True)
    limb = model.body(LIMB).id
    # Pressing the limb up, the way the hinge already turns
    data.xfrc_applied[limb, 2] = LOAD * GRAVITY
    angle, _, _ = _run(model, data, HALF_TRAVEL, 2.0)
    assert angle.max() <= HALF_TRAVEL + SETTLED


@pytest.mark.parametrize("fraction", [0.5, 1.0])
def test_full_torque_at_the_holding_error(fraction: float) -> None:
    """Held still, it pushes in proportion to the error, all of it at 5."""
    model, data = _rig(blocked=True)
    _, _, torque = _run(model, data, HOLDING * fraction, 0.5)
    assert torque[-1] == pytest.approx(STALL * fraction, rel=0.01)


def test_a_30_degree_step_settles_without_overshoot() -> None:
    """Within 0.3 s it sits within 1 degree, never more than 5 % past."""
    model, data = _rig()
    angle, _, _ = _run(model, data, STEP, 1.0)
    after = round(SETTLE_TIME / model.opt.timestep)
    assert np.abs(angle[after:] - STEP).max() < SETTLED
    assert angle.max() <= STEP * (1 + OVERSHOOT)


def test_the_gearbox_inertia_slows_a_quick_start() -> None:
    """Reaching half the top speed takes longer with the armature."""
    times = []
    for armature in (True, False):
        model, data = _rig()
        if not armature:
            model.dof_armature[0] = 0
        _, speed, _ = _run(model, data, HALF_TRAVEL, 0.2, start=-HALF_TRAVEL)
        reached = np.flatnonzero(np.abs(speed) >= TOP_SPEED / 2)
        times.append(reached[0])
    with_armature, without = times
    assert with_armature > 2 * without


def test_the_servo_block_writes_no_number_of_its_own() -> None:
    """Every servo value in hinge.py comes from the servo records."""
    source = Path(hinge_module.__file__).read_text("utf-8")
    start = source.index("# ========= Servo =========")
    end = source.index("# Save model specifications")
    block = ast.parse(
        "\n".join(line.strip() for line in source[start:end].splitlines())
    )
    numbers = {
        node.value
        for node in ast.walk(block)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, int | float)
        and not isinstance(node.value, bool)
    }
    assert numbers <= {0, 1, 2}
