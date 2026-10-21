"""Test: the headless runners and the self-collision check."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import cast

import mujoco
import pytest

from ariel.simulation import runners
from ariel.simulation.mujoco_ops import has_self_collision
from ariel.simulators import compile_model

_gl_probe_worker = importlib.import_module("ariel.opengl._gl_probe_worker")

PLAIN_XML = "<mujoco><worldbody><geom size='.1'/></worldbody></mujoco>"
SHORT = 0.02
LOOP = 5


# --- runners ---------------------------------------------------------------- #


def test_simple_runner_advances_the_clock() -> None:
    """The simple runner steps until the requested duration is reached."""
    model = compile_model(mujoco.MjSpec.from_string(PLAIN_XML))
    data = mujoco.MjData(model)
    runners.simple_runner(model, data, duration=SHORT, steps_per_loop=LOOP)
    assert data.time >= SHORT


class _CountingController:
    def __init__(self) -> None:
        self.calls = 0

    def set_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        del model, data
        self.calls += 1


def test_thread_safe_runner_calls_the_controller_every_step() -> None:
    """The controller runs once per simulation step."""
    model = compile_model(mujoco.MjSpec.from_string(PLAIN_XML))
    data = mujoco.MjData(model)
    controller = _CountingController()
    runners.thread_safe_runner(model, data, controller, duration=SHORT)
    assert controller.calls == int(SHORT / model.opt.timestep)


# --- mujoco_ops ------------------------------------------------------------- #


def test_self_collision_check_reports_compile_errors() -> None:
    """A spec that MuJoCo cannot compile becomes a ValueError."""

    def fail() -> None:
        msg = "boom"
        raise mujoco.FatalError(msg)  # pyright: ignore[reportCallIssue]  # the generated stub loses FatalError in a circular import

    spec = cast("mujoco.MjSpec", SimpleNamespace(compile=fail))
    with pytest.raises(ValueError, match="Failed to compile"):
        has_self_collision(spec)
