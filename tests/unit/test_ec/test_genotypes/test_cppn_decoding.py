"""Test: the CPPN decoding script (run a decoded robot with an NA-CPG)."""

from __future__ import annotations

import inspect
import runpy
from typing import TYPE_CHECKING

import mujoco
import mujoco.viewer
import pytest

from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.ec.genotypes.cppn import cppn_decoding
from ariel.utils import renderers

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule


class _Captured:
    """What the script hands to MuJoCo and to the viewer."""

    control: Callable[[mujoco.MjModel, mujoco.MjData], None] | None = None
    launched = 0


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Captured:
    # Run in a temporary folder with OpenGL and the viewer replaced.
    monkeypatch.chdir(tmp_path)
    seen = _Captured()

    def remember_control(
        function: Callable[[mujoco.MjModel, mujoco.MjData], None] | None,
    ) -> None:
        seen.control = function

    def launch(*, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        seen.launched += 1
        assert seen.control is not None
        seen.control(model, data)

    monkeypatch.setattr(
        renderers, "single_frame_renderer", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(mujoco, "set_mjcb_control", remember_control)
    monkeypatch.setattr(mujoco.viewer, "launch", launch)
    return seen


def _load_script(run_name: str) -> dict[str, object]:
    return runpy.run_path(inspect.getfile(cppn_decoding), run_name=run_name)


def test_run_wires_a_cpg_controller_to_the_robot(
    captured: _Captured,
) -> None:
    """run() spawns the robot and installs a controller callback."""
    namespace = _load_script("cppn_decoding_import")
    run = namespace["run"]
    assert callable(run)
    robot: CoreModule = gecko()
    run(robot, with_viewer=True)
    assert captured.launched == 1
    assert captured.control is not None
    del robot


def test_run_without_a_viewer_does_not_open_one(
    captured: _Captured,
) -> None:
    """The viewer is optional."""
    run = _load_script("cppn_decoding_import")["run"]
    assert callable(run)
    robot: CoreModule = gecko()
    run(robot)
    assert captured.launched == 0
    del robot


def test_script_decodes_and_runs_a_random_robot(captured: _Captured) -> None:
    """As a program it grows a random CPPN robot and runs it in a viewer."""
    _load_script("__main__")
    assert captured.launched == 1
