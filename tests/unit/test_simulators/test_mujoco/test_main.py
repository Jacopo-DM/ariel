"""Tests for the command line report of the MuJoCo settings generator."""

# Standard library
import runpy
import sys
from pathlib import Path
from typing import cast

# Third-party libraries
import mujoco
import pytest

# Local libraries
from ariel.simulators.mujoco.__main__ import main


def test_report_prints_the_table_the_differences_and_the_band_count(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "timestep" in out
    assert "option.timestep" in out
    assert " differ" in out
    assert "values outside their plausible band" in out


def test_report_shows_the_default_and_the_new_value(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main([])
    assert "option.integrator: 0 -> 3" in capsys.readouterr().out


def test_out_writes_a_loadable_mjcf_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "world.xml"
    assert main(["--out", str(target)]) == 0
    assert str(target) in capsys.readouterr().out
    model = cast(
        "mujoco.MjModel", mujoco.MjSpec.from_file(str(target)).compile()
    )
    assert model.opt.timestep == pytest.approx(0.0025)


def test_running_the_package_as_a_script(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["ariel.simulators.mujoco"])
    monkeypatch.delitem(sys.modules, "ariel.simulators.mujoco.__main__")
    with pytest.raises(SystemExit) as stop:
        runpy.run_module("ariel.simulators.mujoco", run_name="__main__")
    assert stop.value.code == 0
    assert " differ" in capsys.readouterr().out
