"""Test: small helpers and the scripts behind ``python -m`` entry points."""

from __future__ import annotations

import importlib.metadata
import inspect
import io
import os
import runpy
import subprocess  # ruff: ignore[suspicious-subprocess-import]  # only builds fake results
import sys
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Self, cast
from unittest.mock import Mock

import mujoco
import pytest
from rich.console import Console

from ariel.ec.genotypes.nde import nde
from ariel.utils import (
    file_ops,
    gl_backend,
    gl_manager,
    runners,
)
from ariel.utils.gl_backend import GLConfig
from ariel.utils.mujoco_ops import has_self_collision
from ariel.utils.optimizers import revde

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_gl_probe_worker = importlib.import_module("ariel.utils._gl_probe_worker")

PLAIN_XML = "<mujoco><worldbody><geom size='.1'/></worldbody></mujoco>"
SHORT = 0.02
LOOP = 5
RENDER_MS = 1.5
LOCK_TIMEOUT = 5.0
USAGE_ERROR = 2


# --- runners ---------------------------------------------------------------- #


def test_simple_runner_advances_the_clock() -> None:
    """The simple runner steps until the requested duration is reached."""
    model = mujoco.MjModel.from_xml_string(PLAIN_XML)
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
    model = mujoco.MjModel.from_xml_string(PLAIN_XML)
    data = mujoco.MjData(model)
    controller = _CountingController()
    runners.thread_safe_runner(model, data, controller, duration=SHORT)
    assert controller.calls == int(SHORT / model.opt.timestep)


# --- mujoco_ops ------------------------------------------------------------- #


def test_self_collision_check_reports_compile_errors() -> None:
    """A spec that MuJoCo cannot compile becomes a ValueError."""

    def fail() -> None:
        msg = "boom"
        raise mujoco.FatalError(msg)

    spec = cast("mujoco.MjSpec", SimpleNamespace(compile=fail))
    with pytest.raises(ValueError, match="Failed to compile"):
        has_self_collision(spec)


# --- file_ops --------------------------------------------------------------- #


def test_save_path_defaults_to_the_data_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without a folder the path lands in the data folder."""
    monkeypatch.setattr(file_ops, "DATA", tmp_path)
    result = file_ops.generate_save_path(
        file_name="out", file_extension=".txt", append_date=False
    )
    assert result == tmp_path / "out.txt"


def test_save_path_without_a_name_uses_a_timestamp(tmp_path: Path) -> None:
    """With only a folder and an extension the name is the timestamp."""
    result = file_ops.generate_save_path(
        file_path=tmp_path, file_extension=".txt", append_date=False
    )
    assert result.suffix == ".txt"
    assert result.parent == tmp_path


def test_save_path_warns_when_there_is_no_extension(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No extension anywhere is reported on the console."""
    file_ops.generate_save_path(file_name="data", file_path=tmp_path)
    assert "No valid file extension" in capsys.readouterr().out


# --- gl_backend ------------------------------------------------------------- #


def test_configure_from_file_survives_a_vanished_folder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A working folder that no longer exists is not an error."""

    def vanished() -> Path:
        raise FileNotFoundError

    monkeypatch.setattr(gl_backend, "config_path", vanished)
    assert gl_backend.configure_from_file() is None


def test_mujoco_version_without_the_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If mujoco is not installed the version is 'unknown'."""

    def missing(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    assert gl_backend.mujoco_version() == "unknown"


def test_wait_for_lock_stops_when_the_lock_disappears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lock released between exists() and stat() ends the wait."""
    lock = tmp_path / "x.lock"
    lock.write_text("1", encoding="utf-8")
    original = type(lock).stat
    calls: list[int] = []

    def vanishing(self: Path, **kwargs: bool) -> os.stat_result:
        calls.append(1)
        if len(calls) > 1:
            raise FileNotFoundError
        return original(self, **kwargs)

    monkeypatch.setattr(type(lock), "stat", vanishing)
    gl_backend._wait_for_lock(lock, LOCK_TIMEOUT)  # ruff: ignore[private-member-access]  # tests the private helper directly
    assert len(calls) > 1


def test_ensure_uses_the_manager_and_tells_about_a_late_choice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Without a run argument the manager probes.

    If mujoco is already imported the user is told that the choice applies
    at the next start.
    """
    saved: list[Path] = []

    def fake_run(target: Path, *, console: Console | None = None) -> GLConfig:
        del console
        saved.append(target)
        return GLConfig(selected="glfw")

    monkeypatch.setattr(gl_manager, "run", fake_run)
    applied = gl_backend.ensure_gl_backend(
        tmp_path / gl_backend.CONFIG_NAME,
        environ={},
        modules={"mujoco": object()},
        console=None,
    )
    assert saved == [tmp_path / gl_backend.CONFIG_NAME]
    assert applied is None
    assert "next time you start" in " ".join(capsys.readouterr().out.split())


def test_ensure_returns_a_config_written_by_another_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the file appears just after the lock is taken it is reused."""
    path = tmp_path / gl_backend.CONFIG_NAME

    def lock_and_write(lock: Path) -> bool:
        lock.write_text("1", encoding="utf-8")
        path.write_text(GLConfig(selected="egl").model_dump_json())
        return True

    monkeypatch.setattr(gl_backend, "_try_lock", lock_and_write)
    never = Mock(return_value=GLConfig())

    applied = gl_backend.ensure_gl_backend(
        path, run=never, environ={}, modules={}
    )
    assert applied == "egl"
    never.assert_not_called()


# --- gl_manager ------------------------------------------------------------- #


def test_atomic_write_cleans_up_when_the_rename_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the final rename fails the temporary file is removed."""

    def refuse(_self: Path, target: Path) -> Path:
        raise PermissionError(str(target))

    monkeypatch.setattr(type(tmp_path), "replace", refuse)
    with pytest.raises(PermissionError):
        gl_manager.run(
            tmp_path / "config.json",
            command=lambda _backend: [sys.executable, "-c", "pass"],
            timeout=LOCK_TIMEOUT,
            console=Console(file=io.StringIO()),
            platform="darwin",
        )
    assert list(tmp_path.iterdir()) == []


def _fake_probe(*_args: object, **_kwargs: object) -> object:
    return subprocess.CompletedProcess(
        args=[], returncode=0, stdout='{"ok": true, "render_ms": 2.0}\n'
    )


def test_gl_manager_main_writes_the_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """main() probes the backends and reports the choice."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "run", _fake_probe)
    assert gl_manager.main() == 0
    assert "Selected backend" in capsys.readouterr().out
    assert (tmp_path / gl_backend.CONFIG_NAME).exists()


def test_gl_manager_script_exits_with_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running gl_manager as a script ends with exit code 0."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "run", _fake_probe)
    with pytest.raises(SystemExit) as stop:
        runpy.run_path(gl_manager.__file__, run_name="__main__")
    assert stop.value.code == 0


# --- the OpenGL probe worker ------------------------------------------------ #


class _FakeRenderer:
    def __init__(self, model: object, size: int, height: int) -> None:
        del model, size, height

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    updates = 0

    def update_scene(self, data: object) -> None:
        del data
        self.updates += 1

    def render(self) -> int:
        return self.updates


def _fake_mujoco(
    renderer: Callable[[object, int, int], object],
) -> SimpleNamespace:
    return SimpleNamespace(
        MjModel=SimpleNamespace(from_xml_string=lambda _xml: object()),
        MjData=lambda _model: object(),
        Renderer=renderer,
        mj_forward=lambda _model, _data: None,
    )


@pytest.fixture
def keep_mujoco_gl(monkeypatch: pytest.MonkeyPatch) -> None:
    # The probe worker sets MUJOCO_GL in this process; put the old value back.
    monkeypatch.setenv("MUJOCO_GL", os.environ.get("MUJOCO_GL", "glfw"))


@pytest.mark.usefixtures("keep_mujoco_gl")
def test_probe_reports_render_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """A working backend reports ok and a render time."""
    monkeypatch.setitem(sys.modules, "mujoco", _fake_mujoco(_FakeRenderer))
    monkeypatch.setattr(
        time, "perf_counter", iter([0.0, RENDER_MS / 200]).__next__
    )
    result = _gl_probe_worker.probe("egl")
    assert result["ok"] is True
    assert result["render_ms"] == pytest.approx(RENDER_MS)


@pytest.mark.usefixtures("keep_mujoco_gl")
def test_probe_reports_a_failing_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any error becomes ok=false with the error text."""

    def broken(*_args: object) -> object:
        msg = "no display"
        raise RuntimeError(msg)

    monkeypatch.setitem(sys.modules, "mujoco", _fake_mujoco(broken))
    result = _gl_probe_worker.probe("glfw")
    assert result == {"ok": False, "error": "RuntimeError: no display"}


def test_probe_main_checks_its_arguments(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Without a backend name the worker prints a usage line."""
    assert _gl_probe_worker.main(["worker"]) == USAGE_ERROR
    assert "usage" in capsys.readouterr().err


@pytest.mark.usefixtures("keep_mujoco_gl")
def test_probe_main_prints_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With a backend name one JSON line is printed."""
    monkeypatch.setitem(sys.modules, "mujoco", _fake_mujoco(_FakeRenderer))
    assert _gl_probe_worker.main(["worker", "egl"]) == 0
    assert '"ok"' in capsys.readouterr().out


@pytest.mark.usefixtures("keep_mujoco_gl")
def test_probe_worker_script(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Running the worker as a script exits cleanly."""
    monkeypatch.setitem(sys.modules, "mujoco", _fake_mujoco(_FakeRenderer))
    monkeypatch.setattr(sys, "argv", ["worker", "egl"])
    with pytest.raises(SystemExit) as stop:
        runpy.run_path(inspect.getfile(_gl_probe_worker), run_name="__main__")
    assert stop.value.code == 0
    assert '"ok"' in capsys.readouterr().out


# --- demo entry points ------------------------------------------------------ #


def test_revde_demo_logs_parents_and_children() -> None:
    """The RevDE demo runs without error."""
    revde.main()


def test_revde_script() -> None:
    """Running revde.py as a script calls main()."""
    runpy.run_path(inspect.getfile(revde), run_name="__main__")


def test_nde_script() -> None:
    """Running nde.py as a script prints the output shapes."""
    runpy.run_path(inspect.getfile(nde), run_name="__main__")


def test_save_path_for_a_folder_that_does_not_exist_yet(
    tmp_path: Path,
) -> None:
    """A target folder that is missing is not created by the path helper."""
    target = tmp_path / "later"
    result = file_ops.generate_save_path(
        file_name="out", file_extension=".txt", file_path=target
    )
    assert result.parent == target
    assert not target.exists()
