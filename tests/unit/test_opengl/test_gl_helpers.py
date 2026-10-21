"""Test: edge cases of the OpenGL backend, manager and probe worker."""

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
from typing import TYPE_CHECKING, Self
from unittest.mock import Mock

import pytest
from rich.console import Console

from ariel.opengl import gl_backend, gl_manager
from ariel.opengl.gl_backend import GLConfig

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_gl_probe_worker = importlib.import_module("ariel.opengl._gl_probe_worker")

RENDER_MS = 1.5
LOCK_TIMEOUT = 5.0
STAT_THAT_MISSES = 2  # the lock vanishes on the second stat()
USAGE_ERROR = 2


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
    """A lock released between exists() and stat() ends the wait.

    Another process holds the lock; it releases it just as this one reads
    the lock's age, and this one then takes the lock and probes.
    """
    path = tmp_path / gl_backend.CONFIG_NAME
    lock = path.with_name(path.name + gl_backend.LOCK_SUFFIX)
    lock.write_text("1", encoding="utf-8")
    original = type(lock).stat
    calls: list[int] = []

    def vanishing(self: Path, **kwargs: bool) -> os.stat_result:
        if self == lock:
            calls.append(1)
            if len(calls) == STAT_THAT_MISSES:  # exists() saw it
                lock.unlink()
                raise FileNotFoundError
        return original(self, **kwargs)

    monkeypatch.setattr(type(lock), "stat", vanishing)
    applied = gl_backend.ensure_gl_backend(
        path,
        run=lambda _target: GLConfig(selected="egl"),
        environ={},
        modules={},
        lock_timeout=LOCK_TIMEOUT,
    )
    assert applied == "egl"
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
