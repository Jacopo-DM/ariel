"""Test: probing OpenGL backends in subprocesses and writing the config."""

import io
import json
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from rich.console import Console

from ariel.opengl import gl_manager
from ariel.opengl.gl_backend import (
    CONFIG_NAME,
    BackendResult,
    platform_backends,
)

_OK = (
    "import json, sys;"
    "ms = {'glfw': 40.0, 'egl': 10.0, 'osmesa': 300.0}[sys.argv[1]];"
    "print('noise before'); print(json.dumps({'ok': True, 'render_ms': ms}))"
)
_FAIL = "import json;print(json.dumps({'ok': False, 'error': 'no display'}))"
_CRASH = "import os; os.abort()"
_HANG = "import time; time.sleep(60)"
_GARBAGE = "print('not json at all')"


def _command(script: str) -> Callable[[str], list[str]]:
    """Return a worker command factory that runs `script` per backend.

    Returns
    -------
    Callable[[str], list[str]]
        The command factory.
    """

    def build(backend: str) -> list[str]:
        return [sys.executable, "-c", script, backend]

    return build


def _per_backend(
    scripts: dict[str, str],
) -> Callable[[str], list[str]]:
    """Return a worker command factory with a different script per backend.

    Returns
    -------
    Callable[[str], list[str]]
        The command factory.
    """

    def build(backend: str) -> list[str]:
        return [sys.executable, "-c", scripts[backend], backend]

    return build


def _quiet() -> Console:
    return Console(file=io.StringIO())


def test_probe_backend_success() -> None:
    """A worker that prints a JSON line yields a successful result."""
    result = gl_manager.probe_backend("egl", _command(_OK), timeout=20)
    assert result.ok is True
    assert result.render_ms == pytest.approx(10.0)
    assert result.error is None


def test_probe_backend_reported_failure() -> None:
    """A worker that reports ok=false keeps its reason."""
    result = gl_manager.probe_backend("glfw", _command(_FAIL), timeout=20)
    assert result.ok is False
    assert result.render_ms is None
    assert result.error == "no display"


def test_probe_backend_crash_is_recorded() -> None:
    """A worker killed by a signal is recorded as failed, not raised."""
    result = gl_manager.probe_backend("glfw", _command(_CRASH), timeout=20)
    assert result.ok is False
    assert result.error is not None
    assert "exit code" in result.error


def test_probe_backend_timeout_is_recorded() -> None:
    """A hanging worker is killed and recorded as timed out."""
    result = gl_manager.probe_backend("egl", _command(_HANG), timeout=0.5)
    assert result.ok is False
    assert result.error is not None
    assert "timed out" in result.error


def test_probe_backend_garbage_output() -> None:
    """Output that is not JSON is recorded as a failure."""
    result = gl_manager.probe_backend("egl", _command(_GARBAGE), timeout=20)
    assert result.ok is False
    assert result.error is not None
    assert "unreadable" in result.error


def test_probe_backend_missing_executable() -> None:
    """A command that cannot be started is recorded as failed."""

    def build(_backend: str) -> list[str]:
        return ["/nonexistent/python-for-ariel-tests"]

    result = gl_manager.probe_backend("egl", build, timeout=5)
    assert result.ok is False
    assert result.error is not None


WORKER_VALUES = (
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats()
    | st.text()
    | st.lists(st.integers(), max_size=2)
)
WORKER_LINES = st.one_of(
    st.text(),
    st.dictionaries(
        st.sampled_from(["ok", "render_ms", "error"]), WORKER_VALUES
    ).map(json.dumps),
    st.lists(WORKER_VALUES, max_size=2).map(json.dumps),
)


@given(st.lists(WORKER_LINES, max_size=3).map("\n".join))
@example('noise\n{"ok": true, "render_ms": 12.5}')
@example('{"ok": true, "": null}')
@example('{"ok": true, "render_ms": null}')
@example('{"ok": true, "render_ms": "fast"}')
def test_parse_worker_output_boundary(stdout: str) -> None:
    """Any worker output gives a BackendResult, never an exception."""
    result = gl_manager._parse_worker_output(stdout)  # ruff: ignore[private-member-access]  # the parser of the probe's stdout
    assert isinstance(result, BackendResult)


@pytest.mark.parametrize(
    "stdout",
    ['{"ok": true}', '{"ok": true, "render_ms": "fast"}'],
)
def test_success_without_a_time_is_unreadable(stdout: str) -> None:
    """A reported success with no numeric render time counts as unreadable."""
    result = gl_manager._parse_worker_output(stdout)  # ruff: ignore[private-member-access]  # the parser of the probe's stdout
    assert result == BackendResult(ok=False, error="worker output unreadable")


def test_select_backend_prefers_fastest() -> None:
    """The working backend with the lowest render time is selected."""
    results = {
        "glfw": BackendResult(ok=True, render_ms=40.0),
        "egl": BackendResult(ok=True, render_ms=10.0),
        "osmesa": BackendResult(ok=True, render_ms=300.0),
    }
    assert gl_manager.select_backend(results) == "egl"


def test_select_backend_ties_follow_priority_order() -> None:
    """Equal render times resolve in the order glfw, egl, osmesa."""
    results = {
        "osmesa": BackendResult(ok=True, render_ms=5.0),
        "egl": BackendResult(ok=True, render_ms=5.0),
        "glfw": BackendResult(ok=False, error="x"),
    }
    assert gl_manager.select_backend(results) == "egl"


def test_select_backend_none_working() -> None:
    """No working backend means no selection."""
    results = {"glfw": BackendResult(ok=False, error="x")}
    assert gl_manager.select_backend(results) is None


def test_platform_backends() -> None:
    """Linux probes three backends; other platforms probe glfw only."""
    assert platform_backends("linux") == ("glfw", "egl", "osmesa")
    assert platform_backends("darwin") == ("glfw",)
    assert platform_backends("win32") == ("glfw",)


def test_run_writes_valid_config(tmp_path: Path) -> None:
    """The manager writes the documented schema and picks the fastest."""
    command = _per_backend({"glfw": _FAIL, "egl": _OK, "osmesa": _CRASH})
    path = tmp_path / CONFIG_NAME
    config = gl_manager.run(
        path, command=command, timeout=20, console=_quiet(), platform="linux"
    )
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk == config.model_dump(mode="json")
    assert on_disk["version"] == 1
    assert on_disk["platform"] == "linux"
    assert set(on_disk) >= {"python", "mujoco", "backends", "selected"}
    assert on_disk["selected"] == "egl"
    assert on_disk["viewer_available"] is False
    assert on_disk["backends"]["glfw"] == {
        "ok": False,
        "render_ms": None,
        "error": "no display",
    }
    assert on_disk["backends"]["egl"]["ok"] is True
    assert on_disk["backends"]["osmesa"]["ok"] is False


def test_run_when_every_backend_fails(tmp_path: Path) -> None:
    """Nothing working gives selected=null and no viewer, without raising."""
    path = tmp_path / CONFIG_NAME
    config = gl_manager.run(
        path,
        command=_command(_CRASH),
        timeout=20,
        console=_quiet(),
        platform="linux",
    )
    assert config.selected is None
    assert config.viewer_available is False
    assert all(not b.ok for b in config.backends.values())


def test_run_viewer_available_when_glfw_works(tmp_path: Path) -> None:
    """viewer_available is true exactly when glfw works."""
    path = tmp_path / CONFIG_NAME
    config = gl_manager.run(
        path,
        command=_command(_OK),
        timeout=20,
        console=_quiet(),
        platform="linux",
    )
    assert config.viewer_available is True


def test_run_leaves_no_temporary_files(tmp_path: Path) -> None:
    """The atomic write leaves only the config file behind."""
    gl_manager.run(
        tmp_path / CONFIG_NAME,
        command=_command(_OK),
        timeout=20,
        console=_quiet(),
        platform="linux",
    )
    assert [p.name for p in tmp_path.iterdir()] == [CONFIG_NAME]


def test_worker_reports_invalid_backend_as_json() -> None:
    """The real worker turns a bogus backend into an ok=false JSON line."""
    result = gl_manager.probe_backend(
        "bogus", gl_manager.default_command, timeout=60
    )
    assert result.ok is False
    assert result.error
