"""Test: reading the OpenGL config and applying it to MUJOCO_GL."""

import io
import json
import os
import subprocess  # ruff: ignore[suspicious-subprocess-import]  # runs fixed local probe commands only
import sys
import threading
import time
from pathlib import Path

import pytest
from rich.console import Console

from ariel.utils import gl_backend
from ariel.utils.gl_backend import CONFIG_NAME


def _write(folder: Path, content: object) -> None:
    text = content if isinstance(content, str) else json.dumps(content)
    (folder / CONFIG_NAME).write_text(text, encoding="utf-8")


def test_read_config_missing_returns_none(tmp_path: Path) -> None:
    """No file is not an error."""
    assert gl_backend.read_config(tmp_path / CONFIG_NAME) is None


def test_read_config_valid(tmp_path: Path) -> None:
    """A valid file is returned as a dict."""
    _write(tmp_path, {"version": 1, "selected": "egl"})
    config = gl_backend.read_config(tmp_path / CONFIG_NAME)
    assert config == {"version": 1, "selected": "egl"}


@pytest.mark.parametrize(
    "content",
    ["{not json", "[1, 2]", '{"selected": 3}', '{"selected": "bogus"}'],
)
def test_read_config_invalid_warns(tmp_path: Path, content: str) -> None:
    """Corrupt or malformed content warns and is ignored."""
    _write(tmp_path, content)
    with pytest.warns(UserWarning, match=CONFIG_NAME):
        assert gl_backend.read_config(tmp_path / CONFIG_NAME) is None


def test_apply_config_sets_selected_backend() -> None:
    """With MUJOCO_GL unset and mujoco not imported, selected is applied."""
    environ: dict[str, str] = {}
    applied = gl_backend.apply_config(
        {"selected": "osmesa"}, environ=environ, modules={}
    )
    assert applied == "osmesa"
    assert environ == {"MUJOCO_GL": "osmesa"}


def test_apply_config_never_overrides_environment() -> None:
    """A preset MUJOCO_GL wins over the file."""
    environ = {"MUJOCO_GL": "egl"}
    applied = gl_backend.apply_config(
        {"selected": "osmesa"}, environ=environ, modules={}
    )
    assert applied is None
    assert environ == {"MUJOCO_GL": "egl"}


def test_apply_config_too_late_when_mujoco_already_imported() -> None:
    """Once mujoco is imported the variable would be ignored: do nothing."""
    environ: dict[str, str] = {}
    applied = gl_backend.apply_config(
        {"selected": "osmesa"}, environ=environ, modules={"mujoco": object()}
    )
    assert applied is None
    assert environ == {}


def test_apply_config_nothing_selected() -> None:
    """selected=null leaves MuJoCo's default alone."""
    environ: dict[str, str] = {}
    assert (
        gl_backend.apply_config({"selected": None}, environ=environ, modules={})
        is None
    )
    assert environ == {}


def test_configure_from_file_reads_and_applies(tmp_path: Path) -> None:
    """The one-call entry point reads the file and applies it."""
    _write(tmp_path, {"version": 1, "selected": "glfw"})
    environ: dict[str, str] = {}
    applied = gl_backend.configure_from_file(
        tmp_path / CONFIG_NAME, environ=environ, modules={}
    )
    assert applied == "glfw"
    assert environ["MUJOCO_GL"] == "glfw"


def test_configure_from_file_swallows_unreadable_folder() -> None:
    """A path that cannot be read never raises."""
    environ: dict[str, str] = {}
    applied = gl_backend.configure_from_file(
        Path("/nonexistent-folder-for-ariel-tests") / CONFIG_NAME,
        environ=environ,
        modules={},
    )
    assert applied is None


def _import_ariel(
    folder: Path, preset: str | None = None
) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "MUJOCO_GL"}
    if preset is not None:
        env["MUJOCO_GL"] = preset
    return subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os, sys; import ariel;"
                "print('mujoco' in sys.modules, os.environ.get('MUJOCO_GL'))"
            ),
        ],
        cwd=folder,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_import_ariel_applies_selected(tmp_path: Path) -> None:
    """A fresh process that imports ariel gets MUJOCO_GL from the file."""
    _write(tmp_path, {"version": 1, "selected": "osmesa"})
    done = _import_ariel(tmp_path)
    assert done.returncode == 0
    assert done.stdout.split() == ["False", "osmesa"]


def test_import_ariel_keeps_preset_variable(tmp_path: Path) -> None:
    """The environment variable still wins after import ariel."""
    _write(tmp_path, {"version": 1, "selected": "osmesa"})
    done = _import_ariel(tmp_path, preset="egl")
    assert done.stdout.split() == ["False", "egl"]


def test_import_ariel_with_nothing_selected(tmp_path: Path) -> None:
    """selected=null leaves MUJOCO_GL unset."""
    _write(tmp_path, {"version": 1, "selected": None})
    done = _import_ariel(tmp_path)
    assert done.returncode == 0
    assert done.stdout.split() == ["False", "None"]


def test_import_ariel_with_corrupt_file_warns(tmp_path: Path) -> None:
    """A corrupt file warns but ariel still imports."""
    _write(tmp_path, "{broken")
    done = _import_ariel(tmp_path)
    assert done.returncode == 0
    assert done.stdout.split() == ["False", "None"]
    assert CONFIG_NAME in done.stderr


def test_import_ariel_without_file(tmp_path: Path) -> None:
    """No file: nothing happens, nothing is printed to stderr."""
    done = _import_ariel(tmp_path)
    assert done.returncode == 0
    assert done.stdout.split() == ["False", "None"]


# --- ensure_gl_backend: probe once when the file is missing --------------- #


class _FakeRun:
    """Stand-in for the manager: writes a config and counts its calls."""

    def __init__(self, selected: str | None = "egl", delay: float = 0.0):
        self.calls = 0
        self.selected = selected
        self.delay = delay

    def __call__(self, path: Path) -> dict[str, object]:
        self.calls += 1
        time.sleep(self.delay)
        config: dict[str, object] = {
            "version": 1,
            "selected": self.selected,
        }
        path.write_text(json.dumps(config), encoding="utf-8")
        return config


def _quiet_console() -> Console:
    return Console(file=io.StringIO(), width=120)


def test_ensure_probes_once_when_file_missing(tmp_path: Path) -> None:
    """A missing file triggers one probe; the next call reuses the file."""
    run = _FakeRun()
    path = tmp_path / CONFIG_NAME
    environ: dict[str, str] = {}
    first = gl_backend.ensure_gl_backend(
        path, run=run, environ=environ, modules={}, console=_quiet_console()
    )
    second = gl_backend.ensure_gl_backend(
        path, run=run, environ={}, modules={}, console=_quiet_console()
    )
    assert run.calls == 1
    assert first == second == "egl"
    assert environ == {"MUJOCO_GL": "egl"}
    assert not (tmp_path / (CONFIG_NAME + ".lock")).exists()


def test_ensure_does_nothing_when_environment_is_set(tmp_path: Path) -> None:
    """A preset MUJOCO_GL makes any probe pointless, so none runs."""
    run = _FakeRun()
    applied = gl_backend.ensure_gl_backend(
        tmp_path / CONFIG_NAME,
        run=run,
        environ={"MUJOCO_GL": "glfw"},
        modules={},
        console=_quiet_console(),
    )
    assert applied is None
    assert run.calls == 0


def test_ensure_when_mujoco_already_imported(tmp_path: Path) -> None:
    """The probe still runs and is saved, but applies from the next run."""
    run = _FakeRun()
    console = _quiet_console()
    environ: dict[str, str] = {}
    applied = gl_backend.ensure_gl_backend(
        tmp_path / CONFIG_NAME,
        run=run,
        environ=environ,
        modules={"mujoco": object()},
        console=console,
    )
    assert applied is None
    assert run.calls == 1
    assert environ == {}
    assert "next time" in console.file.getvalue()  # type: ignore[attr-defined]


def test_ensure_reprobes_a_corrupt_file(tmp_path: Path) -> None:
    """A corrupt file warns and is replaced by a fresh probe."""
    _write(tmp_path, "{broken")
    run = _FakeRun(selected="osmesa")
    with pytest.warns(UserWarning, match=CONFIG_NAME):
        applied = gl_backend.ensure_gl_backend(
            tmp_path / CONFIG_NAME,
            run=run,
            environ={},
            modules={},
            console=_quiet_console(),
        )
    assert applied == "osmesa"
    assert run.calls == 1


def test_ensure_threads_probe_once(tmp_path: Path) -> None:
    """Concurrent callers wait for the first probe instead of repeating it."""
    run = _FakeRun(delay=0.5)
    path = tmp_path / CONFIG_NAME
    results: list[str | None] = []

    def call() -> None:
        results.append(
            gl_backend.ensure_gl_backend(
                path,
                run=run,
                environ={},
                modules={},
                console=_quiet_console(),
            )
        )

    threads = [threading.Thread(target=call) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert run.calls == 1
    assert results == ["egl", "egl", "egl"]


_PROCESS_SCRIPT = """
import io, json, sys, time
from pathlib import Path
from rich.console import Console
from ariel.utils import gl_backend

folder = Path(sys.argv[1])
def fake(path):
    with (folder / "calls.txt").open("a") as f:
        f.write("probe\\n")
    time.sleep(1.0)
    config = {"version": 1, "selected": "egl"}
    path.write_text(json.dumps(config))
    return config

gl_backend.ensure_gl_backend(
    folder / "mujoco_user_config.json", run=fake, environ={}, modules={},
    console=Console(file=io.StringIO()),
)
"""


def test_ensure_two_processes_probe_once(tmp_path: Path) -> None:
    """Two processes starting together produce one probe and one file."""
    procs = [
        subprocess.Popen(  # ruff: ignore[subprocess-without-shell-equals-true]
            [sys.executable, "-c", _PROCESS_SCRIPT, str(tmp_path)]
        )
        for _ in range(2)
    ]
    assert [p.wait(timeout=120) for p in procs] == [0, 0]
    calls = (tmp_path / "calls.txt").read_text().splitlines()
    assert calls == ["probe"]
    assert (tmp_path / CONFIG_NAME).is_file()


def test_ensure_removes_a_stale_lock(tmp_path: Path) -> None:
    """A lock left by a dead process does not block forever."""
    lock = tmp_path / (CONFIG_NAME + ".lock")
    lock.write_text("dead process")
    old = time.time() - 3600
    os.utime(lock, (old, old))
    run = _FakeRun()
    applied = gl_backend.ensure_gl_backend(
        tmp_path / CONFIG_NAME,
        run=run,
        environ={},
        modules={},
        console=_quiet_console(),
        lock_timeout=5.0,
    )
    assert applied == "egl"
    assert run.calls == 1


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="read-only folders cannot be simulated as root",
)
def test_ensure_in_read_only_folder_only_warns(tmp_path: Path) -> None:
    """If the result cannot be saved, ARIEL warns and carries on."""
    folder = tmp_path / "ro"
    folder.mkdir()
    folder.chmod(0o500)
    run = _FakeRun()
    try:
        with pytest.warns(UserWarning, match="cannot save"):
            applied = gl_backend.ensure_gl_backend(
                folder / CONFIG_NAME,
                run=run,
                environ={},
                modules={},
                console=_quiet_console(),
            )
    finally:
        folder.chmod(0o700)
    assert applied is None
    assert run.calls == 0


def test_import_renderers_probes_once_for_real(tmp_path: Path) -> None:
    """Importing ariel.utils.renderers creates the file the first time only."""
    env = {k: v for k, v in os.environ.items() if k != "MUJOCO_GL"}
    command = [sys.executable, "-c", "import ariel.utils.renderers"]
    first = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        command,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert first.returncode == 0
    assert "One-off setup" in first.stdout + first.stderr
    config = json.loads((tmp_path / CONFIG_NAME).read_text(encoding="utf-8"))
    assert set(config["backends"]) == {"glfw", "egl", "osmesa"}
    second = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        command,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert second.returncode == 0
    assert "One-off setup" not in second.stdout + second.stderr


# --- stale results: reprobe when mujoco or the platform changed ----------- #


def _stale_case(
    tmp_path: Path, recorded: dict[str, object]
) -> tuple[_FakeRun, str | None]:
    _write(tmp_path, {"version": 1, "selected": "glfw", **recorded})
    run = _FakeRun(selected="egl")
    applied = gl_backend.ensure_gl_backend(
        tmp_path / CONFIG_NAME,
        run=run,
        environ={},
        modules={},
        console=_quiet_console(),
    )
    return run, applied


def test_ensure_reprobes_when_mujoco_version_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file written for another mujoco version is probed again."""
    monkeypatch.setattr(gl_backend, "mujoco_version", lambda: "9.9.9")
    run, applied = _stale_case(
        tmp_path, {"mujoco": "1.0.0", "platform": sys.platform}
    )
    assert run.calls == 1
    assert applied == "egl"


def test_ensure_reprobes_when_platform_changed(tmp_path: Path) -> None:
    """A file copied from another platform is probed again."""
    run, applied = _stale_case(
        tmp_path,
        {"mujoco": gl_backend.mujoco_version(), "platform": "other-os"},
    )
    assert run.calls == 1
    assert applied == "egl"


def test_ensure_keeps_a_current_file(tmp_path: Path) -> None:
    """A file for this mujoco and platform is used without probing."""
    run, applied = _stale_case(
        tmp_path,
        {"mujoco": gl_backend.mujoco_version(), "platform": sys.platform},
    )
    assert run.calls == 0
    assert applied == "glfw"


def test_ensure_keeps_a_file_without_versions(tmp_path: Path) -> None:
    """A hand-written file that lacks the version fields is not discarded."""
    run, applied = _stale_case(tmp_path, {})
    assert run.calls == 0
    assert applied == "glfw"
