"""Test: reading the OpenGL config and applying it to MUJOCO_GL."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

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
    return subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            "import os, sys; import ariel;"
            "print('mujoco' in sys.modules, os.environ.get('MUJOCO_GL'))",
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
