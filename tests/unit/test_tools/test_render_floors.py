"""Tests for the floor render script in ``tools/`` (the renderer is faked)."""

# Standard library
import importlib.util
import runpy
import sys
from pathlib import Path
from types import ModuleType

# Third-party libraries
import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "tools" / "render_floors.py"
WORLD_COUNT = 9


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("render_floors", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake(calls: list[Path]) -> object:
    def render(*_args: object, save_path: Path, **_kwargs: object) -> None:
        calls.append(save_path)

    return render


def test_render_all_asks_for_one_picture_per_world(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script = _load()
    calls: list[Path] = []
    monkeypatch.setattr(script, "single_frame_renderer", _fake(calls))
    written = script.render_all(tmp_path / "floors")
    assert written == calls
    assert len(written) == WORLD_COUNT
    assert {p.name for p in written} == {f"{n}.png" for n in script.WORLDS}


def test_main_prints_the_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script = _load()
    monkeypatch.setattr(script, "single_frame_renderer", _fake([]))
    monkeypatch.setattr(script, "DATA", tmp_path)
    assert script.main() == 0
    assert capsys.readouterr().out.count("simple-flat.png") == 1


def test_script_runs_as_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("ariel.DATA", tmp_path)
    monkeypatch.delitem(sys.modules, "render_floors", raising=False)
    monkeypatch.setattr(
        "ariel.rendering.single_frame_renderer",
        _fake([]),
    )
    with pytest.raises(SystemExit) as stop:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert stop.value.code == 0
