"""Tests for the structure measurement script in ``tools/``."""

# Standard library
import importlib.util
import runpy
import sys
from pathlib import Path
from types import ModuleType

# Third-party libraries
import pytest

FILES = 7
LONG = 401
WIDE = 6
DEEP = 7
SCRIPT = Path(__file__).resolve().parents[3] / "tools" / "measure_structure.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_structure", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tree(root: Path) -> Path:
    """Build a tiny package with one long, one wide and one deep module.

    Returns
    -------
    Path
        The ``src`` folder of the tree.
    """
    src = root / "src"
    pkg = src / "ariel"
    (pkg / "a").mkdir(parents=True)
    (pkg / "b").mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "a" / "__init__.py").write_text("")
    (pkg / "b" / "__init__.py").write_text("")
    (pkg / "b" / "leaf.py").write_text("x = 1\n")
    (pkg / "a" / "long.py").write_text("y = 1\n" * 401)
    names = "\n".join(f"import ariel.b.m{i}" for i in range(6))
    (pkg / "a" / "wide.py").write_text(names + "\n")
    (pkg / "a" / "deep.py").write_text(
        "from ariel.b.leaf import x\nimport ariel\n",
    )
    return src


def test_module_name_drops_init_and_suffix(tmp_path: Path) -> None:
    script = _load()
    src = _tree(tmp_path)
    assert script.module_name(src / "ariel" / "a" / "long.py", src) == (
        "ariel.a.long"
    )
    assert script.module_name(src / "ariel" / "a" / "__init__.py", src) == (
        "ariel.a"
    )


def test_imports_of_keeps_only_ariel_absolute_imports(tmp_path: Path) -> None:
    script = _load()
    path = tmp_path / "m.py"
    path.write_text(
        "import os\nimport ariel.b\nfrom ariel.c import d\n"
        "from . import e\nfrom os import path\n",
    )
    assert script.imports_of(path) == {"ariel.b", "ariel.c"}


def test_subpackage_of_the_package_itself_is_empty() -> None:
    script = _load()
    assert not script.subpackage("ariel")
    assert script.subpackage("ariel.ec.ea") == "ec"


def test_measure_finds_long_wide_and_deep(tmp_path: Path) -> None:
    script = _load()
    src = _tree(tmp_path)
    result = script.measure(src / "ariel", src, tmp_path)
    assert result.files == FILES
    assert [n for n, _ in result.long_files] == [LONG]
    assert [n for n, _ in result.wide_modules] == [WIDE]
    # wide.py has 6 imports of ariel.b.mN, deep.py one of ariel.b.leaf
    assert result.deep_imports == ((DEEP, "a -> b"),)


def test_measure_skips_deep_imports_outside_src(tmp_path: Path) -> None:
    script = _load()
    src = _tree(tmp_path)
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_x.py").write_text("from ariel.b.leaf import x\n")
    result = script.measure(tests, src, tmp_path)
    assert result.files == 1
    assert result.deep_imports == ()


def test_render_lists_every_fact(tmp_path: Path) -> None:
    script = _load()
    src = _tree(tmp_path)
    text = script.render(
        "src/ariel", script.measure(src / "ariel", src, tmp_path)
    )
    assert "src/ariel: 7 files" in text
    assert "files over 400 lines: 1" in text
    assert "modules importing more than 5 others: 1" in text
    assert "imports that skip a package entry point: 7" in text


def test_main_reports_src_and_tests(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _load().main() == 0
    out = capsys.readouterr().out
    assert "src/ariel:" in out
    assert "tests:" in out


def test_script_runs_as_main(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delitem(sys.modules, "measure_structure", raising=False)
    with pytest.raises(SystemExit) as stop:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert stop.value.code == 0
    assert "src/ariel:" in capsys.readouterr().out
