"""Tests for the notebook runner in ``tools/`` (tiny notebooks, real kernel)."""

# Standard library
import importlib.util
import subprocess  # ruff: ignore[suspicious-subprocess-import]  # only builds fake results
from pathlib import Path
from types import ModuleType

# Third-party libraries
import nbformat
import pytest

BOTH = 2
pytestmark = pytest.mark.usefixtures("quiet_jupyter")
SCRIPT = Path(__file__).resolve().parents[3] / "tools" / "run_notebooks.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_notebooks", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def quiet_jupyter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the new Jupyter folders so the kernels print no migration warning."""
    monkeypatch.setenv("JUPYTER_PLATFORM_DIRS", "1")


def _write(
    path: Path,
    *sources: str,
    requires: list[str] | None = None,
    slow: bool = False,
) -> None:
    notebook = nbformat.v4.new_notebook()  # type: ignore[no-untyped-call]  # nbformat is untyped
    notebook.cells = [
        nbformat.v4.new_code_cell(source)  # type: ignore[no-untyped-call]  # nbformat is untyped
        for source in sources
    ]
    if requires is not None:
        notebook.metadata["requires"] = requires
    if slow:
        notebook.metadata["slow"] = True
    nbformat.write(notebook, path)  # type: ignore[no-untyped-call]  # nbformat is untyped


def test_a_clean_notebook_passes(tmp_path: Path) -> None:
    script = _load()
    path = tmp_path / "good.ipynb"
    _write(path, "x = 1", "assert x == 1")
    outcome = script.run_notebook(path, opengl=False)
    assert outcome.status == "PASS"
    assert outcome.name == "good"
    assert outcome.seconds > 0


def test_a_failing_cell_is_named_in_the_detail(tmp_path: Path) -> None:
    script = _load()
    path = tmp_path / "bad.ipynb"
    _write(path, "x = 1", "raise ValueError('boom')")
    outcome = script.run_notebook(path, opengl=False)
    assert outcome.status == "FAIL"
    assert outcome.detail == "cell 2: ValueError: boom"


def test_a_notebook_that_needs_opengl_is_skipped_without_it(
    tmp_path: Path,
) -> None:
    script = _load()
    path = tmp_path / "gl.ipynb"
    _write(path, "raise RuntimeError('never runs')", requires=["opengl"])
    assert script.run_notebook(path, opengl=False).status == "SKIP"
    assert script.run_notebook(path, opengl=True).status == "FAIL"


def test_a_timeout_is_a_failure(tmp_path: Path) -> None:
    script = _load()
    path = tmp_path / "slow.ipynb"
    _write(path, "import time\ntime.sleep(30)")
    outcome = script.run_notebook(path, opengl=False, timeout=1)
    assert outcome.status == "FAIL"
    assert outcome.detail.startswith("timeout after 1 s")


def test_save_writes_the_outputs_only_when_asked(tmp_path: Path) -> None:
    script = _load()
    path = tmp_path / "out.ipynb"
    _write(path, "print('hello')")
    script.run_notebook(path, opengl=False)
    assert not script.read_notebook(path).cells[0].outputs
    script.run_notebook(path, opengl=False, save=True)
    assert script.read_notebook(path).cells[0].outputs[0].text == "hello\n"


def test_main_prints_one_line_each_and_fails_on_a_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script = _load()
    _write(tmp_path / "a_good.ipynb", "x = 1")
    _write(tmp_path / "b_bad.ipynb", "raise KeyError('k')")
    _write(tmp_path / "c_gl.ipynb", "x = 1", requires=["opengl"])
    monkeypatch.setattr(script, "FUNCTIONAL", tmp_path)
    monkeypatch.setattr(script, "opengl_works", lambda: False)
    assert script.main([]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert [line.split()[:2] for line in lines if line[0] != " "] == [
        ["PASS", "a_good"],
        ["FAIL", "b_bad"],
        ["SKIP", "c_gl"],
    ]
    assert "cell 1: KeyError" in lines[2]
    assert "needs opengl" in lines[4]


def test_main_returns_zero_when_nothing_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script = _load()
    _write(tmp_path / "ok.ipynb", "x = 1")
    monkeypatch.setattr(script, "FUNCTIONAL", tmp_path)
    monkeypatch.setattr(script, "opengl_works", lambda: True)
    assert script.main(["--save"]) == 0


@pytest.mark.parametrize(("code", "expected"), [(0, True), (1, False)])
def test_opengl_works_follows_the_probe_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    code: int,
    expected: bool,  # ruff: ignore[boolean-type-hint-positional-argument]  # pytest parameter
) -> None:
    script = _load()

    def fake(*_args: object, **_kwargs: object) -> object:
        return subprocess.CompletedProcess(args=[], returncode=code)

    monkeypatch.setattr(script.subprocess, "run", fake)
    assert script.opengl_works() is expected


def test_slow_notebooks_run_only_when_asked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script = _load()
    _write(tmp_path / "fast.ipynb", "x = 1")
    _write(tmp_path / "slow.ipynb", "x = 1", slow=True)
    monkeypatch.setattr(script, "FUNCTIONAL", tmp_path)
    monkeypatch.setattr(script, "opengl_works", lambda: True)
    script.main([])
    assert [ln.split()[1] for ln in capsys.readouterr().out.splitlines()] == [
        "fast"
    ]
    script.main(["--slow", "--timeout", "120"])
    assert len(capsys.readouterr().out.splitlines()) == BOTH


def test_file_arguments_run_only_those_notebooks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script = _load()
    _write(tmp_path / "one.ipynb", "x = 1")
    _write(tmp_path / "two.ipynb", "x = 1", slow=True)
    monkeypatch.setattr(script, "opengl_works", lambda: True)
    assert script.main([str(tmp_path / "two.ipynb")]) == 0
    assert capsys.readouterr().out.split()[1] == "two"


def test_a_kernel_crash_is_a_failure(tmp_path: Path) -> None:
    script = _load()
    path = tmp_path / "crash.ipynb"
    _write(path, "import os\nos._exit(1)")
    outcome = script.run_notebook(path, opengl=False)
    assert outcome.status == "FAIL"
    assert outcome.detail.startswith("the kernel died")


def test_an_error_without_a_cell_leaves_out_the_cell_number(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script = _load()
    monkeypatch.setattr(script, "failing_cell", lambda _notebook: 0)
    path = tmp_path / "bad.ipynb"
    _write(path, "raise ValueError('boom')")
    assert script.run_notebook(path, opengl=False).detail == "ValueError: boom"
