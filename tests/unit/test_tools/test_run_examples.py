"""Tests for the examples runner in ``tools/`` (tiny scripts, real runs)."""

# Standard library
import importlib.util
import subprocess  # ruff: ignore[suspicious-subprocess-import]  # only builds fake results
from pathlib import Path
from types import ModuleType

# Third-party libraries
import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "tools" / "run_examples.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_examples", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _examples(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    scripts: dict[str, str],
) -> ModuleType:
    """Point the runner at a folder of the given scripts, with no rows yet.

    Returns
    -------
    ModuleType
        The runner module, with its folder and tables replaced.
    """
    script = _load()
    folder = tmp_path / "examples"
    for name, source in scripts.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    monkeypatch.setattr(script, "EXAMPLES_FOLDER", folder)
    monkeypatch.setattr(script, "EXAMPLES", {})
    monkeypatch.setattr(script, "MODULES", frozenset())
    monkeypatch.setattr(script, "opengl_works", lambda: True)
    return script


def _lines(capsys: pytest.CaptureFixture[str]) -> list[list[str]]:
    out = capsys.readouterr().out.splitlines()
    return [line.split()[:2] for line in out if not line.startswith(" ")]


def test_a_clean_script_passes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script = _examples(monkeypatch, tmp_path, {"ok.py": "x = 1\n"})
    work = tmp_path / "work"
    work.mkdir()
    outcome = script.run_example("ok.py", script.row(), work, opengl=False)
    assert outcome.status == "PASS"
    assert outcome.seconds > 0


def test_a_failure_shows_the_last_line_of_stderr(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script = _examples(
        monkeypatch, tmp_path, {"bad.py": "raise ValueError('boom')\n"}
    )
    outcome = script.run_example("bad.py", script.row(), tmp_path, opengl=False)
    assert outcome.status == "FAIL"
    assert outcome.detail == "exit 1: ValueError: boom"


def test_a_failure_without_stderr_shows_stdout_or_nothing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script = _examples(
        monkeypatch,
        tmp_path,
        {
            "says.py": "import sys\nprint('no database')\nsys.exit(2)\n",
            "silent.py": "import sys\nsys.exit(3)\n",
        },
    )
    said = script.run_example("says.py", script.row(), tmp_path, opengl=False)
    assert said.detail == "exit 2: no database"
    quiet = script.run_example(
        "silent.py", script.row(), tmp_path, opengl=False
    )
    assert quiet.detail == "exit 3: "


def test_a_timeout_is_a_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script = _examples(
        monkeypatch, tmp_path, {"slow.py": "import time\ntime.sleep(30)\n"}
    )
    example = script.Example(timeout=1)
    outcome = script.run_example("slow.py", example, tmp_path, opengl=False)
    assert outcome.status == "FAIL"
    assert outcome.detail == "timeout after 1 s"


def test_a_rendering_script_is_skipped_without_opengl(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script = _examples(
        monkeypatch, tmp_path, {"gl.py": "raise RuntimeError('ran')\n"}
    )
    example = script.row(opengl=True)
    skipped = script.run_example("gl.py", example, tmp_path, opengl=False)
    assert skipped.status == "SKIP"
    assert "needs opengl" in skipped.detail
    ran = script.run_example("gl.py", example, tmp_path, opengl=True)
    assert ran.status == "FAIL"


def test_scripts_get_arguments_no_input_and_a_quiet_matplotlib(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = (
        "import os, sys\n"
        "assert sys.argv[1:] == ['--pop', '2']\n"
        "assert os.environ['MPLBACKEND'] == 'Agg'\n"
        "assert sys.stdin.read() == ''\n"
    )
    script = _examples(monkeypatch, tmp_path, {"env.py": source})
    outcome = script.run_example(
        "env.py", script.row("--pop 2"), tmp_path, opengl=False
    )
    assert outcome.status == "PASS", outcome.detail


def test_a_pattern_becomes_its_last_match(tmp_path: Path) -> None:
    script = _load()
    data = tmp_path / "__data__"
    data.mkdir()
    (data / "best_1.json").write_text("{}")
    (data / "best_2.json").write_text("{}")
    args = ("--morphology", "__data__/best_*.json", "--dur", "1")
    assert script.expand(args, tmp_path) == [
        "--morphology",
        "__data__/best_2.json",
        "--dur",
        "1",
    ]


def test_a_pattern_without_a_match_is_a_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    script = _examples(monkeypatch, tmp_path, {"read.py": "x = 1\n"})
    outcome = script.run_example(
        "read.py", script.row("--brain *.npy"), tmp_path, opengl=False
    )
    assert outcome.status == "FAIL"
    assert outcome.detail == "no file matches *.npy"


def test_last_line_skips_blank_lines() -> None:
    script = _load()
    assert script.last_line("one\ntwo\n\n  \n") == "two"
    assert not script.last_line("\n")


def test_a_reader_runs_after_its_producer_in_the_same_folder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script = _examples(
        monkeypatch,
        tmp_path,
        {
            "write.py": "open('out_7.txt', 'w').write('7')\n",
            "read.py": ("import sys\nassert open(sys.argv[1]).read() == '7'\n"),
        },
    )
    monkeypatch.setattr(
        script,
        "EXAMPLES",
        {
            "read.py": script.row("out_*.txt", after="write.py"),
            "write.py": script.row(),
        },
    )
    assert script.main([]) == 0
    assert _lines(capsys) == [["PASS", "write.py"], ["PASS", "read.py"]]


def test_a_reader_is_skipped_when_its_producer_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script = _examples(
        monkeypatch,
        tmp_path,
        {"write.py": "raise SystemExit(1)\n", "read.py": "x = 1\n"},
    )
    monkeypatch.setattr(
        script,
        "EXAMPLES",
        {"read.py": script.row(after="write.py"), "write.py": script.row()},
    )
    assert script.main([]) == 1
    out = capsys.readouterr().out
    assert "SKIP  read.py" in out
    assert "needs write.py, which did not pass" in out


def test_a_file_in_no_table_fails_and_modules_are_not_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script = _examples(
        monkeypatch,
        tmp_path,
        {
            "a/listed.py": "x = 1\n",
            "a/helper.py": "raise RuntimeError('imported only')\n",
            "a/new.py": "x = 1\n",
        },
    )
    monkeypatch.setattr(script, "EXAMPLES", {"a/listed.py": script.row()})
    monkeypatch.setattr(script, "MODULES", frozenset({"a/helper.py"}))
    assert script.unlisted() == ["a/new.py"]
    assert script.main([]) == 1
    assert _lines(capsys) == [["FAIL", "a/new.py"], ["PASS", "a/listed.py"]]


def test_named_scripts_run_with_their_producers_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script = _examples(
        monkeypatch,
        tmp_path,
        {
            "write.py": "x = 1\n",
            "read.py": "x = 1\n",
            "other.py": "x = 1\n",
            "unlisted.py": "x = 1\n",
        },
    )
    monkeypatch.setattr(
        script,
        "EXAMPLES",
        {
            "other.py": script.row(),
            "read.py": script.row(after="write.py"),
            "write.py": script.row(),
        },
    )
    assert script.chosen(["read.py"]) == ["write.py", "read.py"]
    assert script.main(["read.py", "--anything"]) == 0
    assert _lines(capsys) == [["PASS", "write.py"], ["PASS", "read.py"]]


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


def test_every_example_and_module_is_a_real_file() -> None:
    """The real tables name only files that exist under ``examples/``."""
    script = _load()
    named = [*script.EXAMPLES, *script.MODULES]
    assert all((script.EXAMPLES_FOLDER / name).is_file() for name in named)
    producers = {row.after for row in script.EXAMPLES.values() if row.after}
    assert producers <= set(script.EXAMPLES)
    assert all(not script.EXAMPLES[name].after for name in producers)


def test_a_row_can_set_its_own_timeout() -> None:
    """A slow script gets its own limit; the others keep the default."""
    script = _load()
    slow = script.row(timeout=2 * script.TIMEOUT_SECONDS)
    assert slow.timeout == 2 * script.TIMEOUT_SECONDS
    assert script.row().timeout == script.TIMEOUT_SECONDS
    assert script.EXAMPLES["re_book/3_body_brain_lr.py"].timeout > (
        script.TIMEOUT_SECONDS
    )
