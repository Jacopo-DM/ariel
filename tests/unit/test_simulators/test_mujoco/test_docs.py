"""Tests for the generated parameter document."""

# Standard library
from pathlib import Path

# Third-party libraries
import astropy.units as u
import mujoco
import pytest

# Local libraries
from ariel.simulators.mujoco import docs
from ariel.simulators.mujoco.__main__ import main
from ariel.simulators.mujoco.docs import DOCS_PATH, render_markdown, write_docs
from ariel.simulators.mujoco.mjcf import differences
from ariel.simulators.mujoco.seeds import Seeds

LONG_SCALE_M = 0.5


def test_the_document_in_the_repo_is_current() -> None:
    """Fails when seeds, formulas, bands or the engine defaults changed.

    Fix it with ``python -m ariel.simulators.mujoco --write-docs``.
    """
    assert DOCS_PATH.read_text(encoding="utf-8") == render_markdown(Seeds())


def test_the_document_says_it_is_generated_and_names_the_engine() -> None:
    text = render_markdown(Seeds())
    assert "Do not edit by hand" in text
    assert mujoco.__version__ in text


def test_the_document_lists_every_difference_once() -> None:
    seeds = Seeds()
    text = render_markdown(seeds)
    for setting in differences(seeds):
        assert text.count(f"| `{setting.name}` |") == 1
    assert f"{len(differences(seeds))} of " in text


def test_the_document_follows_the_seeds() -> None:
    longer = Seeds().replace(length_scale=LONG_SCALE_M * u.m)
    assert render_markdown(longer) != render_markdown(Seeds())


def test_the_document_has_the_four_sections() -> None:
    text = render_markdown(Seeds())
    for heading in (
        "## Seeds and formulas",
        "## Differences from MuJoCo's defaults",
        "## Plausible bands",
    ):
        assert heading in text


def test_write_docs_writes_the_rendered_text(tmp_path: Path) -> None:
    target = tmp_path / "out" / "page.md"
    assert write_docs(target) == target
    assert target.read_text(encoding="utf-8") == render_markdown(Seeds())


def test_the_command_line_writes_the_default_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = tmp_path / "page.md"
    monkeypatch.setattr(docs, "DOCS_PATH", target)
    assert main(["--write-docs"]) == 0
    assert target.exists()
    assert str(target) in capsys.readouterr().out
