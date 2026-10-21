"""Test: where saved files go."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from ariel.rendering import file_ops

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

_gl_probe_worker = importlib.import_module("ariel.opengl._gl_probe_worker")


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
