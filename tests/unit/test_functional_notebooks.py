"""Checks that the coverage table of the functional notebooks is complete."""

# Standard library
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "src" / "ariel"
FUNCTIONAL = REPO_ROOT / "tests" / "functional"
NO_NOTEBOOK = "none yet"
NOT_AN_AREA = "(third-party libraries, not an ARIEL area)"
HEADER = {"Area", "---"}
COLUMNS = 2


def _areas() -> set[str]:
    """Collect every package of ``src/ariel`` and each sub-package below it.

    Returns
    -------
    set[str]
        Names like ``ec`` and ``ec/generators``.
    """
    top = {init.parent.name for init in SOURCE.glob("*/__init__.py")}
    sub = {
        f"{init.parent.parent.name}/{init.parent.name}"
        for init in SOURCE.glob("*/*/__init__.py")
    }
    return top | sub


def _rows() -> dict[str, str]:
    """Read the coverage table of ``tests/functional/README.md``.

    Returns
    -------
    dict[str, str]
        Each area mapped to the text of its notebook cell.
    """
    rows: dict[str, str] = {}
    for line in (FUNCTIONAL / "README.md").read_text().splitlines():
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if line.startswith("|") and len(cells) == COLUMNS:
            rows[cells[0].strip("`")] = cells[1]
    return {k: v for k, v in rows.items() if k not in HEADER}


def _notebooks(cell: str) -> list[str]:
    return re.findall(r"`([^`]+)`", cell)


def test_every_area_has_a_row() -> None:
    missing = _areas() - set(_rows())
    assert not missing, f"areas without a row in the README: {sorted(missing)}"


def test_every_row_names_an_area_and_an_existing_notebook() -> None:
    areas = _areas() | {NOT_AN_AREA}
    for area, cell in _rows().items():
        assert area in areas, f"row for an unknown area: {area}"
        for name in _notebooks(cell):
            assert (FUNCTIONAL / f"{name}.ipynb").is_file(), (
                f"row {area} names a missing notebook: {name}"
            )


def test_a_row_names_notebooks_or_says_none_yet() -> None:
    for area, cell in _rows().items():
        assert _notebooks(cell) or cell == NO_NOTEBOOK, f"row {area}: {cell}"


def test_every_notebook_has_a_row() -> None:
    named = {name for cell in _rows().values() for name in _notebooks(cell)}
    notebooks = {path.stem for path in FUNCTIONAL.glob("*.ipynb")}
    assert notebooks <= named, f"notebooks without a row: {notebooks - named}"
