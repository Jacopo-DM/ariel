"""Checks that every section of a functional notebook ends with checks.

A section is the cells under a ``##`` heading, up to the next one. A section
with code ends with plain ``assert`` lines in its last code cell, so a reader
sees what the section showed and the notebook runner fails when it stops
being true (goal notebook-folders).
"""

# Standard library
from pathlib import Path

# Third-party libraries
import pytest
from pydantic import BaseModel, ConfigDict

FUNCTIONAL = Path(__file__).resolve().parents[2] / "tests" / "functional"

# Still to fix, emptied task by task in goal notebook-folders (PLAN.md)
TO_FIX = frozenset({
    "body_phenotypes/body_density",
    "body_phenotypes/cppn_genome",
    "body_phenotypes/decoders",
    "body_phenotypes/morphology",
    "body_phenotypes/overlap_repair",
    "body_phenotypes/prebuilt_robots",
    "ec/ec_engine",
    "ec/nde",
    "ec/tree_genome",
    "external_libs",
    "simulation/cpg_initialisation",
    "simulation/cpg_initialisation_search",
    "simulation/cpg_locomotion",
    "simulation/cpg_steering",
    "simulation/sim_environments",
    "simulation/speed_tests",
    "simulation/utils",
})


class Cell(BaseModel):
    """One cell of a notebook, as far as the rule needs it."""

    # A notebook carries much more (metadata, outputs): ignored, not refused
    model_config = ConfigDict(frozen=True, extra="ignore")

    cell_type: str
    source: str | list[str]


class Notebook(BaseModel):
    """A notebook's cells, read from its JSON file."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    cells: list[Cell]


def unchecked_sections(notebook: Notebook) -> list[str]:
    """Name the sections whose last code cell has no ``assert``.

    Parameters
    ----------
    notebook : Notebook
        The notebook's cells.

    Returns
    -------
    list[str]
        The headings of those sections, in order; code above the first
        ``##`` heading belongs to no section.
    """
    sections: list[tuple[str, list[str]]] = []
    for cell in notebook.cells:
        source = "".join(cell.source)
        if cell.cell_type == "markdown":
            sections.extend(
                (line[3:].strip(), [])
                for line in source.splitlines()
                if line.startswith("## ")
            )
        elif cell.cell_type == "code" and sections:
            sections[-1][1].append(source)
    return [
        title for title, code in sections if code and "assert " not in code[-1]
    ]


def _notebooks() -> dict[str, Path]:
    return {
        path.relative_to(FUNCTIONAL).with_suffix("").as_posix(): path
        for path in sorted(FUNCTIONAL.rglob("*.ipynb"))
        if ".ipynb_checkpoints" not in path.parts
    }


def _cell(kind: str, source: str) -> Cell:
    return Cell(cell_type=kind, source=[source])


def test_a_section_needs_an_assert_in_its_last_code_cell() -> None:
    notebook = Notebook(
        cells=[
            _cell("code", "import numpy"),
            _cell("markdown", "# Title\n## Checked"),
            _cell("code", "x = 1"),
            _cell("code", "assert x == 1"),
            _cell("markdown", "## Unchecked"),
            _cell("code", "assert True"),
            _cell("code", "y = 2"),
            _cell("markdown", "## Only text"),
        ],
    )
    assert unchecked_sections(notebook) == ["Unchecked"]


@pytest.mark.parametrize("name", sorted(_notebooks()))
def test_every_section_ends_with_checks(name: str) -> None:
    notebook = Notebook.model_validate_json(_notebooks()[name].read_text())
    missing = unchecked_sections(notebook)
    if name in TO_FIX:
        assert missing, f"{name} meets the rule now: remove it from TO_FIX"
    else:
        assert not missing, f"{name}: sections without checks: {missing}"


def test_every_name_in_to_fix_is_a_notebook() -> None:
    assert set(_notebooks()) >= TO_FIX
