"""Tests for the derivation table of the MuJoCo settings generator."""

# Third-party libraries
import astropy.units as u
import pytest

# Local libraries
from ariel.simulators.mujoco.seeds import Seeds
from ariel.simulators.mujoco.table import Row, derivation_table

SEED_ROWS = 10
FORMULA_ROWS = 4
LONG_SCALE_M = 0.5


def by_name(rows: list[Row]) -> dict[str, Row]:
    return {row.name: row for row in rows}


def test_table_lists_every_seed_and_every_formula() -> None:
    rows = derivation_table(Seeds())
    kinds = [row.kind for row in rows]
    assert kinds.count("seed") == SEED_ROWS
    assert kinds.count("formula") == FORMULA_ROWS


def test_row_names_are_unique_and_every_row_has_a_source() -> None:
    rows = derivation_table(Seeds())
    names = [row.name for row in rows]
    assert len(names) == len(set(names))
    assert all(row.source for row in rows)


def test_seed_rows_hold_the_canonical_values() -> None:
    rows = by_name(derivation_table(Seeds()))
    assert rows["length_scale"].value == pytest.approx(0.05)
    assert rows["length_scale"].unit == "m"
    assert not rows["friction"].unit


def test_formula_rows_hold_the_derived_values() -> None:
    seeds = Seeds()
    rows = by_name(derivation_table(seeds))
    assert rows["timestep"].value == pytest.approx(seeds.timestep)
    assert rows["timestep"].unit == "s"
    assert "advection" in rows["timestep"].source


def test_table_follows_a_changed_seed() -> None:
    longer = Seeds().replace(length_scale=LONG_SCALE_M * u.m)
    rows = by_name(derivation_table(longer))
    assert rows["contact_time_constant"].value == pytest.approx(0.14)
