"""Tests for the plausible bands of the MuJoCo settings generator."""

# Standard library
import warnings

# Third-party libraries
import astropy.units as u
import pytest

# Local libraries
from ariel.simulators.mujoco.bands import (
    BANDS,
    BandResult,
    evaluate_bands,
    summary,
    warn_outside,
)
from ariel.simulators.mujoco.seeds import Seeds

BAND_COUNT = 9
HIGH_FRICTION = 3.0
VERY_FAST_M_S = 100.0


def by_name(results: list[BandResult]) -> dict[str, BandResult]:
    return {result.name: result for result in results}


def test_every_band_has_a_basis_and_a_valid_range() -> None:
    assert len(BANDS) == BAND_COUNT
    for band in BANDS:
        assert band.basis
        assert band.low < band.high


def test_default_seeds_are_inside_every_band() -> None:
    results = evaluate_bands(Seeds())
    assert len(results) == BAND_COUNT
    assert all(result.inside for result in results)


def test_a_value_outside_its_band_is_reported() -> None:
    results = by_name(evaluate_bands(Seeds().replace(friction=HIGH_FRICTION)))
    assert not results["friction"].inside
    assert results["friction"].value == pytest.approx(HIGH_FRICTION)
    assert results["gravity"].inside


def test_a_derived_value_can_leave_its_band() -> None:
    fast = Seeds().replace(top_speed=VERY_FAST_M_S * u.m / u.s)
    results = by_name(evaluate_bands(fast))
    assert not results["timestep"].inside
    assert results["timestep"].value == pytest.approx(fast.timestep)


def test_band_edges_count_as_inside() -> None:
    edge = Seeds().replace(friction=0.2)
    assert by_name(evaluate_bands(edge))["friction"].inside


def test_gravity_is_checked_as_a_magnitude() -> None:
    results = by_name(evaluate_bands(Seeds()))
    assert results["gravity"].value == pytest.approx(9.80665)


def test_warn_outside_warns_once_per_value_and_counts() -> None:
    results = evaluate_bands(Seeds().replace(friction=HIGH_FRICTION))
    with pytest.warns(UserWarning, match="friction") as caught:
        count = warn_outside(results)
    assert count == 1
    assert len(caught) == 1


def test_warn_outside_is_silent_when_everything_is_inside() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert warn_outside(evaluate_bands(Seeds())) == 0


def test_summary_counts_the_checked_values() -> None:
    assert summary(evaluate_bands(Seeds())) == (
        f"0 of {BAND_COUNT} values outside their plausible band"
    )
    worse = evaluate_bands(Seeds().replace(friction=HIGH_FRICTION))
    assert summary(worse).startswith("1 of")
