"""Test: the measured module records refuse numbers without the right unit."""

from typing import Annotated

import astropy.units as u
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from ariel.parameters import (
    ArielModulesConfig,
    BrickSpec,
    CoreSpec,
    FrozenModel,
    HingeSpec,
    quantity_of,
)

GRAM = 1e-3  # kg
MM = 1e-3  # m
UNITS = [u.g, u.kg, u.mm, u.m, u.s, u.rad, u.N, u.dimensionless_unscaled]


class _Angle(FrozenModel):
    """A record with one signed angle, for the validator's own tests."""

    angle: Annotated[u.Quantity, quantity_of(u.rad, allow_negative=True)]


def test_measured_values_are_stored_in_si() -> None:
    """The defaults are the measurements, converted to kg and m."""
    brick, core, hinge = BrickSpec(), CoreSpec(), HingeSpec()
    assert brick.mass.unit == u.kg
    assert brick.mass.value == pytest.approx(59.528 * GRAM)
    assert [side.value for side in brick.size] == pytest.approx([75 * MM] * 3)
    assert core.mass.value == pytest.approx(1386.121 * GRAM)
    assert [side.value for side in core.size] == pytest.approx([150 * MM] * 3)
    assert hinge.length.value == pytest.approx(75 * MM)
    assert hinge.axis_from_parent.value == pytest.approx(31.74 * MM)
    assert hinge.cable_mass_per_length.unit == u.kg / u.m


def test_hinge_masses_follow_the_route_to_the_core() -> None:
    """Parent side is fixed; the child side carries 7.5 cm of cable a step."""
    hinge = HingeSpec()
    assert hinge.parent_mass.value == pytest.approx(16.209 * GRAM)
    one_step = 16.488 + 8.33 / 2 + 2.21 + 57.236 + 0.11353 * 7.5
    assert hinge.child_mass(1).value == pytest.approx(one_step * GRAM)
    extra = hinge.child_mass(12).value - hinge.child_mass(1).value
    assert extra == pytest.approx(0.11353 * 7.5 * 11 * GRAM)


def test_a_hinge_cannot_sit_on_itself() -> None:
    """A route of zero steps is refused."""
    with pytest.raises(ValueError, match="at least 1 step"):
        HingeSpec().child_mass(0)


def test_a_plain_number_is_refused() -> None:
    """A mass without a unit raises before anything is built."""
    with pytest.raises(ValidationError, match="needs an astropy quantity"):
        BrickSpec.model_validate({"mass": 59.5})


def test_a_wrong_dimension_is_refused() -> None:
    """A length where a mass belongs raises."""
    with pytest.raises(ValidationError, match="needs the dimension of kg"):
        BrickSpec(mass=59.5 * u.mm)


def test_text_with_a_unit_is_read() -> None:
    """An environment value such as '0.1 kg' becomes a quantity."""
    brick = BrickSpec.model_validate({"mass": "100 g"})
    assert brick.mass.value == pytest.approx(0.1)


def test_text_without_a_unit_is_refused() -> None:
    """'0.1' alone has no unit, so it is refused."""
    with pytest.raises(ValidationError, match="got no unit"):
        BrickSpec.model_validate({"mass": "0.1"})


def test_text_that_is_not_a_quantity_is_refused() -> None:
    """Text astropy cannot read names the unit the field needs."""
    with pytest.raises(ValidationError, match="needs a number and a unit"):
        BrickSpec.model_validate({"mass": "heavy"})


def test_a_bare_number_is_not_an_angle() -> None:
    """Radians convert to plain numbers, but an angle needs its unit."""
    with pytest.raises(ValidationError, match="got no unit"):
        _Angle(angle=1.5 * u.dimensionless_unscaled)


def test_a_signed_field_takes_negative_values() -> None:
    """An angle may be negative; a mass may not."""
    assert _Angle(angle=-90 * u.deg).angle.value == pytest.approx(-1.5708, 1e-4)
    with pytest.raises(ValidationError, match="positive and finite"):
        BrickSpec(mass=-1 * u.g)
    with pytest.raises(ValidationError, match="needs a finite value"):
        _Angle(angle=float("inf") * u.rad)


def test_environment_overrides_need_a_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BRICK__MASS reaches the brick's mass, with its unit."""
    monkeypatch.setenv("BRICK__MASS", "0.1 kg")
    assert ArielModulesConfig().brick.mass.value == pytest.approx(0.1)
    monkeypatch.setenv("BRICK__MASS", "0.1")
    with pytest.raises(ValidationError, match="got no unit"):
        ArielModulesConfig()


@given(
    number=st.floats(min_value=1e-6, max_value=1e6),
    unit=st.sampled_from(UNITS),
)
def test_only_masses_are_accepted_as_masses(
    number: float, unit: u.UnitBase
) -> None:
    """Any positive number with any unit: accepted exactly when it is a mass."""
    value = number * unit
    if unit.physical_type == "mass":
        assert BrickSpec(mass=value).mass.unit == u.kg
    else:
        with pytest.raises(ValidationError, match="needs the dimension of kg"):
            BrickSpec(mass=value)
