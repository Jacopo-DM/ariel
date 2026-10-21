"""Test: the DSS-M15 servo record holds the datasheet values in SI units."""

import math

import astropy.units as u
import pytest
from pydantic import ValidationError

from ariel.parameters import DSSM15, KILOGRAM_FORCE, ArielModulesConfig
from tests.unit.misuse import untyped

STALL_TORQUE_6V = 13.5 * 9.80665 / 100  # N m, 13.5 kgf cm
STALL_TORQUE_7V2 = 15 * 9.80665 / 100  # N m, 15 kgf cm
SPEED_6V = math.radians(60) / 0.18  # rad / s, 0.18 s per 60 deg


def test_kilogram_force_is_a_force() -> None:
    """One kgf centimetre is a torque of 0.0980665 N m."""
    torque = (1 * KILOGRAM_FORCE * u.cm).to(u.N * u.m)
    assert torque.value == pytest.approx(0.0980665)


def test_datasheet_values_are_stored_in_si() -> None:
    """Every default is converted from the datasheet unit to SI."""
    servo = DSSM15()
    assert servo.at_6_volts.stall_torque.unit == u.N * u.m
    assert servo.at_6_volts.stall_torque.value == pytest.approx(STALL_TORQUE_6V)
    assert servo.at_7_2_volts.stall_torque.value == pytest.approx(
        STALL_TORQUE_7V2
    )
    assert servo.at_6_volts.speed.value == pytest.approx(SPEED_6V)
    assert servo.angle_range.value == pytest.approx(math.pi)
    assert [side.value for side in servo.size] == pytest.approx([
        0.0545,
        0.02,
        0.0475,
    ])
    assert servo.neutral_pulse_width.value == pytest.approx(0.0015)


def test_higher_voltage_is_faster_and_stronger() -> None:
    """The 7.2 V column beats the 6 V column, as in the datasheet."""
    servo = DSSM15()
    assert servo.at_7_2_volts.speed > servo.at_6_volts.speed
    assert servo.at_7_2_volts.stall_torque > servo.at_6_volts.stall_torque
    assert servo.min_voltage < servo.at_6_volts.voltage <= servo.max_voltage


def test_a_plain_number_is_refused() -> None:
    """A number without a unit raises."""
    with pytest.raises(ValidationError, match="needs an astropy quantity"):
        untyped(DSSM15)(angle_range=180)


def test_a_wrong_dimension_is_refused() -> None:
    """A mass where a length belongs raises."""
    with pytest.raises(ValidationError, match="needs the dimension of m"):
        DSSM15(wire_length=3 * u.kg)


def test_a_mass_times_length_is_not_a_torque() -> None:
    """A mass times a length (kg cm) is refused where a torque belongs."""
    with pytest.raises(ValidationError, match="needs the dimension of"):
        DSSM15().at_6_volts.model_validate({
            **dict(DSSM15().at_6_volts),
            "stall_torque": 13.5 * u.kg * u.cm,
        })


def test_modules_config_holds_the_servo() -> None:
    """The hinge's actuator is the DSS-M15."""
    assert isinstance(ArielModulesConfig().ACTUATOR, DSSM15)
