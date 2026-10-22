"""Test: the DSS-M15 servo record holds the datasheet values in SI units."""

import math

import astropy.units as u
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from ariel.parameters import (
    DSSM15,
    KILOGRAM_FORCE,
    ArielModulesConfig,
    ServoSettings,
)
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
    assert isinstance(ArielModulesConfig().servo, DSSM15)


# --- servo settings (goal servo-dynamics, SD1) ------------------------------ #

DAMPING_6V = STALL_TORQUE_6V / SPEED_6V  # N m s / rad
STIFFNESS_6V = STALL_TORQUE_6V / math.radians(5)  # N m / rad, 5 deg error
ROTOR_INERTIA = 0.816e-7  # kg m2, 0.816 g cm2 (maxon A-max 16, 110079)
GEAR_RATIO = 310


def test_settings_derive_the_6_volt_servo() -> None:
    """At 6 V the hinge gets the datasheet's torque line and 5 deg holding."""
    settings, servo = ServoSettings(), DSSM15()
    assert settings.voltage.value == pytest.approx(6.0)
    assert settings.stall_torque(servo).value == pytest.approx(STALL_TORQUE_6V)
    assert settings.no_load_speed(servo).value == pytest.approx(SPEED_6V)
    assert settings.damping(servo).value == pytest.approx(DAMPING_6V)
    assert settings.stiffness(servo).value == pytest.approx(STIFFNESS_6V)
    assert settings.armature(servo).value == pytest.approx(
        ROTOR_INERTIA * GEAR_RATIO**2
    )
    assert settings.damping(servo).unit == u.N * u.m * u.s / u.rad
    assert settings.stiffness(servo).unit == u.N * u.m / u.rad
    assert settings.armature(servo).unit == u.kg * u.m**2
    assert "110079" in settings.rotor_inertia_source


def test_settings_at_7_2_volts_use_that_column() -> None:
    """A 7.2 V supply reads the datasheet's 7.2 V rating."""
    settings, servo = ServoSettings(voltage=7.2 * u.V), DSSM15()
    assert settings.stall_torque(servo).value == pytest.approx(STALL_TORQUE_7V2)


def test_a_voltage_without_a_rating_is_refused() -> None:
    """The datasheet rates 6 V and 7.2 V only; 5 V has no numbers."""
    with pytest.raises(ValueError, match="no rating at 5 V"):
        ServoSettings(voltage=5 * u.V).stall_torque(DSSM15())


def test_settings_refuse_a_bare_number() -> None:
    """The holding error needs its unit, like every other quantity."""
    with pytest.raises(ValidationError, match="got no unit"):
        ServoSettings(holding_error=5 * u.dimensionless_unscaled)


def test_modules_config_holds_the_servo_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SERVO_SETTINGS__VOLTAGE switches the hinge's supply."""
    assert ArielModulesConfig().servo_settings == ServoSettings()
    monkeypatch.setenv("SERVO_SETTINGS__VOLTAGE", "7.2 V")
    assert ArielModulesConfig().servo_settings.voltage.value == pytest.approx(
        7.2
    )


@given(
    volts=st.sampled_from([6.0, 7.2]),
    degrees=st.floats(min_value=0.1, max_value=90),
)
def test_derived_values_give_back_the_stall_torque(
    volts: float, degrees: float
) -> None:
    """Both gains multiply back to the stall torque.

    Stiffness times the holding error, and damping times the no-load speed,
    for any rated voltage and any holding error.
    """
    servo = DSSM15()
    settings = ServoSettings(voltage=volts * u.V, holding_error=degrees * u.deg)
    stall = settings.stall_torque(servo).value
    holding = math.radians(degrees)
    assert settings.stiffness(servo).value * holding == pytest.approx(stall)
    assert settings.damping(servo).value * settings.no_load_speed(
        servo
    ).value == pytest.approx(stall)
