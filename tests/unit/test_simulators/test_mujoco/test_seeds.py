"""Tests for the physical-scale seeds of the MuJoCo settings generator."""

# Third-party libraries
import astropy.units as u
import numpy as np
import pytest
from pydantic import ValidationError

# Local libraries
from ariel.simulators.mujoco.seeds import Seeds

DEFAULT_LENGTH_M = 0.05
DEFAULT_SPEED_M_S = 1.0
DEFAULT_JOINT_RAD_S = 100.0
DEFAULT_ALPHA = 0.05
DEFAULT_KAPPA = 0.28
DEFAULT_GAMMA = 0.5
GRAVITY_M_S2 = 9.80665
FIVE_CENTIMETRES = 5.0
LONGER_LENGTH_M = 0.1


def test_defaults_are_the_documented_seeds() -> None:
    seeds = Seeds()
    assert seeds.length_scale == DEFAULT_LENGTH_M * u.m
    assert seeds.top_speed == DEFAULT_SPEED_M_S * u.m / u.s
    assert seeds.joint_frequency == DEFAULT_JOINT_RAD_S * u.rad / u.s
    assert seeds.gravity.value == pytest.approx(GRAVITY_M_S2)
    assert seeds.alpha == DEFAULT_ALPHA
    assert seeds.kappa == DEFAULT_KAPPA
    assert seeds.gamma == DEFAULT_GAMMA


def test_quantities_are_stored_in_the_canonical_unit() -> None:
    seeds = Seeds(length_scale=FIVE_CENTIMETRES * u.cm)
    assert seeds.length_scale.unit == u.m
    assert seeds.length_scale.value == pytest.approx(DEFAULT_LENGTH_M)


def test_inverse_seconds_count_as_radians_per_second() -> None:
    seeds = Seeds(joint_frequency=DEFAULT_JOINT_RAD_S / u.s)
    assert seeds.joint_frequency.unit == u.rad / u.s
    assert seeds.joint_frequency.value == pytest.approx(DEFAULT_JOINT_RAD_S)


def test_a_wrong_dimension_raises() -> None:
    with pytest.raises(ValidationError, match="length_scale"):
        Seeds(length_scale=1.0 * u.s)


def test_a_number_without_a_unit_raises() -> None:
    with pytest.raises(ValidationError, match="astropy quantity"):
        Seeds(length_scale=DEFAULT_LENGTH_M)  # type: ignore[arg-type]  # the test passes the wrong type on purpose


@pytest.mark.parametrize(
    "bad",
    [0.0 * u.m, -1.0 * u.m, float("nan") * u.m, float("inf") * u.m],
)
def test_a_quantity_must_be_positive_and_finite(bad: u.Quantity) -> None:
    with pytest.raises(ValidationError, match="positive and finite"):
        Seeds(length_scale=bad)


def test_a_quantity_must_be_a_scalar() -> None:
    with pytest.raises(ValidationError, match="scalar"):
        Seeds(length_scale=np.array([1.0, 2.0]) * u.m)


@pytest.mark.parametrize("name", ["alpha", "kappa", "gamma", "friction"])
@pytest.mark.parametrize("bad", [0.0, -0.1, float("nan")])
def test_plain_seeds_must_be_positive(name: str, bad: float) -> None:
    with pytest.raises(ValidationError, match=name):
        Seeds.model_validate({name: bad})


def test_seeds_are_frozen() -> None:
    seeds = Seeds()
    with pytest.raises(ValidationError, match="frozen"):
        seeds.alpha = 0.1  # type: ignore[misc]  # the test assigns to a frozen field on purpose


def test_replace_returns_a_new_object_and_keeps_the_old_one() -> None:
    seeds = Seeds()
    longer = seeds.replace(length_scale=LONGER_LENGTH_M * u.m)
    assert longer.length_scale == LONGER_LENGTH_M * u.m
    assert seeds.length_scale == DEFAULT_LENGTH_M * u.m
    assert longer.top_speed == seeds.top_speed


def test_replace_validates_the_change() -> None:
    with pytest.raises(ValidationError, match="length_scale"):
        Seeds().replace(length_scale=1.0 * u.s)


def test_replace_rejects_an_unknown_seed() -> None:
    with pytest.raises(ValidationError, match="extra"):
        Seeds().replace(mass=1.0 * u.kg)


ADVECTION_S = 0.0025
CONTACT_S = 0.014
JOINT_S = 0.005
LONG_SCALE_M = 0.5
FAST_M_S = 10.0
FAST_ADVECTION_S = 0.00025
STIFF_RAD_S = 1000.0
STIFF_JOINT_S = 0.5 / STIFF_RAD_S
SLOW_JOINT_RAD_S = 50.0
LARGE_ALPHA = 10.0


def test_derived_times_follow_the_chain() -> None:
    seeds = Seeds()
    assert seeds.advection_limit == pytest.approx(ADVECTION_S)
    assert seeds.contact_time_constant == pytest.approx(CONTACT_S)
    assert seeds.joint_limit == pytest.approx(JOINT_S)
    assert seeds.timestep == pytest.approx(ADVECTION_S)


def test_the_timestep_is_the_smallest_limit() -> None:
    fast = Seeds().replace(top_speed=FAST_M_S * u.m / u.s)
    assert fast.timestep == pytest.approx(FAST_ADVECTION_S)
    assert fast.timestep_limit == "advection"
    stiff = Seeds().replace(joint_frequency=STIFF_RAD_S * u.rad / u.s)
    assert stiff.timestep == pytest.approx(STIFF_JOINT_S)
    assert stiff.timestep_limit == "joint"


def test_the_contact_limit_can_bind() -> None:
    seeds = Seeds().replace(
        alpha=LARGE_ALPHA, joint_frequency=SLOW_JOINT_RAD_S * u.rad / u.s
    )
    assert seeds.timestep == pytest.approx(CONTACT_S / 2)
    assert seeds.timestep_limit == "contact"


def test_replace_recomputes_the_derived_values() -> None:
    longer = Seeds().replace(length_scale=LONG_SCALE_M * u.m)
    assert longer.advection_limit == pytest.approx(ADVECTION_S * 10)
    assert longer.contact_time_constant == pytest.approx(CONTACT_S * 10)


def test_the_contact_time_constant_respects_the_documented_floor() -> None:
    seeds = Seeds()
    assert seeds.contact_time_constant >= 2 * seeds.timestep


def test_derived_values_are_part_of_the_dump() -> None:
    dumped = Seeds().model_dump()
    assert dumped["timestep_limit"] == "advection"
    assert dumped["timestep"] == pytest.approx(ADVECTION_S)
