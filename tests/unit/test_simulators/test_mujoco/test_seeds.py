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
DEFAULT_JOINT_RAD_S = 200.0
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
