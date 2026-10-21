"""Tests for the MJCF writer and the difference list."""

# Third-party libraries

import astropy.units as u
import mujoco
import pytest

from ariel.simulators import compile_model

# Local libraries
from ariel.simulators.mujoco.mjcf import (
    build_spec,
    differences,
    generate_xml,
    summary,
)
from ariel.simulators.mujoco.seeds import Seeds
from ariel.simulators.mujoco.settings import (
    Value,
    engine_defaults,
    read_spec,
    settings,
)

GRAVITY_M_S2 = 9.80665
SETTING_COUNT = 156
LONG_SCALE_M = 0.5


def same(left: Value, right: Value) -> bool:
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(
            a == pytest.approx(b, rel=1e-5)
            for a, b in zip(left, right, strict=True)
        )
    if (
        isinstance(left, int | float)
        and isinstance(right, int | float)
        and (isinstance(left, float) or isinstance(right, float))
    ):
        return float(left) == pytest.approx(float(right), rel=1e-5)
    return left == right


def test_the_xml_loads_and_carries_the_seeds() -> None:
    seeds = Seeds()
    spec = mujoco.MjSpec.from_string(generate_xml(seeds))
    model = compile_model(spec)
    assert model.opt.timestep == pytest.approx(seeds.timestep)
    assert model.opt.gravity[2] == pytest.approx(-GRAVITY_M_S2)
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert spec.default.geom.solref[0] == pytest.approx(
        seeds.contact_time_constant
    )


def test_reading_the_loaded_xml_gives_back_every_setting() -> None:
    seeds = Seeds().replace(friction=0.7, length_scale=LONG_SCALE_M * u.m)
    loaded = read_spec(mujoco.MjSpec.from_string(generate_xml(seeds)))
    for row in settings(seeds):
        assert same(loaded[row.name], row.value), row.name


def test_build_spec_sets_enums_and_arrays() -> None:
    spec = build_spec(Seeds())
    assert spec.option.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert list(spec.option.gravity) == [0.0, 0.0, -GRAVITY_M_S2]


def test_differences_list_only_changed_values() -> None:
    changed = differences(Seeds())
    names = {row.name for row in changed}
    assert {
        "option.timestep",
        "option.gravity",
        "option.integrator",
        "compiler.degree",
        "default.geom.solref",
    } <= names
    assert "option.tolerance" not in names
    defaults = engine_defaults()
    assert all(not same(row.value, defaults[row.name]) for row in changed)


def test_an_unchanged_friction_seed_is_not_a_difference() -> None:
    assert "default.geom.friction" not in {
        row.name for row in differences(Seeds())
    }
    changed = differences(Seeds().replace(friction=0.7))
    assert "default.geom.friction" in {row.name for row in changed}


def test_summary_counts_differences_and_settings() -> None:
    seeds = Seeds()
    assert summary(seeds) == (
        f"{len(differences(seeds))} of {SETTING_COUNT} differ"
    )


def test_balanceinertia_is_written_although_to_xml_drops_it() -> None:
    seeds = Seeds()
    assert 'balanceinertia="true"' in generate_xml(seeds)
    assert "balanceinertia" not in str(mujoco.MjSpec().to_xml())
