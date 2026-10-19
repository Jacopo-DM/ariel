"""Tests for the MuJoCo settings the generator holds."""

# Third-party libraries
import astropy.units as u
import mujoco
import pytest

# Local libraries
from ariel.simulators.mujoco.seeds import Seeds
from ariel.simulators.mujoco.settings import (
    SECTIONS,
    Setting,
    engine_defaults,
    read_object,
    settings,
)

ENGINE_TIMESTEP = 0.002
ENGINE_OFFWIDTH = 640
GRAVITY_M_S2 = 9.80665
LONG_SCALE_M = 0.5
DEFAULT_SOLREF_DAMPING = 1.0
OFFSCREEN_HEIGHT = 960
OFFSCREEN_WIDTH = 1280


def by_name(rows: list[Setting]) -> dict[str, Setting]:
    return {row.name: row for row in rows}


def test_engine_defaults_read_the_installed_engine() -> None:
    found = engine_defaults()
    assert found["option.timestep"] == pytest.approx(ENGINE_TIMESTEP)
    assert found["option.gravity"] == (0.0, 0.0, -9.81)
    assert found["compiler.degree"] is True
    assert found["size.nconmax"] == -1
    assert found["visual.global_.offwidth"] == ENGINE_OFFWIDTH
    assert found["default.geom.solref"] == (0.02, 1.0)


def test_engine_defaults_cover_the_five_sections_only() -> None:
    found = engine_defaults()
    assert {name.split(".")[0] for name in found} == set(SECTIONS)
    assert not any(".global." in name for name in found)


def test_engine_defaults_hold_plain_numbers_only() -> None:
    for value in engine_defaults().values():
        assert isinstance(value, bool | int | float | str | tuple)
        if isinstance(value, tuple):
            assert all(isinstance(item, float) for item in value)


def test_engine_enums_are_stored_as_integers() -> None:
    found = engine_defaults()
    assert found["option.integrator"] == int(mujoco.mjtIntegrator.mjINT_EULER)


def test_settings_have_the_names_of_the_engine_defaults() -> None:
    rows = settings(Seeds())
    assert {row.name for row in rows} == set(engine_defaults())
    assert len({row.name for row in rows}) == len(rows)


def test_every_setting_has_a_reason_and_a_source() -> None:
    for row in settings(Seeds()):
        assert row.reason
        assert row.source


def test_untouched_settings_keep_the_engine_value() -> None:
    defaults = engine_defaults()
    kept = [row for row in settings(Seeds()) if row.kind == "fixed"]
    assert kept
    changed = {
        "option.integrator",
        "compiler.balanceinertia",
        "compiler.degree",
        "visual.global_.offheight",
        "visual.global_.offwidth",
    }
    for row in kept:
        if row.name not in changed:
            assert row.value == defaults[row.name]


def test_physical_settings_come_from_the_seeds() -> None:
    seeds = Seeds()
    rows = by_name(settings(seeds))
    assert rows["option.timestep"].value == pytest.approx(seeds.timestep)
    assert rows["option.timestep"].kind == "formula"
    assert rows["option.gravity"].value == (0.0, 0.0, -GRAVITY_M_S2)
    assert rows["option.gravity"].kind == "seed"
    assert rows["option.density"].value == pytest.approx(
        seeds.air_density.value
    )
    assert rows["option.viscosity"].value == pytest.approx(
        seeds.air_viscosity.value
    )


def test_contact_softness_follows_the_time_constant() -> None:
    seeds = Seeds()
    rows = by_name(settings(seeds))
    solref = rows["default.geom.solref"].value
    assert isinstance(solref, tuple)
    assert solref[0] == pytest.approx(seeds.contact_time_constant)
    assert solref[1] == DEFAULT_SOLREF_DAMPING
    assert rows["default.geom.solref"].kind == "formula"


def test_friction_seed_sets_the_sliding_term_only() -> None:
    seeds = Seeds().replace(friction=0.7)
    rows = by_name(settings(seeds))
    friction = rows["default.geom.friction"].value
    assert isinstance(friction, tuple)
    assert friction[0] == pytest.approx(0.7)
    engine = engine_defaults()["default.geom.friction"]
    assert isinstance(engine, tuple)
    assert friction[1:] == engine[1:]


def test_choices_kept_from_ariel_are_fixed_with_a_reason() -> None:
    rows = by_name(settings(Seeds()))
    integrator = rows["option.integrator"]
    assert integrator.kind == "fixed"
    assert integrator.value == int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
    assert "implicitfast" in integrator.reason
    assert rows["compiler.balanceinertia"].value is True
    assert rows["compiler.degree"].value is False
    assert rows["visual.global_.offheight"].value == OFFSCREEN_HEIGHT
    assert rows["visual.global_.offwidth"].value == OFFSCREEN_WIDTH


def test_a_changed_seed_changes_its_settings() -> None:
    longer = Seeds().replace(length_scale=LONG_SCALE_M * u.m)
    rows = by_name(settings(longer))
    assert rows["option.timestep"].value == pytest.approx(longer.timestep)
    solref = rows["default.geom.solref"].value
    assert isinstance(solref, tuple)
    assert solref[0] == pytest.approx(longer.contact_time_constant)


class _Inner:
    __module__ = "mujoco.fake"
    number = 3


class _Outer:
    visible = 1.5
    _private = 2.0
    global_ = 4
    eulerseq = "xyz"
    inner = _Inner()
    stranger = object()

    def method(self) -> None:
        """Do nothing; methods are not settings."""


def test_read_object_keeps_plain_values_and_engine_sub_objects_only() -> None:
    assert read_object(_Outer(), "thing") == {
        "thing.visible": 1.5,
        "thing.global_": 4,
        "thing.inner.number": 3,
    }


def test_read_object_can_be_limited_to_some_names() -> None:
    assert read_object(_Outer(), "thing", ("visible", "_private")) == {
        "thing.visible": 1.5,
    }
