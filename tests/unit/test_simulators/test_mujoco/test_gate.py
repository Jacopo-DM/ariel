"""Test: the compile gate warns about implausible models and only then."""

import mujoco
import pytest

from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.environments import SimpleFlatWorld
from ariel.simulators import CheckResult, check_model, compile_model


def _model(body: str, option: str = "") -> mujoco.MjModel:
    """Compile a small XML world without the gate.

    Returns
    -------
    mujoco.MjModel
        The compiled model.
    """
    xml = f"<mujoco>{option}<worldbody>{body}</worldbody></mujoco>"
    return compile_model(mujoco.MjSpec.from_string(xml))


BOX = (
    '<body name="box"><freejoint/>'
    '<geom name="cube" type="box" size="{s} {s} {s}" mass="{m}"/></body>'
)
HINGED = (
    '<body name="a"><freejoint/>'
    '<geom type="box" size=".05 .05 .05" mass="{m1}"/>'
    '<body name="b" pos=".2 0 0"><joint name="j" type="hinge"/>'
    '<geom type="box" size=".01 .01 .01" mass="{m2}"/></body></body>'
)


def _checks(model: mujoco.MjModel) -> list[str]:
    return [result.check for result in check_model(model)]


def test_a_plausible_model_is_silent(recwarn: pytest.WarningsRecorder) -> None:
    """A 10 cm box of 0.5 kg passes every check."""
    model = _model(BOX.format(s=0.05, m=0.5))
    assert check_model(model) == []
    assert len(recwarn) == 0


def test_a_tiny_geom_warns() -> None:
    """A half-size below a millimetre (C1)."""
    tiny = (
        '<body name="box"><freejoint/>'
        '<geom type="box" size=".05 .05 .05" mass=".5"/>'
        '<geom name="speck" type="sphere" size=".0004" mass=".001"/></body>'
    )
    with pytest.warns(UserWarning, match="geom_half_size"):
        model = _model(tiny)
    assert _checks(model) == ["geom_half_size"]


def test_a_heavy_body_warns() -> None:
    """A moving body above 20 kg (C2)."""
    with pytest.warns(UserWarning, match="body_mass"):
        _model(BOX.format(s=0.5, m=50))


def test_an_impossible_density_warns() -> None:
    """A 10 cm box of 20 kg is denser than steel (C3)."""
    with pytest.warns(UserWarning, match="density"):
        model = _model(BOX.format(s=0.05, m=19))
    (found,) = check_model(model)
    assert isinstance(found, CheckResult)
    assert (found.check, found.subject, found.unit) == (
        "density",
        "body 'box'",
        "kg / m3",
    )
    assert found.value == pytest.approx(19 / 0.001)
    assert (found.low, found.high) == (50.0, 8000.0)


def test_a_large_mass_ratio_warns() -> None:
    """Two parts joined by a hinge, one 200 times the other (C4)."""
    with pytest.warns(UserWarning, match="joint_mass_ratio"):
        _model(HINGED.format(m1=2.0, m2=0.01))


def test_a_moderate_mass_ratio_is_silent() -> None:
    """A ratio of 10 across a hinge passes."""
    assert _checks(_model(HINGED.format(m1=0.1, m2=0.01))) == []


def test_a_large_timestep_warns() -> None:
    """A timestep over half the contact time constant (C5)."""
    option = '<option timestep="0.015"/>'
    with pytest.warns(UserWarning, match="timestep_over_contact"):
        _model(BOX.format(s=0.05, m=0.5), option)


def test_other_geom_shapes_are_measured() -> None:
    """Spheres, capsules, cylinders and ellipsoids get a volume too."""
    shapes = "".join(
        f'<body name="{kind}" pos="{i} 0 0"><freejoint/>'
        f'<geom type="{kind}" size=".05 .05 .05" mass="0.2"/></body>'
        for i, kind in enumerate(("sphere", "capsule", "cylinder", "ellipsoid"))
    )
    assert _checks(_model(shapes)) == []


def test_a_mesh_body_has_no_density_check() -> None:
    """A body with a geom of unknown volume is skipped by C3."""
    mesh = '<asset><mesh name="m" vertex="0 0 0 .1 0 0 0 .1 0 0 0 .1"/></asset>'
    body = (
        '<body name="tet"><freejoint/>'
        '<geom type="mesh" mesh="m" mass="50"/></body>'
    )
    with pytest.warns(UserWarning, match="body_mass"):
        model = _model(body, mesh)
    assert _checks(model) == ["body_mass"]


def test_unnamed_elements_are_numbered() -> None:
    """A finding on an element without a name shows its id."""
    unnamed = (
        '<body><freejoint/><geom type="box" size=".05 .05 .05" mass="19"/>'
        "</body>"
    )
    with pytest.warns(UserWarning, match="body #1"):
        model = _model(unnamed)
    assert check_model(model)[0].subject == "body #1"


def test_a_world_compiles_through_the_gate(
    recwarn: pytest.WarningsRecorder,
) -> None:
    """A world with a robot compiles and nothing warns."""
    world = SimpleFlatWorld()
    world.spawn(gecko().spec)
    assert world.compile().nbody > 1
    assert len(recwarn) == 0


def test_stiffness_contacts_have_no_timestep_check() -> None:
    """A negative solref is a stiffness, not a time constant (C5 skips)."""
    body = (
        '<body name="box"><freejoint/><geom type="box" size=".05 .05 .05"'
        ' mass=".5" solref="-1000 -10"/></body>'
    )
    assert _checks(_model(body)) == []
