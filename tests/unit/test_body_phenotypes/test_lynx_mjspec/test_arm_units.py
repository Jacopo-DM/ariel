"""Test: the Lynx arm takes quantities and builds the same arm as before."""

import astropy.units as u
import mujoco
import pytest
from pydantic import ValidationError

from ariel.body_phenotypes.lynx_mjspec.lynx_arm import ArmConfig, LynxArm
from ariel.simulators import compile_model

# The arm before its numbers became quantities (physical-checks PC5)
BODY_MASSES = {
    "lynx_base": 2.0,
    "link_1": 0.3,
    "link_6": 0.2,
    "end_effector": 0.05,
}
BASE_SIZE = [0.08, 0.05]
JOINT_SIZE = [0.04, 0.03]
TUBE_SIZE = [0.03, 0.05]
GRIPPER_SIZE = [0.01, 0.04]
TCP_HEIGHT = 0.08


def test_the_arm_keeps_its_masses_and_sizes() -> None:
    model = compile_model(LynxArm().spec)
    for body, mass in BODY_MASSES.items():
        assert float(model.body(body).mass[0]) == pytest.approx(mass)
    sizes = [model.geom_size[geom][:2].tolist() for geom in range(4)]
    expected = (BASE_SIZE, JOINT_SIZE, TUBE_SIZE, JOINT_SIZE)
    assert sizes == [pytest.approx(size) for size in expected]
    gripper = model.body("end_effector").geomadr[0]
    assert model.geom_size[gripper][:2].tolist() == pytest.approx(GRIPPER_SIZE)
    assert model.site("tcp").pos[2] == pytest.approx(TCP_HEIGHT)


def test_tube_lengths_take_any_length_unit() -> None:
    config = ArmConfig(tube_lengths=[20 * u.cm] * 5)
    assert config.tube_lengths[0].value == pytest.approx(0.2)
    model = compile_model(LynxArm(config).spec)
    assert model.body("link_2").pos[2] == pytest.approx(0.2)


def test_a_plain_tube_length_is_refused() -> None:
    with pytest.raises(ValidationError, match="needs an astropy quantity"):
        ArmConfig.model_validate({"tube_lengths": [0.1] * 5})


def test_an_angle_in_degrees_turns_the_joint_axis() -> None:
    """-90 degrees given in degrees is the same yaw as -pi/2 rad."""
    in_degrees = ArmConfig(rotation_angles=[-90 * u.deg])
    in_radians = ArmConfig(rotation_angles=[-1.5707963 * u.rad])
    axes = [
        compile_model(LynxArm(config).spec).jnt_axis[0].tolist()
        for config in (in_degrees, in_radians)
    ]
    assert axes[0] == pytest.approx(axes[1], abs=1e-6)


def test_a_length_is_not_an_angle() -> None:
    with pytest.raises(ValidationError, match="needs the dimension of rad"):
        ArmConfig(rotation_angles=[1 * u.m])


def test_the_compiled_arm_is_plausible(
    recwarn: pytest.WarningsRecorder,
) -> None:
    model = compile_model(LynxArm().spec)
    assert isinstance(model, mujoco.MjModel)
    assert len(recwarn) == 0
