"""Test: the robot state read from MuJoCo data."""

# Third-party libraries
import mujoco
import numpy as np
import pytest

# Local libraries
from ariel.simulation.controllers.utils.data_get import get_state_from_data

ROBOT_XML = """
<mujoco>
    <worldbody>
    <body name="core" pos="0 0 1"><freejoint/><geom size=".1"/>
        <body name="arm" pos=".2 0 0">
        <joint name="hinge" type="hinge" axis="0 0 1"/><geom size=".05"/>
        </body>
    </body>
    </worldbody>
</mujoco>"""
HINGE_ANGLE = 0.3
HALF = 0.5


@pytest.mark.parametrize(
    ("quat", "expected_imaginary"),
    [
        ((HALF, HALF, -HALF, HALF), (HALF, -HALF, HALF)),
        ((-HALF, HALF, -HALF, HALF), (-HALF, HALF, -HALF)),
    ],
)
def test_state_is_quaternion_without_w_then_joints(
    quat: tuple[float, float, float, float],
    expected_imaginary: tuple[float, float, float],
) -> None:
    """The position is left out and a negative w flips the quaternion."""
    model = mujoco.MjModel.from_xml_string(ROBOT_XML)
    data = mujoco.MjData(model)
    data.qpos[0:3] = (5.0, 6.0, 7.0)
    data.qpos[3:7] = quat
    data.qpos[7] = HINGE_ANGLE
    state = get_state_from_data(data)
    np.testing.assert_allclose(state, [*expected_imaginary, HINGE_ANGLE])
