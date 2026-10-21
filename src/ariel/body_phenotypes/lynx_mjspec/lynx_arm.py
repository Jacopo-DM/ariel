"""Lynxmotion arm body built as a MuJoCo MjSpec."""

from typing import Annotated

import astropy.units as u
import mujoco
import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, Field

from ariel.parameters import (
    FrozenModel,
    kilograms,
    metres,
    quantity_of,
    radians,
)

type Length = Annotated[u.Quantity, quantity_of(u.m)]
type Angle = Annotated[u.Quantity, quantity_of(u.rad, allow_negative=True)]

# The arm's parts (not measured on a real arm; the numbers ARIEL used)
BASE_RADIUS = 8 * u.cm
BASE_HEIGHT = 10 * u.cm
BASE_MASS = 2 * u.kg
JOINT_RADIUS = 4 * u.cm
JOINT_HALF_LENGTH = 3 * u.cm
JOINT_MASS = 200 * u.g
TUBE_RADIUS = 3 * u.cm
TUBE_MASS = 100 * u.g
GRIPPER_RADIUS = 1 * u.cm
GRIPPER_HALF_LENGTH = 4 * u.cm
GRIPPER_MASS = 50 * u.g
TCP_HEIGHT = 8 * u.cm


class ArmConfig(FrozenModel):
    """Joint count, tube lengths and angles that define one arm."""

    # The defaults go through the same unit check and SI conversion
    model_config = ConfigDict(
        frozen=True, extra="forbid", validate_default=True
    )

    num_joints: int = 6
    genotype_tube: list[int] = Field(default_factory=lambda: [1, 1, 1, 1, 1])
    genotype_joints: int = 6
    tube_lengths: list[Length] = Field(default_factory=lambda: [10 * u.cm] * 5)
    rotation_angles: list[Angle] = Field(
        default_factory=lambda: [
            0 * u.rad,
            -1.57 * u.rad,
            0 * u.rad,
            0 * u.rad,
            0 * u.rad,
            0 * u.rad,
        ]
    )
    task: str = "reach"


class LynxArm:
    """
    A unified MjSpec builder for the Lynxmotion Arm.

    Builds the entire robot in a single kinematic tree.

    """

    def __init__(self, config: ArmConfig | None = None) -> None:
        """Create the arm spec from `config` (a default arm if None)."""
        self.spec = mujoco.MjSpec()
        self.spec.compiler.degree = False

        # Default configuration
        self.config = config or ArmConfig()

        self.base_height = metres(BASE_HEIGHT)
        self.joint_radius = metres(JOINT_RADIUS)
        self.tube_radius = metres(TUBE_RADIUS)

        self._build_robot()

    def _rotate_axis_z(  # ruff: ignore[no-self-use]  # keeps the interface shared with the sibling classes
        self, axis: NDArray[np.float64], yaw: float
    ) -> NDArray[np.float64]:
        """Rotate a vector around Z axis.

        Returns
        -------
        NDArray[np.float64]
            The rotated vector.
        """
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1],
        ])
        return rotation @ axis

    def _build_robot(self) -> None:
        """Construct the kinematic chain hierarchically."""
        # 1. ROOT / BASE
        base_link = self.spec.worldbody.add_body(
            name="lynx_base", pos=[0, 0, 0]
        )

        base_link.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[metres(BASE_RADIUS), self.base_height / 2, 0],
            pos=[0, 0, self.base_height / 2],
            rgba=[0.2, 0.2, 0.2, 1],
            mass=kilograms(BASE_MASS),
            group=1,
        )

        current_parent = base_link
        current_z_offset = self.base_height

        num_joints = min(self.config.num_joints, self.config.genotype_joints)

        # 2. ITERATIVE CHAIN
        for i in range(num_joints):
            link_name = f"link_{i + 1}"
            link_body = current_parent.add_body(
                name=link_name,
                pos=[0, 0, current_z_offset],
            )

            # Base alternating axes
            base_axis = (
                np.array([0, 1, 0]) if i % 2 == 0 else np.array([0, 0, 1])
            )

            # Apply yaw rotation to axis instead of body
            yaw = (
                radians(self.config.rotation_angles[i])
                if i < len(self.config.rotation_angles)
                else 0.0
            )

            joint_axis = self._rotate_axis_z(base_axis, yaw).tolist()

            # Add Joint
            joint_name = f"joint_{i + 1}"
            link_body.add_joint(
                name=joint_name,
                type=mujoco.mjtJoint.mjJNT_HINGE,
                axis=joint_axis,
                range=[-2.8, 2.8],
                limited=True,
                damping=2.0,
                frictionloss=0.01,
            )

            # Add Actuator (PD position control)
            self.spec.add_actuator(
                name=f"motor_{i + 1}",
                trntype=mujoco.mjtTrn.mjTRN_JOINT,
                target=joint_name,
                ctrlrange=[-2.8, 2.8],
                ctrllimited=True,
                dynprm=[1, 0, 0] + [0] * 7,
                gainprm=[20, 0, 0] + [0] * 7,
                biasprm=[0, -20, 0] + [0] * 7,
            )

            # Joint visual (motor housing)
            link_body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[self.joint_radius, metres(JOINT_HALF_LENGTH), 0],
                axisangle=[1, 0, 0, 1.5708],
                rgba=[0.1, 0.1, 0.1, 1],
                group=1,
                mass=kilograms(JOINT_MASS),
            )

            # Tube (if enabled)
            tube_length = 0.0
            if (
                i < len(self.config.genotype_tube)
                and self.config.genotype_tube[i] == 1
            ):
                tube_length = metres(self.config.tube_lengths[i])

                link_body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    size=[self.tube_radius, tube_length / 2, 0],
                    pos=[0, 0, tube_length / 2],
                    rgba=[0.7, 0.2, 0.2, 1],
                    group=1,
                    mass=kilograms(TUBE_MASS),
                )

            # Update chain
            current_parent = link_body
            current_z_offset = tube_length

        # 3. END EFFECTOR
        ee_body = current_parent.add_body(
            name="end_effector",
            pos=[0, 0, current_z_offset],
        )

        ee_body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[metres(GRIPPER_RADIUS), metres(GRIPPER_HALF_LENGTH), 0],
            pos=[0, 0, metres(GRIPPER_HALF_LENGTH)],
            rgba=[0.8, 0.8, 0.8, 1],
            group=1,
            mass=kilograms(GRIPPER_MASS),
        )

        ee_body.add_site(
            name="tcp",
            pos=[0, 0, metres(TCP_HEIGHT)],
            rgba=[1, 0, 0, 1],
            size=[0.01, 0.01, 0.01],
        )
