"""Hinge module: an actuated joint of a RoboGen-lite robot body.

Todo:
----
    [ ] ".rotate" as superclass method?
"""

# Standard library
from functools import cache

# Third-party libraries
import mujoco
import numpy as np

from ariel.body_phenotypes import quaternions as qnp

# Local libraries
from ariel.body_phenotypes.robogen_lite.config import ModuleFaces, ModuleType
from ariel.body_phenotypes.robogen_lite.modules.module import Module
from ariel.parameters import ArielModulesConfig, kilograms, metres

# Global constants
SHRINK = 0.99

# The measured hinge (ariel.parameters.module_specs) in MuJoCo's numbers.
# The stator is the fork on the parent, from the parent face to the rotation
# axis; the rotor is the cradle holding the servo, from the axis to the child
# face. Half of each side in metres; masses in kg (the rotor's mass depends
# on the hinge's place in the body, see HingeModule).
_HINGE = ArielModulesConfig().hinge
_HALF_WIDTH = metres(_HINGE.width) / 2
_AXIS = metres(_HINGE.axis_from_parent)
STATOR_MASS = kilograms(_HINGE.parent_mass)
STATOR_DIMENSIONS = (_HALF_WIDTH, _AXIS / 2, _HALF_WIDTH)
ROTOR_DIMENSIONS = (
    _HALF_WIDTH,
    (metres(_HINGE.length) - _AXIS) / 2,
    _HALF_WIDTH,
)


@cache
def _rotor_mass(steps_from_core: int) -> float:
    """Return the rotor's mass in kg; cached, astropy arithmetic is slow.

    Parameters
    ----------
    steps_from_core : int
        Edges between the core and the hinge in the body's tree.

    Returns
    -------
    float
        The mass of the cradle, the servo and its cable, in kg.
    """
    return kilograms(_HINGE.child_mass(steps_from_core))


class HingeModule(Module):
    """Hinge module specifications."""

    index: int | None = None
    module_type: ModuleType = ModuleType.HINGE
    body: mujoco.MjsBody

    def __init__(  # ruff: ignore[too-many-locals]  # long routine kept as written; splitting it is a refactor outside this goal
        self, index: int, steps_from_core: int = 1
    ) -> None:
        """Initialize the hinge module.

        Parameters
        ----------
        index
            The index of the hinge module being instantiated
        steps_from_core
            Edges between the core and this hinge in the body's tree; the
            servo's cable runs back to the core, so it sets the rotor's mass
        """
        rotor_mass = _rotor_mass(steps_from_core)
        # Set the index of the module
        self.index = index

        # Create the parent spec.
        spec = mujoco.MjSpec()

        # ========= Hinge =========
        hinge_name = self.module_type.name.lower()
        hinge = spec.worldbody.add_body(
            name=hinge_name,
        )

        # ========= Stator =========
        stator_name = "stator"
        stator = hinge.add_body(
            name=stator_name,
            pos=[0, STATOR_DIMENSIONS[1], 0],
        )
        stator.add_geom(
            name=stator_name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            mass=STATOR_MASS,
            size=np.array(STATOR_DIMENSIONS) * SHRINK,  # z-fighting
            rgba=(223 / 255, 41 / 255, 53 / 255, 1),
        )

        # ========= Rotor =========
        rotor_name = "rotor"
        rotor = hinge.add_body(
            name=rotor_name,
            pos=[0, STATOR_DIMENSIONS[1] * 2 + ROTOR_DIMENSIONS[1], 0],
        )
        rotor.add_geom(
            name=rotor_name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            mass=rotor_mass,
            size=ROTOR_DIMENSIONS,
            rgba=(160 / 255, 24 / 255, 33 / 255, 1),
        )

        # ======== Attachment Points =========
        self.sites = {}
        self.sites[ModuleFaces.FRONT] = rotor.add_site(
            name=f"{hinge_name}-front",
            pos=[0, ROTOR_DIMENSIONS[1], 0],
        )

        # ========= Servo =========
        # Robot actuators
        kp = 1
        kv = 1  # critically damp oscillator
        servo_axis = (0, 0, 1)

        servo_name = "servo"
        rotor.add_joint(
            name=servo_name,
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=servo_axis,
            pos=[0, -ROTOR_DIMENSIONS[1], 0],
        )

        # Actuator parameters are defined over a range of 10...
        dynprm = np.zeros(10)
        gainprm = np.zeros(10)
        biasprm = np.zeros(10)

        # ... but only a few of the parameters are actually used
        gainprm[0] = kp
        biasprm[:3] = [0, -kp, -kv]

        # Contact exclusion
        spec.add_exclude(
            bodyname1=stator_name,
            bodyname2=rotor_name,
        )

        # --- Actuator(s) --- #
        dyntype = mujoco.mjtDyn.mjDYN_NONE
        gaintype = mujoco.mjtGain.mjGAIN_FIXED
        biastype = mujoco.mjtBias.mjBIAS_AFFINE
        trntype = mujoco.mjtTrn.mjTRN_JOINT
        spec.add_actuator(
            name=servo_name,
            dyntype=dyntype,
            gaintype=gaintype,
            biastype=biastype,
            dynprm=dynprm.tolist(),
            gainprm=gainprm.tolist(),
            biasprm=biasprm.tolist(),
            trntype=trntype,
            target=servo_name,
            ctrlrange=(
                -np.pi / 2,
                np.pi / 2,
            ),  # [-90, 90] degrees (range of 180)
        )

        # Save model specifications
        self.spec = spec
        self.body = hinge
        self.rotate(angle=0)  # Initialize with no rotation

    def rotate(
        self,
        angle: float,
    ) -> None:
        """
        Rotate the hinge module by a specified angle.

        Parameters
        ----------
        angle
            The angle in degrees to rotate the hinge.
        """
        # Convert angle to quaternion
        quat = qnp.from_euler_angles([
            np.deg2rad(180),
            -np.deg2rad(180 - angle),
            np.deg2rad(0),
        ])

        # Set the quaternion for the hinge body
        self.body.quat = np.round(quat, decimals=3)
