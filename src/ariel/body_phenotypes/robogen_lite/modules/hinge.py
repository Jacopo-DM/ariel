"""Hinge module: an actuated joint of a RoboGen-lite robot body.

Todo:
----
    [ ] ".rotate" as superclass method?
"""

# Standard library
import math
from functools import cache

# Third-party libraries
import mujoco
import numpy as np

from ariel.body_phenotypes import quaternions as qnp

# Local libraries
from ariel.body_phenotypes.robogen_lite.config import ModuleFaces, ModuleType
from ariel.body_phenotypes.robogen_lite.modules.module import Module
from ariel.parameters import ArielModulesConfig, kilograms, metres, radians

# Global constants
SHRINK = 0.99

# The measured hinge (ariel.parameters.module_specs) in MuJoCo's numbers.
# The stator is the fork on the parent, from the parent face to the rotation
# axis; the rotor is the cradle holding the servo, from the axis to the child
# face. Half of each side in metres; masses in kg (the rotor's mass depends
# on the hinge's place in the body, see HingeModule).
_MODULES = ArielModulesConfig()
_HINGE = _MODULES.hinge
_HALF_WIDTH = metres(_HINGE.width) / 2
_AXIS = metres(_HINGE.axis_from_parent)
STATOR_MASS = kilograms(_HINGE.parent_mass)
STATOR_DIMENSIONS = (_HALF_WIDTH, _AXIS / 2, _HALF_WIDTH)
ROTOR_DIMENSIONS = (
    _HALF_WIDTH,
    (metres(_HINGE.length) - _AXIS) / 2,
    _HALF_WIDTH,
)

# The DSS-M15 as MuJoCo sees it (goal servo-dynamics): the servo's datasheet
# and the settings of the simulated servo, in SI numbers
_SERVO, _SETTINGS = _MODULES.servo, _MODULES.servo_settings
SERVO_HALF_TRAVEL = radians(_SERVO.angle_range) / 2
SERVO_STALL_TORQUE = float(_SETTINGS.stall_torque(_SERVO).value)  # N m
SERVO_DAMPING = float(_SETTINGS.damping(_SERVO).value)  # N m s / rad
SERVO_STIFFNESS = float(_SETTINGS.stiffness(_SERVO).value)  # N m / rad
SERVO_ARMATURE = float(_SETTINGS.armature(_SERVO).value)  # kg m2


def _rotor_inertia(mass: float) -> float:
    """Return the rotor box's inertia about the hinge axis, in kg m2.

    The axis runs along the box's z, through the middle of the face that
    meets the stator: a box's own inertia plus the parallel-axis term.

    Parameters
    ----------
    mass : float
        The rotor's mass, in kg.

    Returns
    -------
    float
        ``mass * (a^2 + 4 b^2) / 3`` for half-sizes ``a`` across and ``b``
        along the rotor.
    """
    across, along, _ = ROTOR_DIMENSIONS
    return mass * (across**2 + 4 * along**2) / 3


@cache
def _servo_damping(steps_from_core: int) -> float:
    """Return the actuator's velocity gain for a critically damped hinge.

    The hinge (rotor plus the gearbox armature) then settles on a new angle
    without ringing; the joint's own damping counts towards it.

    Parameters
    ----------
    steps_from_core : int
        Edges between the core and the hinge; they set the rotor's mass.

    Returns
    -------
    float
        ``2 sqrt(kp I) - damping``, in N m s / rad.
    """
    inertia = _rotor_inertia(_rotor_mass(steps_from_core)) + SERVO_ARMATURE
    return 2 * math.sqrt(SERVO_STIFFNESS * inertia) - SERVO_DAMPING


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

        # Create the parent spec; its angles (the joint range) are radians,
        # as the world's, or MuJoCo reads the range as degrees on attach
        spec = mujoco.MjSpec()
        spec.compiler.degree = False

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
        # The DSS-M15 (goal servo-dynamics): a position servo whose torque
        # is capped at the stall torque; the joint's damping gives the
        # motor's torque-speed line, its armature the gearbox inertia
        servo_axis = (0, 0, 1)
        servo_name = "servo"
        rotor.add_joint(
            name=servo_name,
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=servo_axis,
            pos=[0, -ROTOR_DIMENSIONS[1], 0],
            range=(-SERVO_HALF_TRAVEL, SERVO_HALF_TRAVEL),
            limited=mujoco.mjtLimited.mjLIMITED_TRUE,
            damping=SERVO_DAMPING,
            armature=SERVO_ARMATURE,
        )

        # Actuator parameters are defined over MuJoCo's fixed lengths...
        dynprm = np.zeros(mujoco.mjNDYN)
        gainprm = np.zeros(mujoco.mjNGAIN)
        biasprm = np.zeros(mujoco.mjNBIAS)

        # ... but only a few are used: torque = kp (target - angle) - kv speed
        gainprm[0] = SERVO_STIFFNESS
        biasprm[1] = -SERVO_STIFFNESS
        biasprm[2] = -_servo_damping(steps_from_core)

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
            ctrlrange=(-SERVO_HALF_TRAVEL, SERVO_HALF_TRAVEL),
            forcerange=(-SERVO_STALL_TORQUE, SERVO_STALL_TORQUE),
            forcelimited=mujoco.mjtLimited.mjLIMITED_TRUE,
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
