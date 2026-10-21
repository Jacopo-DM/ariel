"""Brick module: a rigid block of a RoboGen-lite robot body.

Todo:
----
    [ ] ".rotate" as superclass method?
"""

# Third-party libraries
import mujoco
import numpy as np

from ariel.body_phenotypes import quaternions as qnp

# Local libraries
from ariel.body_phenotypes.robogen_lite.config import ModuleFaces, ModuleType
from ariel.body_phenotypes.robogen_lite.modules.module import Module

# --- Robogen Configuration --- #
# Module weight (kg)
BRICK_MASS = 0.055

# Module half-sizes (length, width, height) in meters
BRICK_DIMENSIONS = (0.05, 0.05, 0.05)
# ------------------------------ #


class BrickModule(Module):
    """Brick module specifications."""

    index: int | None = None
    module_type: ModuleType = ModuleType.BRICK
    body: mujoco.MjsBody

    def __init__(self, index: int) -> None:
        """Initialize the brick module.

        Parameters
        ----------
        index
            The index of the brick module being instantiated
        """
        # Set the index of the module
        self.index = index

        # Create the parent spec.
        spec = mujoco.MjSpec()

        # ========= BRICK =========
        brick_name = self.module_type.name.lower()
        brick = spec.worldbody.add_body(
            name=brick_name,
        )
        brick.add_geom(
            name=brick_name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            mass=BRICK_MASS,
            size=BRICK_DIMENSIONS,
            pos=[0, BRICK_DIMENSIONS[0], 0],
            rgba=(28 / 255, 119 / 255, 195 / 255, 1),
        )

        # ========= Attachment Points =========
        self.sites = {}
        self.sites[ModuleFaces.FRONT] = brick.add_site(
            name=f"{brick_name}-front",
            pos=[0, BRICK_DIMENSIONS[1] * 2, 0],
            quat=np.round(
                qnp.from_euler_angles([
                    np.deg2rad(0),
                    np.deg2rad(180),
                    np.deg2rad(180),
                ]),
                decimals=3,
            ).tolist(),
        )
        self.sites[ModuleFaces.LEFT] = brick.add_site(
            name=f"{brick_name}-left",
            pos=[
                -BRICK_DIMENSIONS[0],
                BRICK_DIMENSIONS[1],
                0,
            ],
            quat=np.round(
                qnp.from_euler_angles([
                    np.deg2rad(90),
                    -np.deg2rad(90),
                    -np.deg2rad(90),
                ]),
                decimals=3,
            ).tolist(),
        )
        self.sites[ModuleFaces.RIGHT] = brick.add_site(
            name=f"{brick_name}-right",
            pos=[
                BRICK_DIMENSIONS[0],
                BRICK_DIMENSIONS[1],
                0,
            ],
            quat=np.round(
                qnp.from_euler_angles([
                    np.deg2rad(90),
                    np.deg2rad(90),
                    -np.deg2rad(90),
                ]),
                decimals=3,
            ).tolist(),
        )
        self.sites[ModuleFaces.TOP] = brick.add_site(
            name=f"{brick_name}-top",
            pos=[
                0,
                BRICK_DIMENSIONS[1],
                BRICK_DIMENSIONS[2],
            ],
            quat=np.round(
                qnp.from_euler_angles([
                    np.deg2rad(0),
                    np.deg2rad(180),
                    np.deg2rad(90),
                ]),
                decimals=3,
            ).tolist(),
        )
        self.sites[ModuleFaces.BOTTOM] = brick.add_site(
            name=f"{brick_name}-bottom",
            pos=[
                0,
                BRICK_DIMENSIONS[1],
                -BRICK_DIMENSIONS[2],
            ],
            quat=np.round(
                qnp.from_euler_angles([
                    np.deg2rad(0),
                    np.deg2rad(0),
                    -np.deg2rad(90),
                ]),
                decimals=3,
            ).tolist(),
        )

        # Save model specifications
        self.spec = spec
        self.body = brick
        self.rotate(angle=0)  # Initialize with no rotation

    def rotate(
        self,
        angle: float,
    ) -> None:
        """
        Rotate the brick module by a specified angle.

        Parameters
        ----------
        angle : float
            The angle in degrees to rotate the brick.
        """
        # Convert angle to quaternion
        quat = qnp.from_euler_angles([
            np.deg2rad(180),
            -np.deg2rad(180 - angle),
            np.deg2rad(0),
        ])

        # Set the quaternion for the brick body
        self.body.quat = np.round(quat, decimals=3)
