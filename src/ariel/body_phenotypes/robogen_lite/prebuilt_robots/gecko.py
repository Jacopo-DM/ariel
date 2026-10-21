"""Prebuilt gecko robot body.

The gecko has a core, a neck, a spine and four legs (flippers). The front
two flippers have two hinges each, rotated 90 degrees to each other; the
back two are rotated 45 degrees against the body to encourage forward
movement. See ``parts`` for the table format.
"""

# Local libraries
from ariel.body_phenotypes.robogen_lite.config import ModuleFaces
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.parts import (
    brick,
    build,
    hinge,
)

GECKO = (
    hinge(1, "core", ModuleFaces.FRONT),  # neck
    brick(2, "hinge_0", ModuleFaces.FRONT),  # abdomen
    hinge(3, "brick_0", ModuleFaces.FRONT),  # spine
    brick(4, "hinge_1", ModuleFaces.FRONT),  # butt
    hinge(5, "core", ModuleFaces.LEFT, 90),  # front left leg
    hinge(15, "hinge_2", ModuleFaces.FRONT, 90),
    brick(6, "hinge_3", ModuleFaces.FRONT),
    hinge(7, "core", ModuleFaces.RIGHT, -90),  # front right leg
    hinge(17, "hinge_4", ModuleFaces.FRONT, 90),
    brick(8, "hinge_5", ModuleFaces.FRONT),
    hinge(9, "brick_1", ModuleFaces.LEFT, 45),  # back left leg
    brick(10, "hinge_6", ModuleFaces.FRONT),
    hinge(11, "brick_1", ModuleFaces.RIGHT, -45),  # back right leg
    brick(12, "hinge_7", ModuleFaces.FRONT),
)


def gecko() -> CoreModule:
    """Build the gecko robot body.

    Returns
    -------
    CoreModule
        The core module of the robot, with all other modules attached.
    """
    return build(GECKO)
