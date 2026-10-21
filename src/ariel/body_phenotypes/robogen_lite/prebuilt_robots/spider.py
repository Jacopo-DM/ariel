"""Prebuilt spider robot body.

The spider has a core and four legs (front, left, right, back). Each leg is
a hinge that turns horizontally, a hinge that turns vertically and a brick
at the end. See ``parts`` for the table format.
"""

# Local libraries
from ariel.body_phenotypes.robogen_lite.config import ModuleFaces
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.parts import (
    brick,
    build,
    hinge,
)

SPIDER = (
    hinge(1, "core", ModuleFaces.FRONT, 0),
    hinge(2, "hinge_0", ModuleFaces.FRONT, 90),
    brick(3, "hinge_1", ModuleFaces.FRONT),
    hinge(4, "core", ModuleFaces.LEFT, 0),
    hinge(5, "hinge_2", ModuleFaces.FRONT, 90),
    brick(6, "hinge_3", ModuleFaces.FRONT),
    hinge(7, "core", ModuleFaces.RIGHT, 0),
    hinge(8, "hinge_4", ModuleFaces.FRONT, 90),
    brick(9, "hinge_5", ModuleFaces.FRONT),
    hinge(10, "core", ModuleFaces.BACK, 0),
    hinge(11, "hinge_6", ModuleFaces.FRONT, 90),
    brick(12, "hinge_7", ModuleFaces.FRONT),
)


def spider() -> CoreModule:
    """Build the spider robot body.

    Returns
    -------
    CoreModule
        The core module of the robot, with all other modules attached.
    """
    return build(SPIDER)
