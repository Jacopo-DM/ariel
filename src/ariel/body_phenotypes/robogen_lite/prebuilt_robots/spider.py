"""Prebuilt spider robot body."""

from ariel.body_phenotypes.robogen_lite.config import ModuleFaces
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.parts import attach


def spider() -> CoreModule:
    """Spider robot body.

    Body Description
    ---------
    The spider body consists of a core module, 4 legs. Each leg constains two
    hinges,
    with blocks attached at the end of them.

    Returns
    -------
    CoreModule
        The core module of the robot, with all other modules attached.
    """
    core = CoreModule(
        index=0,
    )

    # --- LEG 1 (Front) ---
    l1_1 = HingeModule(
        index=1,
    )
    l1_1.rotate(0)  # Horizontal (Yaw)

    l1_2 = HingeModule(
        index=2,
    )
    l1_2.rotate(90)  # Vertical (Pitch)

    l1_3 = BrickModule(
        index=3,
    )

    # --- LEG 2 (Left) ---
    l2_1 = HingeModule(
        index=4,
    )
    l2_1.rotate(0)  # Horizontal (Yaw)

    l2_2 = HingeModule(
        index=5,
    )
    l2_2.rotate(90)  # Vertical (Pitch)

    l2_3 = BrickModule(
        index=6,
    )

    # --- LEG 3 (Right) ---
    l3_1 = HingeModule(
        index=7,
    )
    l3_1.rotate(0)  # Horizontal (Yaw)

    l3_2 = HingeModule(
        index=8,
    )
    l3_2.rotate(90)  # Vertical (Pitch)

    l3_3 = BrickModule(
        index=9,
    )

    # --- LEG 4 (Back) ---
    l4_1 = HingeModule(
        index=10,
    )
    l4_1.rotate(0)  # Horizontal (Yaw)

    l4_2 = HingeModule(
        index=11,
    )
    l4_2.rotate(90)  # Vertical (Pitch)

    l4_3 = BrickModule(
        index=12,
    )

    # --- ATTACH BODIES ---

    """Front-leg"""
    attach(core, core, ModuleFaces.FRONT, l1_1, "l1_1")
    attach(core, l1_1, ModuleFaces.FRONT, l1_2, "l1_2", 90)
    attach(core, l1_2, ModuleFaces.FRONT, l1_3, "l1_3")

    """Left-leg"""
    attach(core, core, ModuleFaces.LEFT, l2_1, "l2_1")
    attach(core, l2_1, ModuleFaces.FRONT, l2_2, "l2_2", 90)
    attach(core, l2_2, ModuleFaces.FRONT, l2_3, "l2_3")

    """Right-leg"""
    attach(core, core, ModuleFaces.RIGHT, l3_1, "l3_1")
    attach(core, l3_1, ModuleFaces.FRONT, l3_2, "l3_2", 90)
    attach(core, l3_2, ModuleFaces.FRONT, l3_3, "l3_3")

    """Back-leg"""
    attach(core, core, ModuleFaces.BACK, l4_1, "l4_1")
    attach(core, l4_1, ModuleFaces.FRONT, l4_2, "l4_2", 90)
    attach(core, l4_2, ModuleFaces.FRONT, l4_3, "l4_3")

    return core
