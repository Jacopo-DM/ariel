"""Prebuilt gecko robot body."""

from pathlib import Path

from ariel.body_phenotypes.robogen_lite.config import ModuleFaces
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.parts import attach


def gecko() -> CoreModule:
    """Gecko robot body.

    Create and attach bodies/sites, then print relative orientations between
    abdomen and spine bodies to debug hinge placement/orientation.

    Body Description
    ---------
    The gecko body consists of a core module, 4 legs (flippers), a neck and a
    spine. For better mobility the front two flippers have 2 hinges (joints),
    each rotated 90 degrees to each other. Additionally, the back two flippers
    are rotated 45 degrees compared to the body, to encourage forward movement.

    Returns
    -------
    CoreModule
        The core module of the robot, with all other modules attached.
    """
    core = CoreModule(
        index=0,
    )
    neck = HingeModule(
        index=1,
    )

    abdomen = BrickModule(
        index=2,
    )
    spine = HingeModule(
        index=3,
    )

    butt = BrickModule(
        index=4,
    )

    fl_leg = HingeModule(
        index=5,
    )
    fl_leg.rotate(90)
    fl_leg2 = HingeModule(
        index=15,
    )
    fl_leg2.rotate(90)

    fl_flipper = BrickModule(
        index=6,
    )

    fr_leg = HingeModule(
        index=7,
    )
    fr_leg.rotate(-90)
    fr_leg2 = HingeModule(
        index=17,
    )
    fr_leg2.rotate(90)

    fr_flipper = BrickModule(
        index=8,
    )
    bl_leg = HingeModule(
        index=9,
    )
    bl_leg.rotate(45)
    bl_flipper = BrickModule(
        index=10,
    )
    br_leg = HingeModule(
        index=11,
    )
    br_leg.rotate(-45)
    br_flipper = BrickModule(
        index=12,
    )

    # Attach bodies
    attach(core, core, ModuleFaces.FRONT, neck, "neck")

    attach(core, neck, ModuleFaces.FRONT, abdomen, "abdomen")
    attach(core, abdomen, ModuleFaces.FRONT, spine, "spine")
    attach(core, spine, ModuleFaces.FRONT, butt, "butt")
    attach(core, core, ModuleFaces.LEFT, fl_leg, "fl_leg", 90)

    attach(core, fl_leg, ModuleFaces.FRONT, fl_leg2, "fl_flipper", 90)
    attach(core, fl_leg2, ModuleFaces.FRONT, fl_flipper, "fl_flipper2")

    attach(core, core, ModuleFaces.RIGHT, fr_leg, "fr_leg", -90)

    attach(core, fr_leg, ModuleFaces.FRONT, fr_leg2, "fr_flipper", 90)
    attach(core, fr_leg2, ModuleFaces.FRONT, fr_flipper, "fr_flipper2")

    attach(core, butt, ModuleFaces.LEFT, bl_leg, "bl_leg", 45)
    attach(core, bl_leg, ModuleFaces.FRONT, bl_flipper, "bl_flipper")
    attach(core, butt, ModuleFaces.RIGHT, br_leg, "br_leg", -45)
    attach(core, br_leg, ModuleFaces.FRONT, br_flipper, "br_flipper")
    return core


if __name__ == "__main__":
    gecko_rob = gecko()
    gecko_xml = gecko_rob.spec.to_xml()
    Path(
        "ariel/src/ariel/body_phenotypes/robogen_lite/prebuilt_robots/gecko.xml",
    ).write_text(gecko_xml, encoding="utf-8")
