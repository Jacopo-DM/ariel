"""Spider robot body variants built from core, hinge and brick modules."""

import math
import sys

from ariel.body_phenotypes.robogen_lite.config import ModuleFaces
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.parts import (
    attach as attach_part,
)

current_module = sys.modules[__name__]

SUBMODULES = HingeModule | BrickModule
MODULES = CoreModule | SUBMODULES

faces = [
    ModuleFaces.FRONT,
    ModuleFaces.LEFT,
    ModuleFaces.BACK,
    ModuleFaces.RIGHT,
]
F, L, B, R = faces


def make_core() -> CoreModule:
    """Return a new core module named "C".

    Returns
    -------
    CoreModule
        The new core module.
    """
    core = CoreModule(index=0)
    core.name = "C"  # type: ignore[attr-defined]  # known bug
    return core


def attach(
    parent: MODULES,
    face: ModuleFaces,
    module: SUBMODULES,
    name: str,
    rotation: float = 0,
    *,
    root: CoreModule,
) -> SUBMODULES:
    """Attach `module` to `parent` on `face`, record it, and return it.

    Returns
    -------
    SUBMODULES
        The attached module.
    """
    name = f"{parent.name}-{name}"  # type: ignore[union-attr]  # known bug
    attach_part(root, parent, face, module, name + "-", rotation)

    module.name = name  # type: ignore[union-attr]  # known bug

    if rotation != 0:
        module.rotate(rotation)

    return module


def body_spider() -> CoreModule:
    """Return the spider body: a core with four legs.

    Returns
    -------
    CoreModule
        The spider body.
    """
    core = make_core()

    for i, f in enumerate(faces, start=1):
        h0 = attach(core, f, HingeModule(index=i), f"{f.name[0]}H", root=core)
        b0 = attach(h0, F, BrickModule(index=4 + i), "B", root=core)
        h1 = attach(
            b0, F, HingeModule(index=8 + i), "H", rotation=90, root=core
        )
        attach(h1, F, BrickModule(index=12 + i), "B", root=core)

    return core


def body_spider45() -> CoreModule:
    """Return the spider body rotated by 45 degrees about the vertical axis.

    Returns
    -------
    CoreModule
        The spider body, rotated.
    """
    core = body_spider()
    core.spec.body("core").quat = (
        math.cos(math.pi / 8),
        0,
        0,
        math.sin(math.pi / 8),
    )
    return core
