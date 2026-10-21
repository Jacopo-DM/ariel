# ruff: file-ignore[suspicious-non-cryptographic-random-usage]  # pseudo-random numbers drive the search, no security use
"""Randomized 20-module prebuilt morphology.

This module exposes a reproducible random morphology generator that creates
exactly 20 modules (1 core + 19 submodules), making it larger than the
standard spider prebuilt body.
"""

import random

from ariel.body_phenotypes.robogen_lite.config import ModuleFaces
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.parts import attach

HINGE_PROBABILITY = 0.6


def random_spider20(seed: int = 1337) -> CoreModule:
    """Build a reproducible random 20-module morphology.

    Parameters
    ----------
    seed
        Seed for reproducible random morphology generation.

    Returns
    -------
    CoreModule
        Core of a randomized robot with exactly 20 total modules.

    Raises
    ------
    RuntimeError
        If no open attachment site is left.
    """
    rng = random.Random(seed)

    core = CoreModule(index=0)
    seed_faces = [
        ModuleFaces.FRONT,
        ModuleFaces.LEFT,
        ModuleFaces.BACK,
        ModuleFaces.RIGHT,
    ]

    # Queue of still-available attachment sites: (parent_module, parent_face)
    open_sites: list[
        tuple[CoreModule | HingeModule | BrickModule, ModuleFaces]
    ] = [(core, face) for face in seed_faces if face in core.sites]

    hinge_rotations = [0, 45, 90]

    # Create 19 submodules so total modules (including core) is exactly 20.
    for idx in range(1, 20):
        if not open_sites:
            msg = (
                "No open attachment sites remain while building random_spider20"
            )
            raise RuntimeError(msg)

        site_idx = rng.randrange(len(open_sites))
        parent, attach_face = open_sites.pop(site_idx)

        # Slight hinge bias usually increases articulation in random bodies.
        rotation = 0
        if rng.random() < HINGE_PROBABILITY:
            child = HingeModule(index=idx)
            rotation = rng.choice(hinge_rotations)
            child.rotate(rotation)
        else:
            child = BrickModule(index=idx)  # type: ignore[assignment]  # known bug

        attach(core, parent, attach_face, child, f"m{idx}", rotation)

        # Only enqueue attachment faces that this module actually exposes.
        open_sites.extend((child, child_face) for child_face in child.sites)

    return core
