"""
Robots from Fuda's dataset, some robots adjusted to fit new ARIEL modules.

The ones that are adjusted have "modified" in the name.

The rest are by me!!! John.

The shape of each body is a table of parts in ``john_tables``.
"""

# Local libraries
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.john_tables import (
    BABY_A,
    BABY_B,
    CENTIPEDE_3,
    CENTIPEDE_4,
    CENTIPEDE_5,
    GECKO_SMALL,
    IGUANA,
    LINKIN_MODIFIED,
    SNAKE,
    SPIDER_8,
    SPIDER_12,
    SPIDER_16,
    TURTLE,
)
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.parts import build


def baby_a() -> CoreModule:
    """Build a custom robot body made with the 3D editor.

    Returns
    -------
    CoreModule
        The core module of the finished robot body.
    """
    return build(BABY_A)


def baby_b() -> CoreModule:
    """Build a custom robot body made with the 3D editor.

    Returns
    -------
    CoreModule
        The core module of the finished robot body.
    """
    return build(BABY_B)


def gecko_small() -> CoreModule:
    """Build a custom robot body made with the 3D editor.

    Returns
    -------
    CoreModule
        The core module of the finished robot body.
    """
    return build(GECKO_SMALL)


def linkin_modified() -> CoreModule:
    """Build a custom robot body made with the 3D editor.

    Returns
    -------
    CoreModule
        The core module of the finished robot body.
    """
    return build(LINKIN_MODIFIED)


def snake() -> CoreModule:
    """Build a custom robot body made with the 3D editor.

    Returns
    -------
    CoreModule
        The core module of the finished robot body.
    """
    return build(SNAKE)


def turtle() -> CoreModule:
    """Build a custom robot body made with the 3D editor.

    Returns
    -------
    CoreModule
        The core module of the finished robot body.
    """
    return build(TURTLE)


def iguana() -> CoreModule:
    """Build a custom robot body made with the 3D editor.

    Returns
    -------
    CoreModule
        The core module of the finished robot body.
    """
    return build(IGUANA)


def spider_8() -> CoreModule:
    """Build a custom robot body made with the 3D editor.

    Returns
    -------
    CoreModule
        The core module of the finished robot body.
    """
    return build(SPIDER_8)


def spider_12() -> CoreModule:
    """Build a custom robot body made with the 3D editor.

    Returns
    -------
    CoreModule
        The core module of the finished robot body.
    """
    return build(SPIDER_12)


def spider_16() -> CoreModule:
    """Build a custom robot body made with the 3D editor.

    Returns
    -------
    CoreModule
        The core module of the finished robot body.
    """
    return build(SPIDER_16)


def centipede_3() -> CoreModule:
    """Build a custom robot body made with the 3D editor.

    Returns
    -------
    CoreModule
        The core module of the finished robot body.
    """
    return build(CENTIPEDE_3)


def centipede_4() -> CoreModule:
    """Build a custom robot body made with the 3D editor.

    Returns
    -------
    CoreModule
        The core module of the finished robot body.
    """
    return build(CENTIPEDE_4)


def centipede_5() -> CoreModule:
    """Build a custom robot body made with the 3D editor.

    Returns
    -------
    CoreModule
        The core module of the finished robot body.
    """
    return build(CENTIPEDE_5)
