"""Every prebuilt robot of ARIEL, by name, for the tests."""

from __future__ import annotations

# Standard library
from typing import TYPE_CHECKING

# Local libraries
from ariel.body_phenotypes.robogen_lite.prebuilt_robots import (
    gecko,
    john_set,
    spider,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule

JOHN = (
    "baby_a",
    "baby_b",
    "gecko_small",
    "linkin_modified",
    "snake",
    "turtle",
    "iguana",
    "spider_8",
    "spider_12",
    "spider_16",
    "centipede_3",
    "centipede_4",
    "centipede_5",
)
PREBUILT: dict[str, Callable[[], CoreModule]] = {
    **{f"john_set.{name}": getattr(john_set, name) for name in JOHN},
    "gecko": gecko.gecko,
    "spider": spider.spider,
}
