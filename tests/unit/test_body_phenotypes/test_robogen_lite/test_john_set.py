"""Test: the hand-built robot bodies of John's set compile and step."""

import hashlib
from collections.abc import Callable
from typing import cast

import mujoco
import pytest

from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots import john_set

# Looked up by name, so each key is the name of its function
BUILDERS: dict[str, Callable[[], CoreModule]] = {
    name: getattr(john_set, name)
    for name in (
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
}


@pytest.mark.parametrize("name", list(BUILDERS))
def test_body_builds_and_compiles(name: str) -> None:
    """Every body in the set is a CoreModule that compiles and steps."""
    core = BUILDERS[name]()
    assert isinstance(core, CoreModule)
    model = cast("mujoco.MjModel", core.spec.compile())
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    assert model.nbody > 1
    del core, model, data


# sha256 (first 16 hex digits) of ``spec.to_xml()`` for every body, taken from
# the hand-written builders before they became a table. A change here means a
# robot changed shape, names or order. Taken again on 2026-10-21, when the
# modules got their measured sizes and masses (physical-checks PC3).
XML_HASHES = {
    "baby_a": "13c9854493d25c5e",
    "baby_b": "1928d4975f508ac4",
    "gecko_small": "16c02b4b7e2d5189",
    "linkin_modified": "362587c94b75457f",
    "snake": "10db453350b0fdda",
    "turtle": "15d61300fc6f7673",
    "iguana": "d3354a77f548e88c",
    "spider_8": "cbf1afd2f5427391",
    "spider_12": "deb556b6d7383fa3",
    "spider_16": "f9d531951b4f3f92",
    "centipede_3": "8263f1cbe586f6b5",
    "centipede_4": "b83d2772556e73d2",
    "centipede_5": "5621784d5ed35e2a",
}


def test_every_body_has_a_recorded_hash() -> None:
    """The hash table covers exactly the bodies the module offers."""
    assert sorted(XML_HASHES) == sorted(BUILDERS)


@pytest.mark.parametrize("name", list(BUILDERS))
def test_body_xml_is_unchanged(name: str) -> None:
    """The compiled-spec XML of each body matches the recorded hash."""
    core = BUILDERS[name]()
    digest = hashlib.sha256(core.spec.to_xml().encode()).hexdigest()[:16]
    assert digest == XML_HASHES[name]
    del core
