"""Test: the hand-built robot bodies of John's set compile and step."""

import hashlib
from collections.abc import Callable

import mujoco
import pytest

from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots import john_set
from ariel.simulators import compile_model

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
    model = compile_model(core.spec)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    assert model.nbody > 1
    del core, model, data


# sha256 (first 16 hex digits) of ``spec.to_xml()`` for every body, taken from
# the hand-written builders before they became a table. A change here means a
# robot changed shape, names or order. Taken again on 2026-10-21, when the
# modules got their measured sizes and masses (physical-checks PC3), and on
# 2026-10-22, when every hinge became a DSS-M15 (servo-dynamics SD2).
XML_HASHES = {
    "baby_a": "544c813b6b2289db",
    "baby_b": "6818963c0b30a20a",
    "gecko_small": "000f7dc355175b7f",
    "linkin_modified": "3b99ac0db6fb30d9",
    "snake": "db93990cf85b1d52",
    "turtle": "ad2d98a4e742d582",
    "iguana": "62d7b664e6b618c0",
    "spider_8": "60ce9ba1d45f605f",
    "spider_12": "db7a43f8119476cc",
    "spider_16": "33542a1ef2254591",
    "centipede_3": "8a89db162113d412",
    "centipede_4": "7118d206fc708c75",
    "centipede_5": "ddd4799630af3d26",
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
