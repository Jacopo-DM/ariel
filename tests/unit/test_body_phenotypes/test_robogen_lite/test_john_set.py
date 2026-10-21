"""Test: the hand-built robot bodies of John's set compile and step."""

import hashlib
from collections.abc import Callable

import mujoco
import pytest

from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots import john_set

BUILDERS: list[Callable[[], CoreModule]] = [
    john_set.baby_a,
    john_set.baby_b,
    john_set.gecko_small,
    john_set.linkin_modified,
    john_set.snake,
    john_set.turtle,
    john_set.iguana,
    john_set.spider_8,
    john_set.spider_12,
    john_set.spider_16,
    john_set.centipede_3,
    john_set.centipede_4,
    john_set.centipede_5,
]


@pytest.mark.parametrize("builder", BUILDERS, ids=lambda f: f.__name__)
def test_body_builds_and_compiles(builder: Callable[[], CoreModule]) -> None:
    """Every body in the set is a CoreModule that compiles and steps."""
    core = builder()
    assert isinstance(core, CoreModule)
    model = core.spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    assert model.nbody > 1
    del core, model, data


# sha256 (first 16 hex digits) of ``spec.to_xml()`` for every body, taken from
# the hand-written builders before they became a table. A change here means a
# robot changed shape, names or order.
XML_HASHES = {
    "baby_a": "072fdd2263904d89",
    "baby_b": "12fc5667a91747c6",
    "gecko_small": "04473cf596aec17d",
    "linkin_modified": "832b1960e1c29c3d",
    "snake": "52280e37a134ae8b",
    "turtle": "a4869e5324c55b9f",
    "iguana": "b955c97f97b65dcc",
    "spider_8": "bcb4c03cd04a7e3e",
    "spider_12": "d9fc03b81751c7c8",
    "spider_16": "30ec920d3aafdcb5",
    "centipede_3": "85e596e9b7c718ae",
    "centipede_4": "ecf9cbff9f094a33",
    "centipede_5": "86599d34b2a37465",
}


def test_every_body_has_a_recorded_hash() -> None:
    """The hash table covers exactly the bodies the module offers."""
    assert sorted(XML_HASHES) == sorted(b.__name__ for b in BUILDERS)


@pytest.mark.parametrize("builder", BUILDERS, ids=lambda f: f.__name__)
def test_body_xml_is_unchanged(builder: Callable[[], CoreModule]) -> None:
    """The compiled-spec XML of each body matches the recorded hash."""
    core = builder()
    digest = hashlib.sha256(core.spec.to_xml().encode()).hexdigest()[:16]
    assert digest == XML_HASHES[builder.__name__]
    del core
