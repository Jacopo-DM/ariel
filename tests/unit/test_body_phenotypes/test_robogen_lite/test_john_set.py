"""Test: the hand-built robot bodies of John's set compile and step."""

from collections.abc import Callable

import mujoco
import pytest

from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots import john_set

BUILDERS: list[Callable[[], CoreModule]] = [
    john_set.baby_a,
    john_set.baby_b,
    john_set.gecko,
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
