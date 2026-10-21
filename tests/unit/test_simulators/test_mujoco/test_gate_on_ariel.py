"""Test: every robot and world ARIEL ships passes the plausibility checks."""

import inspect

import pytest

from ariel.body_phenotypes.lynx_mjspec.lynx_arm import LynxArm
from ariel.body_phenotypes.lynx_mjspec.table import TableWorld
from ariel.simulation import environments
from ariel.simulation.environments import BaseWorld, SimpleFlatWorld
from ariel.simulators import check_model
from tests.unit.test_body_phenotypes.test_robogen_lite.prebuilt_bodies import (
    PREBUILT,
)

WORLDS = sorted(
    name
    for name, cls in inspect.getmembers(environments, inspect.isclass)
    if issubclass(cls, BaseWorld) and cls is not BaseWorld
)


@pytest.mark.parametrize("name", sorted(PREBUILT))
def test_a_prebuilt_robot_is_plausible(name: str) -> None:
    world = SimpleFlatWorld()
    world.spawn(PREBUILT[name]().spec)
    assert check_model(world.compile()) == []


@pytest.mark.parametrize("name", WORLDS)
def test_a_world_with_a_robot_is_plausible(name: str) -> None:
    world = getattr(environments, name)(load_precompiled=False)
    world.spawn(PREBUILT["gecko"]().spec)
    assert check_model(world.compile()) == []


def test_the_lynx_arm_on_its_table_is_plausible() -> None:
    world = TableWorld()
    world.spawn(LynxArm().spec)
    assert check_model(world.compile()) == []
