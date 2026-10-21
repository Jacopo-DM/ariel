"""Test: every world uses the one green of ``floor_colors``."""

# Standard library
import importlib

# Third-party libraries
import mujoco
import pytest

# Local libraries
from ariel.simulation.environments import (
    AmphitheatreTerrainWorld,
    BaseWorld,
    CompoundWorld,
    CraterTerrainWorld,
    OlympicArena,
    RuggedTerrainWorld,
    RuggedTiltedWorld,
    SimpleFlatWorld,
    SimpleTiltedWorld,
)
from ariel.simulation.environments.floor_colors import (
    FLOOR_DARK,
    FLOOR_LIGHT,
    FLOOR_RGBA,
)

SimpleFlatWorldWithTarget = importlib.import_module(
    "ariel.simulation.environments._simple_flat_with_target"
).SimpleFlatWorldWithTarget


def _checker_worlds() -> list[BaseWorld]:
    return [
        SimpleFlatWorld(load_precompiled=False),
        SimpleFlatWorldWithTarget(load_precompiled=False),
        SimpleTiltedWorld(load_precompiled=False),
        CompoundWorld(load_precompiled=False),
    ]


def _plain_worlds() -> list[BaseWorld]:
    return [
        SimpleTiltedWorld(checker_floor=False, load_precompiled=False),
        CompoundWorld(checker_floor=False, load_precompiled=False),
        RuggedTerrainWorld(load_precompiled=False),
        RuggedTiltedWorld(load_precompiled=False),
        CraterTerrainWorld(load_precompiled=False),
        AmphitheatreTerrainWorld(load_precompiled=False),
    ]


@pytest.mark.parametrize("world", _checker_worlds(), ids=lambda w: w.name)
def test_checkered_floors_alternate_the_two_greens(world: BaseWorld) -> None:
    """The floor texture holds the dark and the light green."""
    texture = world.spec.texture(world.mujoco_config.floor_name)
    assert tuple(texture.rgb1) == pytest.approx(FLOOR_DARK)
    assert tuple(texture.rgb2) == pytest.approx(FLOOR_LIGHT)


@pytest.mark.parametrize("world", _plain_worlds(), ids=lambda w: w.name)
def test_plain_floors_use_the_light_green(world: BaseWorld) -> None:
    """The floor geom carries the light green as its colour."""
    geom = world.spec.geom(world.mujoco_config.floor_name)
    assert tuple(geom.rgba) == pytest.approx(FLOOR_RGBA)


def test_the_olympic_flat_sections_use_the_greens() -> None:
    """The grid texture of the arena is the same green checker."""
    world = OlympicArena(load_precompiled=False)
    texture = world.spec.texture("grid")
    assert tuple(texture.rgb1) == pytest.approx(FLOOR_DARK)
    assert tuple(texture.rgb2) == pytest.approx(FLOOR_LIGHT)


def test_a_tilted_world_without_a_checker_floor_compiles() -> None:
    """Before the floors shared a colour this crashed with AttributeError."""
    world = SimpleTiltedWorld(checker_floor=False, load_precompiled=False)
    model = world.compile()
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor") >= 0
    del world, model
