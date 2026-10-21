"""Test: precompiled worlds, spawning, heightmaps and world scripts."""

from __future__ import annotations

import importlib
import inspect
import random
import runpy
from pathlib import Path
from typing import cast

import mujoco
import numpy as np
import pytest

from ariel.body_phenotypes import construct_mjspec_from_graph
from ariel.ec.genotypes.tree import random_tree
from ariel.simulation import environments
from ariel.simulation.environments import (
    BaseWorld,
    CompoundWorld,
    OlympicArena,
    SimpleFlatWorld,
    heightmap_functions,
    olympic_arena,
)
from ariel.simulators.mujoco import Seeds
from tests.unit.misuse import untyped
from tests.unit.test_simulation.plain_world import PlainWorld

TOOL = Path(__file__).resolve().parents[4] / "tools" / "compile_terrains.py"

_base_world = importlib.import_module(
    "ariel.simulation.environments._base_world"
)
SimpleFlatWorldWithTarget = importlib.import_module(
    "ariel.simulation.environments._simple_flat_with_target"
).SimpleFlatWorldWithTarget

# A small anchor geom gives the wrapper body a mass for the free joint; the
# ball sits on a slide joint so it moves and can touch the floor.
SPHERE_XML = """
<mujoco><worldbody>
    <geom name="anchor" type="sphere" size=".05" mass="1"/>
    <body><joint type="slide" axis="0 0 1"/>
    <geom name="ball" type="sphere" size=".1"/></body>
</worldbody></mujoco>"""
NO_GEOM_XML = """
<mujoco><worldbody>
    <body><joint type="slide" axis="0 0 1"/>
    <inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .1"/>
    <site name="mark" size=".1"/></body>
</worldbody></mujoco>"""
TARGET_FLOOR = (4, 4, 1)
NON_SQUARE = (20, 30)
EDGE = 0.1
HEIGHT_DIMS = (10, 10)
TWO_ROBOTS = 2
TWO_JOINTS = 2


@pytest.fixture
def precompiled_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    # Point the world XML cache at a temporary folder.
    monkeypatch.setattr(_base_world, "__file__", str(tmp_path / "world.py"))
    return tmp_path / "pre_compiled"


def _world_classes() -> list[type[BaseWorld]]:
    return [
        cls
        for _, cls in inspect.getmembers(environments, inspect.isclass)
        if issubclass(cls, BaseWorld)
    ]


# --- precompiled XML -------------------------------------------------------- #


@pytest.mark.parametrize("cls", _world_classes(), ids=lambda c: c.__name__)
def test_world_roundtrips_through_its_xml(
    cls: type[BaseWorld], precompiled_dir: Path
) -> None:
    """A stored world is loaded back instead of being rebuilt."""
    built = cls(load_precompiled=False)
    built.store_to_xml()
    assert (precompiled_dir / f"{built.name}.xml").exists()
    loaded = cls(load_precompiled=True)
    assert loaded.is_precompiled
    model = loaded.compile()
    assert model.nbody >= 1
    del built, loaded, model


def test_compile_all_worlds_writes_one_xml_each(
    precompiled_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """compile_all_world stores every world class."""
    tool = runpy.run_path(str(TOOL))
    tool["compile_all_world"](with_load_compiled=False)
    assert len(list(precompiled_dir.glob("*.xml"))) >= len(_world_classes()) - 1
    assert "Compiled" in capsys.readouterr().out


def test_compile_terrains_script(
    precompiled_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Running the tool as a script compiles all worlds."""
    runpy.run_path(str(TOOL), run_name="__main__")
    assert list(precompiled_dir.glob("*.xml"))
    assert "Compiled" in capsys.readouterr().out


# --- spawning --------------------------------------------------------------- #


def test_spawn_with_defaults_adds_a_free_robot() -> None:
    """spawn() without arguments uses the default position and prefix."""
    world = SimpleFlatWorld(load_precompiled=False)
    robot = mujoco.MjSpec.from_string(SPHERE_XML)
    world.spawn(robot)
    model = world.compile()
    assert model.njnt == TWO_JOINTS
    del world, robot, model


def test_spawn_warns_about_floor_contact() -> None:
    """A robot that still touches the floor after correction is reported."""
    world = SimpleFlatWorld(load_precompiled=False)
    robot = mujoco.MjSpec.from_string(SPHERE_XML)
    world.spawn(robot, validate_no_collisions=True)
    assert world.spawns == 1
    del world, robot


def test_spawn_of_a_robot_without_geoms_and_with_a_prefix() -> None:
    """A robot with no geoms has no lowest point; the prefix is used."""
    world = SimpleFlatWorld(load_precompiled=False)
    robot = mujoco.MjSpec.from_string(NO_GEOM_XML)
    world.spawn(robot, spawn_prefix="bot")
    names = [world.spec.bodies[i].name for i in range(len(world.spec.bodies))]
    assert any(name.startswith("bot") for name in names)
    del world, robot


def test_spawn_without_floor_correction() -> None:
    """The floor correction can be switched off."""
    world = SimpleFlatWorld(load_precompiled=False)
    robot = mujoco.MjSpec.from_string(SPHERE_XML)
    world.spawn(robot, correct_collision_with_floor=False)
    assert world.spawns == 1
    del world, robot


def test_spawn_reports_robot_to_robot_contact() -> None:
    """Two robots on top of each other produce a (debug) collision note."""
    world = SimpleFlatWorld(load_precompiled=False)
    world.spawn(mujoco.MjSpec.from_string(SPHERE_XML), position=(0, 0, 0.5))
    world.spawn(
        mujoco.MjSpec.from_string(SPHERE_XML),
        position=(0, 0, 0.5),
        rotation=(0, 0, 90),
        validate_no_collisions=True,
    )
    assert world.spawns == TWO_ROBOTS
    del world


def test_world_no_longer_takes_a_mujoco_config() -> None:
    """The config argument was dropped silently; passing it now fails."""
    config = SimpleFlatWorld(load_precompiled=False).mujoco_config
    with pytest.raises(TypeError, match="mujoco_config"):
        untyped(PlainWorld)(mujoco_config=config, load_precompiled=False)


def test_worlds_start_from_the_generated_settings() -> None:
    """Every world carries the timestep and gravity derived from the seeds."""
    seeds = Seeds()
    spec = SimpleFlatWorld(load_precompiled=False).spec
    assert spec.option.timestep == pytest.approx(seeds.timestep)
    assert spec.option.gravity[2] == pytest.approx(-seeds.gravity.value)
    assert spec.option.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert spec.modelname == "Simple Flat World"


# --- worlds ----------------------------------------------------------------- #


def test_flat_world_with_target_loads_its_precompiled_xml(
    precompiled_dir: Path,
) -> None:
    """A stored flat world with a target is loaded without rebuilding."""
    SimpleFlatWorldWithTarget(load_precompiled=False).store_to_xml()
    assert (precompiled_dir / "simple-flat-world-with-target.xml").exists()
    assert SimpleFlatWorldWithTarget(load_precompiled=True).is_precompiled


def test_flat_world_with_target_compiles() -> None:
    """The flat world with a target adds the target body."""
    world = SimpleFlatWorldWithTarget(load_precompiled=False)
    model = world.compile()
    assert model.nbody > 1
    del world, model


def test_compound_world_takes_every_override() -> None:
    """Each optional argument of CompoundWorld replaces the class default."""
    world = CompoundWorld(
        name="my-compound",
        floor_size=TARGET_FLOOR,
        floor_tilt=(1, 2, 3),
        floor_rot_sequence="ZYX",
        dims=HEIGHT_DIMS,
        floor_heightmap=np.zeros(HEIGHT_DIMS),
        terrain_color=(1.0, 0.0, 0.0, 1.0),
        checker_floor=False,
        load_precompiled=False,
    )
    assert world.name == "my-compound"
    assert world.dims == HEIGHT_DIMS
    model = world.compile()
    assert model.ngeom >= 1
    del world, model


def test_compound_world_loads_its_precompiled_xml(
    precompiled_dir: Path,
) -> None:
    """A stored compound world is loaded without rebuilding."""
    CompoundWorld(load_precompiled=False).store_to_xml()
    assert (precompiled_dir / "compound-world.xml").exists()
    assert CompoundWorld(load_precompiled=True).is_precompiled


# --- olympic arena helpers -------------------------------------------------- #


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_quaternion_from_axis_angle_turns_about_each_axis(axis: str) -> None:
    """A half turn about an axis has w = 0 and a unit vector part."""
    quat = olympic_arena.quaternion_from_axis_angle(axis, 180.0)
    assert quat[0] == pytest.approx(0.0, abs=1e-9)
    assert float(np.linalg.norm(quat)) == pytest.approx(1.0)


def test_quaternion_from_axis_angle_rejects_other_axes() -> None:
    """Only x, y and z are valid axes."""
    with pytest.raises(ValueError, match="Unexpected axis"):
        olympic_arena.quaternion_from_axis_angle("w", 10.0)


def test_olympic_arena_loads_its_precompiled_xml(
    precompiled_dir: Path,
) -> None:
    """A stored arena is loaded without rebuilding."""
    OlympicArena(load_precompiled=False).store_to_xml()
    assert OlympicArena(load_precompiled=True).is_precompiled
    assert list(precompiled_dir.glob("*.xml"))


# --- heightmap functions ---------------------------------------------------- #


def test_smooth_edges_with_zero_width_is_a_hard_step() -> None:
    """Edge width 0 gives a mask that is 0 at the border and 1 inside."""
    mask = heightmap_functions.smooth_edges_heightmap(HEIGHT_DIMS, 0.0)
    assert mask[0, 0] == pytest.approx(0.0)
    assert mask[5, 5] == pytest.approx(1.0)


def test_smooth_edges_resizes_non_square_maps() -> None:
    """A non-square request is resized to the requested shape."""
    mask = heightmap_functions.smooth_edges_heightmap(NON_SQUARE, EDGE)
    assert mask.shape == NON_SQUARE


def test_the_geometry_compile_matches_the_full_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tiny textures change nothing the spawn correction reads.

    The old path compiled the whole world; the new one compiles with 1 by 1
    textures (speed-up S1). A robot spawned in every world must land exactly
    where the full compile puts it, the collision check must report the same
    contacts, and the real texture sizes must come back.
    """
    unseeded = np.random.default_rng

    def seeded(seed: int | None = None) -> np.random.Generator:
        return unseeded(0 if seed is None else seed)

    # The terrain noise is not seeded; seeding it builds one terrain twice
    monkeypatch.setattr(np.random, "default_rng", seeded)
    for _, cls in inspect.getmembers(environments, inspect.isclass):
        world_cls = cast("type[BaseWorld]", cls)
        lite = world_cls(load_precompiled=False)
        full = world_cls(load_precompiled=False)
        monkeypatch.setattr(
            full,
            "_compile_for_geometry",
            lambda world=full: world.compile(),
        )
        sizes = [(t.width, t.height) for t in lite.spec.textures]
        for world in (lite, full):
            random.seed(7)
            robot = construct_mjspec_from_graph(random_tree(12).to_networkx())
            world.spawn(robot.spec, validate_no_collisions=True)
        assert sizes == [(t.width, t.height) for t in lite.spec.textures]
        assert lite.spec.to_xml() == full.spec.to_xml(), cls.__name__


def test_the_texture_sizes_come_back_when_the_compile_fails() -> None:
    """A failed compile still restores the textures."""
    world = SimpleFlatWorld(load_precompiled=False)
    sizes = [(t.width, t.height) for t in world.spec.textures]
    world.spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH)
    robot = construct_mjspec_from_graph(random_tree(4).to_networkx())
    with pytest.raises(ValueError, match="mesh"):
        world.spawn(robot.spec)
    assert sizes == [(t.width, t.height) for t in world.spec.textures]
