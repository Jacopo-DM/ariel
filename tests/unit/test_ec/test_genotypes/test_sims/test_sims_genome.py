"""Test: the Sims genotype's model, checks and JSON form."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from pydantic import JsonValue, ValidationError

from ariel.body_phenotypes import ModuleFaces, ModuleRotationsIdx, ModuleType
from ariel.ec.genotypes.sims import SimsConnection, SimsGenome, SimsNode
from tests.unit.test_ec.test_genotypes.test_sims.genomes import (
    arm_genome,
    genomes,
)

if TYPE_CHECKING:
    from pathlib import Path

CORE = {"type": "CORE"}
BRICK = {"type": "BRICK"}
HINGE = {"type": "HINGE"}
ARM_LIMIT = 3


def _genome(
    nodes: list[dict[str, JsonValue]],
    connections: list[dict[str, JsonValue]] | None = None,
) -> SimsGenome:
    return SimsGenome.model_validate({
        "nodes": nodes,
        "connections": connections or [],
    })


def test_default_genome_is_the_core_alone() -> None:
    """A genome made with no arguments is one core and nothing else."""
    genome = SimsGenome()
    assert genome.nodes == (SimsNode(type=ModuleType.CORE),)
    assert genome.connections == ()


def test_node_defaults() -> None:
    """A node defaults to no rotation and a copy limit of 1."""
    node = SimsNode(type=ModuleType.BRICK)
    assert node.rotation is ModuleRotationsIdx.DEG_0
    assert node.copy_limit == 1


def test_json_names_members() -> None:
    """JSON holds member names, not numbers, and reads back the same."""
    genome = arm_genome()
    data = json.loads(genome.model_dump_json())
    assert data["nodes"][1] == {
        "type": "BRICK",
        "rotation": "DEG_0",
        "copy_limit": ARM_LIMIT,
    }
    assert data["connections"][0]["face"] == "LEFT"
    assert data["connections"][0]["reflect"] is True
    assert SimsGenome.model_validate(data) == genome


def test_save_and_load(tmp_path: Path) -> None:
    """A saved genome loads back equal."""
    path = tmp_path / "genome.json"
    arm_genome().save_json(path)
    assert SimsGenome.load_json(path) == arm_genome()


def test_genome_is_frozen() -> None:
    """A genome cannot be changed in place."""
    genome = arm_genome()
    with pytest.raises(ValidationError):
        genome.nodes = ()  # type: ignore[misc]  # ty: ignore[invalid-assignment]  # the point: assignment fails


@pytest.mark.parametrize(
    ("nodes", "connections"),
    [
        pytest.param(
            [CORE, BRICK],
            [{"parent": 1, "child": 1, "face": "FRONT"}],
            id="self-loop",
        ),
        pytest.param(
            [CORE, BRICK, HINGE],
            [
                {"parent": 1, "child": 1, "face": "TOP"},
                {"parent": 1, "child": 2, "face": "TOP"},
            ],
            id="two-on-one-face",
        ),
        pytest.param([CORE, BRICK, BRICK], [], id="unreached-nodes"),
        pytest.param(
            [CORE, BRICK],
            [
                {"parent": 0, "child": 1, "face": "RIGHT"},
                {
                    "parent": 1,
                    "child": 1,
                    "face": "BOTTOM",
                    "terminal_only": True,
                },
            ],
            id="terminal-only",
        ),
    ],
)
def test_allowed_shapes(
    nodes: list[dict[str, JsonValue]],
    connections: list[dict[str, JsonValue]],
) -> None:
    """Loops, shared faces, unreached nodes and terminal flags are valid."""
    genome = _genome(nodes, connections)
    assert len(genome.connections) == len(connections)


@pytest.mark.parametrize(
    ("nodes", "connections", "message"),
    [
        pytest.param([], [], "must be the core", id="no-nodes"),
        pytest.param([BRICK], [], "must be the core", id="root-not-core"),
        pytest.param(
            [{"type": "CORE", "copy_limit": 2}],
            [],
            "copy limit of 1",
            id="core-limit",
        ),
        pytest.param([CORE, CORE], [], "only the root", id="second-core"),
        pytest.param([CORE, {"type": "NONE"}], [], "NONE", id="none-node"),
        pytest.param(
            [{"type": "CORE", "rotation": "DEG_45"}],
            [],
            "does not allow",
            id="core-tilt",
        ),
        pytest.param(
            [CORE, {"type": "BRICK", "copy_limit": 0}],
            [],
            "greater than",
            id="limit-0",
        ),
        pytest.param(
            [CORE, {"type": "SPHERE"}],
            [],
            "not a ModuleType name",
            id="bad-type",
        ),
        pytest.param(
            [CORE, {"type": "BRICK", "rotation": "DEG_MINUS_45"}],
            [],
            "not a ModuleRotationsIdx name",
            id="builder-only-tilt",
        ),
        pytest.param(
            [CORE, BRICK],
            [{"parent": 0, "child": 2, "face": "FRONT"}],
            "does not exist",
            id="missing-child",
        ),
        pytest.param(
            [CORE, BRICK],
            [{"parent": 5, "child": 1, "face": "FRONT"}],
            "does not exist",
            id="missing-parent",
        ),
        pytest.param(
            [CORE, BRICK],
            [{"parent": 1, "child": 0, "face": "FRONT"}],
            "never a child",
            id="core-as-child",
        ),
        pytest.param(
            [CORE, HINGE],
            [{"parent": 1, "child": 1, "face": "TOP"}],
            "has no TOP face",
            id="hinge-top",
        ),
        pytest.param(
            [CORE, BRICK, BRICK],
            [
                {"parent": 0, "child": 1, "face": "FRONT"},
                {"parent": 1, "child": 2, "face": "BACK"},
            ],
            "has no BACK face",
            id="brick-back",
        ),
        pytest.param(
            [CORE, BRICK],
            [{"parent": 0, "child": 1, "face": "SIDE"}],
            "not a ModuleFaces name",
            id="bad-face",
        ),
        pytest.param(
            [CORE, {"type": "BRICK", "colour": "red"}],
            [],
            "Extra inputs",
            id="unknown-field",
        ),
    ],
)
def test_rejected_shapes(
    nodes: list[dict[str, JsonValue]],
    connections: list[dict[str, JsonValue]],
    message: str,
) -> None:
    """Each broken genome raises ValidationError with a message naming why."""
    with pytest.raises(ValidationError, match=message):
        _genome(nodes, connections)


def test_members_are_accepted_as_well_as_names() -> None:
    """Python code may pass enum members; JSON passes names."""
    connection = SimsConnection(parent=0, child=1, face=ModuleFaces.RIGHT)
    assert connection.face is ModuleFaces.RIGHT


@given(genomes())
def test_round_trip(genome: SimsGenome) -> None:
    """Loading a dumped genome gives the same genome back."""
    assert SimsGenome.model_validate_json(genome.model_dump_json()) == genome


@given(st.text())
@example(arm_genome().model_dump_json())
@example('{"nodes": [{"type": 7}]}')
def test_load_boundary(raw: str) -> None:
    """Any text loads as a genome or raises ValidationError, nothing else."""
    try:
        genome = SimsGenome.model_validate_json(raw)
    except ValidationError:
        return
    assert isinstance(genome, SimsGenome)
