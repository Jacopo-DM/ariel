"""Test: mutation operators for Sims genomes."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from ariel.body_phenotypes import ModuleFaces, ModuleType
from ariel.ec.genotypes.sims import (
    SimsConnection,
    SimsGenome,
    SimsNode,
    mutate,
    unfold,
)
from ariel.ec.genotypes.sims.mutations import (
    add_connection,
    add_node,
    change_connection,
    change_node,
    drop_unreached,
    reached,
    remove_connection,
    remove_node,
)
from tests.unit.test_ec.test_genotypes.test_sims.genomes import (
    arm_genome,
    genomes,
)

SEEDS = range(60)
CAP = 6
CORE_AND_ONE = 2  # nodes after the first template is added
CORE = SimsNode(type=ModuleType.CORE)
BRICK = SimsNode(type=ModuleType.BRICK, copy_limit=2)
HINGE = SimsNode(type=ModuleType.HINGE)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _differences[T](before: tuple[T, ...], after: tuple[T, ...]) -> int:
    return sum(a != b for a, b in zip(before, after, strict=True))


def _with_unreached() -> SimsGenome:
    """Core -> brick 1; hinge 2 unreached, with a connection 2 -> 1.

    Returns
    -------
    SimsGenome
        A genome with one template no path reaches.
    """
    return SimsGenome(
        nodes=(CORE, BRICK, HINGE),
        connections=(
            SimsConnection(parent=0, child=1, face=ModuleFaces.FRONT),
            SimsConnection(parent=2, child=1, face=ModuleFaces.FRONT),
        ),
    )


def test_reached() -> None:
    """Only nodes a path of connections leads to count as reached."""
    assert reached(arm_genome()) == {0, 1}
    assert reached(_with_unreached()) == {0, 1}
    assert reached(SimsGenome()) == {0}


@pytest.mark.parametrize("seed", SEEDS)
def test_change_node_changes_one_template(seed: int) -> None:
    """One template changes; only its own connections may change face."""
    genome = SimsGenome(
        nodes=(CORE, BRICK, BRICK),
        connections=(
            SimsConnection(parent=0, child=1, face=ModuleFaces.LEFT),
            SimsConnection(parent=1, child=2, face=ModuleFaces.TOP),
            SimsConnection(parent=2, child=1, face=ModuleFaces.RIGHT),
        ),
    )
    changed = change_node(genome, _rng(seed), CAP)
    assert _differences(genome.nodes, changed.nodes) == 1
    (index,) = (
        i
        for i, (a, b) in enumerate(
            zip(genome.nodes, changed.nodes, strict=True)
        )
        if a != b
    )
    for old, new in zip(genome.connections, changed.connections, strict=True):
        if old != new:
            assert old.parent == index
            assert new.face is ModuleFaces.FRONT  # it became a hinge
            assert old.model_copy(update={"face": new.face}) == new


def test_change_node_without_room_keeps_copy_limits() -> None:
    """With a cap of 1 the copy limit is never what changes."""
    genome = SimsGenome(nodes=(CORE, SimsNode(type=ModuleType.BRICK)))
    for seed in SEEDS:
        changed = change_node(genome, _rng(seed), 1)
        assert changed.nodes[1].copy_limit == 1
        assert changed != genome


@pytest.mark.parametrize(
    ("nodes", "parent", "mirror"),
    [
        pytest.param((CORE, BRICK, HINGE), 0, True, id="core-parent"),
        pytest.param((CORE, HINGE), 1, False, id="hinge-parent-no-mirror"),
    ],
)
def test_change_connection_changes_one_connection(
    nodes: tuple[SimsNode, ...], parent: int, *, mirror: bool
) -> None:
    """Exactly one connection changes; mirror off never switches it on."""
    genome = SimsGenome(
        nodes=nodes,
        connections=(
            SimsConnection(parent=0, child=1, face=ModuleFaces.FRONT),
            SimsConnection(parent=parent, child=1, face=ModuleFaces.FRONT),
        ),
    )
    for seed in SEEDS:
        changed = change_connection(genome, _rng(seed), mirror=mirror)
        assert changed.nodes == genome.nodes
        assert _differences(genome.connections, changed.connections) == 1
        assert mirror or not any(c.reflect for c in changed.connections)


def test_change_connection_can_switch_a_mirror_off() -> None:
    """With mirror off, an existing mirror flag can still be switched off."""
    genome = SimsGenome(
        nodes=(CORE, HINGE),
        connections=(
            SimsConnection(
                parent=0, child=1, face=ModuleFaces.LEFT, reflect=True
            ),
        ),
    )
    flags = {
        change_connection(genome, _rng(seed), mirror=False)
        .connections[0]
        .reflect
        for seed in SEEDS
    }
    assert flags == {True, False}


def test_add_node_adds_a_reached_template() -> None:
    """The new template is last, and the new connection points at it."""
    genome = arm_genome()
    for seed in SEEDS:
        grown = add_node(genome, _rng(seed), CAP, mirror=False)
        assert grown.nodes[:-1] == genome.nodes
        assert grown.connections[-1].child == len(genome.nodes)
        assert not grown.connections[-1].reflect
        assert len(reached(grown)) == len(reached(genome)) + 1


def test_remove_node_renumbers() -> None:
    """Removing template 1 drops its connections and renumbers template 2."""
    genome = SimsGenome(
        nodes=(CORE, BRICK, HINGE),
        connections=(
            SimsConnection(parent=0, child=1, face=ModuleFaces.FRONT),
            SimsConnection(parent=0, child=2, face=ModuleFaces.LEFT),
        ),
    )
    for seed in SEEDS:
        smaller = remove_node(genome, _rng(seed))
        assert len(smaller.nodes) == len(genome.nodes) - 1
        assert len(smaller.connections) == 1
        assert smaller.connections[0].child == 1


def test_add_and_remove_connection() -> None:
    """A connection is appended, or one is taken away."""
    genome = arm_genome()
    for seed in SEEDS:
        assert add_connection(genome, _rng(seed), mirror=True).connections[
            :-1
        ] == (genome.connections)
        shorter = remove_connection(genome, _rng(seed))
        assert len(shorter.connections) == len(genome.connections) - 1
        assert set(shorter.connections) < set(genome.connections)


def test_drop_unreached() -> None:
    """The unreached hinge goes, with its connection; the rest stays."""
    dropped = drop_unreached(_with_unreached())
    assert dropped.nodes == (CORE, BRICK)
    assert dropped.connections == _with_unreached().connections[:1]


def test_mutate_core_alone_adds_a_template() -> None:
    """A genome of the core alone can only grow."""
    for seed in SEEDS:
        assert len(mutate(SimsGenome(), _rng(seed)).nodes) == CORE_AND_ONE


@given(
    genome=genomes(),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    mirror=st.booleans(),
)
def test_mutate_always_changes_a_genome(
    genome: SimsGenome, seed: int, *, mirror: bool
) -> None:
    """Every mutation gives a different, valid genome that unfolds."""
    mutated = mutate(genome, _rng(seed), mirror=mirror)
    assert mutated != genome
    unfold(mutated)
    reflects = sum(c.reflect for c in mutated.connections)
    assert mirror or reflects <= sum(c.reflect for c in genome.connections)
