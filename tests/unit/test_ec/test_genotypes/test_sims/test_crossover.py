"""Test: crossover and grafting of Sims genomes."""

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
    crossover,
    graft,
    unfold,
)
from ariel.ec.genotypes.sims.mutations import reached
from tests.unit.test_ec.test_genotypes.test_sims.genomes import (
    arm_genome,
    genomes,
)

SEEDS = range(40)
CORE = SimsNode(type=ModuleType.CORE)
BRICK = SimsNode(type=ModuleType.BRICK, copy_limit=2)
HINGE = SimsNode(type=ModuleType.HINGE)
TALL = SimsNode(type=ModuleType.BRICK, copy_limit=5)
SEED_RANGE = st.integers(min_value=0, max_value=2**32 - 1)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _bricks() -> SimsGenome:
    """Core and three bricks in a chain: 0 -> 1 -> 2 -> 3.

    Returns
    -------
    SimsGenome
        Parent A of the crossover tests.
    """
    return SimsGenome(
        nodes=(CORE, BRICK, BRICK, BRICK),
        connections=(
            SimsConnection(parent=0, child=1, face=ModuleFaces.FRONT),
            SimsConnection(parent=1, child=2, face=ModuleFaces.TOP),
            SimsConnection(parent=2, child=3, face=ModuleFaces.LEFT),
        ),
    )


def _hinges() -> SimsGenome:
    """Core and two hinges: 0 -> 1, 1 -> 2.

    Returns
    -------
    SimsGenome
        Parent B of the crossover tests.
    """
    return SimsGenome(
        nodes=(CORE, HINGE, HINGE),
        connections=(
            SimsConnection(parent=0, child=1, face=ModuleFaces.RIGHT),
            SimsConnection(parent=1, child=2, face=ModuleFaces.FRONT),
        ),
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_crossover_joins_the_two_rows(seed: int) -> None:
    """Templates before the cut come from A, the rest from B."""
    first, second = _bricks(), _hinges()
    child = crossover(first, second, _rng(seed))
    nodes = child.nodes
    cut = next(
        (i for i, n in enumerate(nodes) if n is not first.nodes[i]),
        len(nodes),
    )
    assert nodes == (*first.nodes[:cut], *second.nodes[cut:])
    for connection in child.connections:
        source = first if connection.parent < cut else second
        assert connection.face in {
            c.face for c in source.connections if c.parent == connection.parent
        }
        assert connection.child < len(nodes)


def test_crossover_redirects_connections_past_the_end() -> None:
    """A's connection to its third brick points at an existing template."""
    first, second = _bricks(), _hinges()
    children = [crossover(first, second, _rng(seed)) for seed in SEEDS]
    cut_at_two = [
        c for c in children if c.nodes[1] is BRICK and c.nodes[2] is BRICK
    ]
    assert cut_at_two  # some seeds cut after A's second brick
    for child in cut_at_two:
        assert len(child.nodes) == len(second.nodes)
        assert all(c.child < len(child.nodes) for c in child.connections)


def test_crossover_with_a_bare_core() -> None:
    """Crossing with the core alone gives the core alone."""
    assert crossover(arm_genome(), SimsGenome(), _rng(0)) == SimsGenome()


def test_crossover_of_a_genome_with_itself_keeps_its_genes() -> None:
    """Crossing a genome with itself keeps every template and connection."""
    genome = _bricks()
    for seed in SEEDS:
        child = crossover(genome, genome, _rng(seed))
        assert child.nodes == genome.nodes
        assert set(child.connections) == set(genome.connections)


@pytest.mark.parametrize("seed", SEEDS)
def test_graft_points_one_connection_at_the_other_parent(seed: int) -> None:
    """The child keeps A's core and reaches at least one of B's templates."""
    first = SimsGenome(
        nodes=(CORE, TALL),
        connections=(
            SimsConnection(parent=0, child=1, face=ModuleFaces.FRONT),
            SimsConnection(parent=1, child=1, face=ModuleFaces.FRONT),
        ),
    )
    child = graft(first, _hinges(), _rng(seed))
    assert child.nodes[0] == CORE
    assert HINGE in child.nodes
    assert reached(child) == set(range(len(child.nodes)))
    assert child.connections  # the grafted connection at least


def test_graft_without_anything_to_graft_returns_the_first_parent() -> None:
    """No connection in A, or no template in B: the child is A."""
    assert graft(SimsGenome(), _hinges(), _rng(0)) == SimsGenome()
    assert graft(arm_genome(), SimsGenome(), _rng(0)) == arm_genome()


def test_graft_leaves_the_second_core_behind() -> None:
    """B's connections from its own core are not copied."""
    first = arm_genome()
    core_connections = sum(c.parent == 0 for c in first.connections)
    for seed in SEEDS:
        child = graft(first, _hinges(), _rng(seed))
        assert child.nodes.count(CORE) == 1
        assert sum(c.parent == 0 for c in child.connections) == core_connections


@given(genomes(), genomes(), SEED_RANGE)
def test_crossover_gives_valid_children(
    first: SimsGenome, second: SimsGenome, seed: int
) -> None:
    """Any two parents cross into a valid genome that unfolds."""
    child = crossover(first, second, _rng(seed))
    assert child.nodes[0] == CORE
    unfold(child)


@given(genomes(), genomes(), SEED_RANGE)
def test_graft_gives_valid_children(
    first: SimsGenome, second: SimsGenome, seed: int
) -> None:
    """Any two parents graft into a valid genome with nothing unreached."""
    child = graft(first, second, _rng(seed))
    unfold(child)
    if child is not first:
        assert reached(child) == set(range(len(child.nodes)))
