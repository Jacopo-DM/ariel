"""Test: drawing random Sims genomes."""

from __future__ import annotations

import networkx as nx
import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from ariel.body_phenotypes import IDX_OF_CORE, construct_mjspec_from_graph
from ariel.ec.genotypes.sims import (
    SimsGenome,
    max_copy_limit,
    random_genome,
    unfold,
)
from ariel.ec.genotypes.sims.generate import (
    EXTRA_PER_TEMPLATE,
    MAX_TEMPLATES,
    MIN_MODULES,
)
from ariel.simulators import compile_model

SEED = 42
SAMPLES = 150
BIG_BODY = 100
SMALL_BODY = 4


def _sample(
    seed: int, max_modules: int = 30, *, mirror: bool = True
) -> list[SimsGenome]:
    rng = np.random.default_rng(seed)
    return [
        random_genome(rng, max_modules, mirror=mirror) for _ in range(SAMPLES)
    ]


def test_same_seed_same_genome() -> None:
    """Two generators with one seed draw the same genomes."""
    assert _sample(SEED)[:5] == _sample(SEED)[:5]


def test_sizes_and_limits_stay_in_range() -> None:
    """Template count, connection count and copy limits stay in range."""
    for max_modules in (30, BIG_BODY):
        cap = max_copy_limit(max_modules)
        for genome in _sample(SEED, max_modules):
            templates = len(genome.nodes) - 1
            assert 1 <= templates <= MAX_TEMPLATES
            assert (
                templates
                <= len(genome.connections)
                <= (1 + EXTRA_PER_TEMPLATE) * templates
            )
            assert all(1 <= n.copy_limit <= cap for n in genome.nodes)
            assert genome.connections[0].parent == IDX_OF_CORE
            assert len(unfold(genome, max_modules)) >= MIN_MODULES


def test_every_template_is_reached() -> None:
    """Template k has a connection from the core or an earlier template."""
    for genome in _sample(SEED):
        for k in range(1, len(genome.nodes)):
            assert any(
                c.child == k and c.parent < k for c in genome.connections
            )


def test_small_maximum_lowers_the_floor() -> None:
    """With a maximum under 10 modules, the floor is the maximum."""
    rng = np.random.default_rng(SEED)
    genome = random_genome(rng, max_modules=SMALL_BODY)
    assert len(unfold(genome, SMALL_BODY)) == SMALL_BODY


def test_mirror_off_draws_no_mirror_flags() -> None:
    """With mirror off, no connection carries the mirror flag."""
    connections = [
        c for g in _sample(SEED, mirror=False) for c in g.connections
    ]
    assert not any(c.reflect for c in connections)


def test_every_gene_value_appears() -> None:
    """Over many genomes, every type, flag and copy limit turns up."""
    genomes = _sample(SEED)
    nodes = [n for g in genomes for n in g.nodes[1:]]
    connections = [c for g in genomes for c in g.connections]
    assert {n.type.name for n in nodes} == {"BRICK", "HINGE"}
    assert {n.rotation.name for n in nodes} == {"DEG_0", "DEG_45", "DEG_90"}
    assert {n.copy_limit for n in nodes} == set(
        range(1, max_copy_limit(30) + 1)
    )
    assert {c.reflect for c in connections} == {True, False}
    assert {c.terminal_only for c in connections} == {True, False}
    assert {c.face.name for c in connections} >= {
        "FRONT",
        "BACK",
        "LEFT",
        "RIGHT",
        "TOP",
        "BOTTOM",
    }


@given(st.integers(min_value=0, max_value=2**32 - 1))
def test_random_genomes_build(seed: int) -> None:
    """A random genome unfolds into a body the builder compiles."""
    genome = random_genome(np.random.default_rng(seed))
    graph = unfold(genome)
    assert len(graph) >= MIN_MODULES
    assert nx.is_arborescence(graph)
    compile_model(construct_mjspec_from_graph(graph).spec)
