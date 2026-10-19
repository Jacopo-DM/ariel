"""Test: subtree crossover of tree genomes."""

import copy
import random

import pytest

from ariel.ec.genotypes.tree import (
    crossover,
    crossover_subtree,
    random_tree,
    subtree_swap,
)
from ariel.ec.genotypes.tree.tree_genome import (
    TreeEdge,
    TreeGenome,
    TreeNode,
)
from ariel.ec.genotypes.tree.validation import validate_genome_dict
from tests.unit.test_ec.test_genotypes.test_tree.trees import (
    core_hinge_brick,
    core_only,
    reject_everything,
)


def test_subtree_swap_core_preserved_in_both() -> None:
    """Both genomes still contain their core nodes after a subtree swap."""
    a = random_tree(4)
    b = random_tree(4)
    # pick a non-core node from each
    a_nid = next(n for n in a.nodes if n != 0)
    b_nid = next(n for n in b.nodes if n != 0)
    subtree_swap(a, b, a_nid, b_nid)
    assert 0 in a.nodes
    assert 0 in b.nodes


def test_crossover_subtree_returns_two_genomes() -> None:
    """crossover_subtree always returns a pair of TreeGenome objects."""
    a = random_tree(5)
    b = random_tree(5)
    c1, c2 = crossover_subtree(a, b)
    assert isinstance(c1, TreeGenome)
    assert isinstance(c2, TreeGenome)


def test_crossover_subtree_children_valid() -> None:
    """Children produced by crossover are valid genomes."""
    for _ in range(10):
        a = random_tree(5)
        b = random_tree(5)
        c1, c2 = crossover_subtree(a, b)
        validate_genome_dict(c1.to_dict())
        validate_genome_dict(c2.to_dict())


def test_crossover_subtree_parents_unchanged() -> None:
    """crossover_subtree does not modify the parent genomes."""
    a = random_tree(4)
    b = random_tree(4)
    a_nodes_before = copy.deepcopy(a.nodes)
    b_nodes_before = copy.deepcopy(b.nodes)
    crossover_subtree(a, b)
    assert a.nodes == a_nodes_before
    assert b.nodes == b_nodes_before


def test_crossover_subtree_core_only_returns_copies() -> None:
    """Core-only genomes (no candidates) return deep copies of parents."""
    a = core_only()
    b = core_only()
    c1, c2 = crossover_subtree(a, b)
    assert 0 in c1.nodes
    assert 0 in c2.nodes


def test_crossover_of_core_only_genomes_returns_copies() -> None:
    """With no non-core node to swap the children equal the parents."""
    a, b = core_only(), core_only()
    c1, c2 = crossover_subtree(a, b)
    assert c1 is not a
    assert c1.nodes.keys() == a.nodes.keys()
    assert c2.nodes.keys() == b.nodes.keys()


def test_crossover_falls_back_to_parent_copies_when_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If validation rejects the children the parents are copied."""
    monkeypatch.setattr(crossover, "validate_genome_dict", reject_everything)
    a, b = core_hinge_brick(), core_hinge_brick()
    c1, c2 = crossover_subtree(a, b)
    assert c1.to_dict() == a.to_dict()
    assert c2.to_dict() == b.to_dict()


def test_crossover_returns_swapped_children_when_validation_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With validation stubbed out the children swap their subtrees."""
    monkeypatch.setattr(crossover, "validate_genome_dict", lambda _genome: None)
    a = TreeGenome(
        nodes={
            0: TreeNode(type="CORE", rotation="DEG_0"),
            1: TreeNode(type="BRICK", rotation="DEG_0"),
        },
        edges=[TreeEdge(parent=0, child=1, face="FRONT")],
    )
    b = copy.deepcopy(a)
    b.nodes[1].type = "HINGE"
    c1, c2 = crossover_subtree(a, b)
    assert c1.nodes[1].type == "HINGE"
    assert c2.nodes[1].type == "BRICK"
    assert a.nodes[1].type == "BRICK"


def test_subtree_swap_reattaches_each_subtree_at_the_old_slot() -> None:
    """A swapped subtree hangs from the parent and face the other one left."""
    a = core_hinge_brick()
    b = core_hinge_brick()
    b.nodes[1].type = "BRICK"
    b.nodes[1].rotation = "DEG_90"
    a_slot = next(e for e in a.edges if e.child == 1)
    subtree_swap(a, b, 1, 1)
    validate_genome_dict(a.to_dict())
    validate_genome_dict(b.to_dict())
    new_edge = next(e for e in a.edges if e.parent == a_slot.parent)
    assert new_edge.face == a_slot.face
    assert a.nodes[new_edge.child].rotation == "DEG_90"


def test_crossover_of_valid_parents_gives_valid_children() -> None:
    """Swapping real subtrees passes validation and keeps the parents."""
    random.seed(11)
    a = core_hinge_brick()
    b = core_hinge_brick()
    c1, c2 = crossover_subtree(a, b)
    validate_genome_dict(c1.to_dict())
    validate_genome_dict(c2.to_dict())
    assert a.to_dict() == core_hinge_brick().to_dict()


def test_crossover_1000_pairs_validate_and_often_differ() -> None:
    """1000 random pairs: every child validates; many differ from parents."""
    random.seed(42)
    sizes = random.Random(7)  # ruff: ignore[suspicious-non-cryptographic-random-usage]  # test data
    differing = 0
    for _ in range(1000):
        a = random_tree(sizes.randint(2, 8))
        b = random_tree(sizes.randint(2, 8))
        c1, c2 = crossover_subtree(a, b)
        validate_genome_dict(c1.to_dict())
        validate_genome_dict(c2.to_dict())
        if c1.to_dict() != a.to_dict() or c2.to_dict() != b.to_dict():
            differing += 1
    assert differing > 0


def test_subtree_swap_of_parentless_roots_attaches_nothing() -> None:
    """A subtree root with no incoming edge has no slot to refill."""
    a = core_hinge_brick()
    b = core_hinge_brick()
    a.edges = []
    b.edges = []
    subtree_swap(a, b, 1, 1)
    assert 0 in a.nodes
    assert 0 in b.nodes
    assert not a.edges
    assert not b.edges
