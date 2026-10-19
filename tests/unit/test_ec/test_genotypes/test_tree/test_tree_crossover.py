"""Test: subtree crossover of tree genomes."""

import copy

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
    """With validation stubbed out the children swap their subtrees.

    Known bug (PLAN.md): the swapped subtrees are not reattached, so real
    validation always fails and crossover returns copies of the parents.
    """
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


def test_crossover_of_valid_parents_returns_their_copies() -> None:
    """Known bug (PLAN.md): the children are always copies of the parents."""
    a = core_hinge_brick()
    b = core_hinge_brick()
    c1, c2 = crossover_subtree(a, b)
    assert c1.to_dict() == a.to_dict()
    assert c2.to_dict() == b.to_dict()
