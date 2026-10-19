"""Test: mutation operators of tree genomes."""

import copy
import random
from collections.abc import Callable

import pytest

from ariel.ec.genotypes.tree import (
    mutate_hoist,
    mutate_replace_node,
    mutate_shrink,
    mutate_subtree_replacement,
    mutations,
    random_tree,
)
from ariel.ec.genotypes.tree.tree_genome import (
    TreeEdge,
    TreeGenome,
    TreeNode,
)
from ariel.ec.genotypes.tree.validation import validate_genome_dict
from tests.unit.test_ec.test_genotypes.test_tree.trees import (
    BRANCH_ID,
    KEPT_ID,
    ORPHAN_CHILD_ID,
    ORPHAN_ID,
    SIBLING_ID,
    brick_with_two_children,
    core_hinge,
    core_hinge_brick,
    core_only,
    orphan_genome,
    reject_everything,
)


def test_mutate_replace_node_preserves_core() -> None:
    """mutate_replace_node never changes the core node."""
    g = core_hinge()
    for _ in range(20):
        mutate_replace_node(g)
    assert g.nodes[0].type == "CORE"


def test_mutate_replace_node_core_only_is_noop() -> None:
    """mutate_replace_node on a core-only genome is a no-op."""
    g = core_only()
    original_nodes = copy.deepcopy(g.nodes)
    mutate_replace_node(g)
    assert g.nodes == original_nodes


def test_mutate_replace_node_result_is_valid() -> None:
    """After mutation the genome remains valid."""
    g = random_tree(5)
    for _ in range(10):
        mutate_replace_node(g)
    validate_genome_dict(g.to_dict())


def test_mutate_shrink_reduces_or_keeps_size() -> None:
    """mutate_shrink does not increase the node count."""
    g = core_hinge_brick()
    original_size = len(g.nodes)
    mutate_shrink(g)
    assert len(g.nodes) <= original_size


def test_mutate_shrink_core_only_is_noop() -> None:
    """mutate_shrink on a core-only genome is a no-op."""
    g = core_only()
    mutate_shrink(g)
    assert len(g.nodes) == 1


def test_mutate_shrink_result_is_valid() -> None:
    """Genome remains valid after shrink mutation."""
    g = random_tree(6)
    mutate_shrink(g)
    validate_genome_dict(g.to_dict())


def test_mutate_hoist_core_only_is_noop() -> None:
    """mutate_hoist on a core-only genome is a no-op."""
    g = core_only()
    mutate_hoist(g)
    assert len(g.nodes) == 1


def test_mutate_hoist_result_is_valid() -> None:
    """Genome remains valid after hoist mutation."""
    g = random_tree(6)
    for _ in range(5):
        mutate_hoist(g)
        validate_genome_dict(g.to_dict())


def test_mutate_subtree_replacement_core_only_is_noop() -> None:
    """mutate_subtree_replacement on a core-only genome is a no-op."""
    g = core_only()
    mutate_subtree_replacement(g)
    assert len(g.nodes) == 1


def test_mutate_subtree_replacement_result_is_valid() -> None:
    """Genome remains valid after subtree replacement mutation."""
    g = random_tree(5)
    for _ in range(5):
        mutate_subtree_replacement(g)
    validate_genome_dict(g.to_dict())


@pytest.mark.parametrize(
    "mutation",
    [
        mutate_replace_node,
        mutate_subtree_replacement,
        mutate_shrink,
        mutate_hoist,
    ],
)
def test_failed_validation_restores_the_genome(
    monkeypatch: pytest.MonkeyPatch,
    mutation: Callable[[TreeGenome], None],
) -> None:
    """A mutation that cannot be validated leaves the genome as it was."""
    monkeypatch.setattr(mutations, "validate_genome_dict", reject_everything)
    for seed in range(20):
        random.seed(seed)
        g = core_hinge_brick()
        before = g.to_dict()
        mutation(g)
        assert g.to_dict() == before


@pytest.mark.parametrize(
    "mutation",
    [
        mutate_replace_node,
        mutate_subtree_replacement,
        mutate_shrink,
        mutate_hoist,
    ],
)
def test_mutations_leave_a_core_only_genome_alone(
    mutation: Callable[[TreeGenome], None],
) -> None:
    """With only the core there is nothing to mutate."""
    g = core_only()
    mutation(g)
    assert list(g.nodes) == [0]


def test_shrink_and_hoist_skip_nodes_without_a_parent() -> None:
    """A node cut off from the core cannot be shrunk or hoisted."""
    shrunk = orphan_genome()
    shrunk.nodes.pop(ORPHAN_CHILD_ID)
    shrunk.edges = []
    mutate_shrink(shrunk)
    assert ORPHAN_ID in shrunk.nodes
    hoisted = orphan_genome()
    mutate_hoist(hoisted)
    assert ORPHAN_ID in hoisted.nodes
    assert ORPHAN_CHILD_ID in hoisted.nodes


def test_hoist_skips_a_missing_parent_node() -> None:
    """An edge that points at a missing node is dropped by the hoist."""
    g = TreeGenome(
        nodes={
            0: TreeNode(type="CORE", rotation="DEG_0"),
            ORPHAN_CHILD_ID: TreeNode(type="BRICK", rotation="DEG_0"),
        },
        edges=[
            TreeEdge(parent=0, child=ORPHAN_ID, face="FRONT"),
            TreeEdge(parent=ORPHAN_ID, child=ORPHAN_CHILD_ID, face="FRONT"),
        ],
    )
    mutate_hoist(g)
    assert ORPHAN_ID not in g.nodes


def test_subtree_replacement_of_a_detached_node_adds_nothing() -> None:
    """A node without a parent edge is removed and nothing is grafted."""
    g = orphan_genome()
    g.nodes.pop(ORPHAN_CHILD_ID)
    g.edges = []
    random.seed(0)
    mutate_subtree_replacement(g)
    assert list(g.nodes) == [0]


def test_replace_node_drops_children_on_faces_the_new_type_lacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hinge only has a FRONT face, so a LEFT child of the node goes."""
    g = brick_with_two_children()

    def choose(options: list[object]) -> object:
        if all(isinstance(o, int) for o in options):
            return BRANCH_ID  # the node to replace
        return next(
            (o for o in options if getattr(o, "name", "") == "HINGE"),
            options[0],
        )

    monkeypatch.setattr(random, "choice", choose)
    mutate_replace_node(g)
    assert g.nodes[BRANCH_ID].type == "HINGE"
    assert SIBLING_ID not in g.nodes
    assert KEPT_ID in g.nodes


def test_hoist_removes_the_other_children_of_the_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hoisting child 2 removes its sibling 3 together with the parent."""
    g = brick_with_two_children()

    def choose(options: list[int]) -> int:
        return options[0] if len(options) == 1 else KEPT_ID

    monkeypatch.setattr(random, "choice", choose)
    mutate_hoist(g)
    assert SIBLING_ID not in g.nodes
    assert BRANCH_ID not in g.nodes
    assert KEPT_ID in g.nodes
