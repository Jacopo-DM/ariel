"""Test: structure helpers (nodes, subtrees, ids, depth) of tree genomes."""

from ariel.ec.genotypes.tree import (
    add_node,
    get_top_ancestor,
    get_tree_depth,
    prune_invalid_edges,
    random_tree,
    remove_subtree,
    validate_tree_depth,
)
from ariel.ec.genotypes.tree.tree_genome import (
    TreeEdge,
    TreeGenome,
    TreeNode,
)
from ariel.ec.genotypes.tree.validation import validate_genome_dict
from tests.unit.test_ec.test_genotypes.test_tree.trees import (
    BAD_FACE_ID,
    BRICK_ID,
    CHAIN_DEPTH,
    CORE_HINGE_NODE_COUNT,
    MAX_MODULES,
    MAX_NODES,
    SINGLE_MODULE_MAX_NODES,
    TAKEN_FACE_ID,
    core_hinge,
    core_hinge_brick,
    core_only,
)


def test_add_node_appends_node() -> None:
    """add_node inserts a new node into the genome."""
    g = core_only()
    add_node(
        g, parent=0, face="FRONT", node_id=1, mtype="HINGE", rotation="DEG_0"
    )
    assert 1 in g.nodes
    assert g.nodes[1] == TreeNode(type="HINGE", rotation="DEG_0")


def test_add_node_appends_edge() -> None:
    """add_node appends a directed edge."""
    g = core_only()
    add_node(
        g, parent=0, face="BACK", node_id=1, mtype="BRICK", rotation="DEG_0"
    )
    assert TreeEdge(parent=0, child=1, face="BACK") in g.edges


def test_remove_subtree_removes_node_and_edge() -> None:
    """remove_subtree deletes the target node and its incoming edge."""
    g = core_hinge()
    remove_subtree(g, node_id=1)
    assert 1 not in g.nodes
    assert all(e.child != 1 for e in g.edges)


def test_remove_subtree_removes_descendants() -> None:
    """Removing a non-leaf removes all descendant nodes."""
    g = core_hinge_brick()
    remove_subtree(g, node_id=1)
    assert 1 not in g.nodes
    assert BRICK_ID not in g.nodes


def test_remove_subtree_nonexistent_is_noop() -> None:
    """Removing a node that does not exist is silently ignored."""
    g = core_hinge()
    remove_subtree(g, node_id=99)
    assert len(g.nodes) == CORE_HINGE_NODE_COUNT


def test_get_top_ancestor_direct_child_of_core() -> None:
    """Direct child of core returns itself as top ancestor."""
    g = core_hinge()
    assert get_top_ancestor(g, 1) == 1


def test_get_top_ancestor_grandchild() -> None:
    """Grandchild returns its parent (first child of core) as top ancestor."""
    g = core_hinge_brick()
    result = get_top_ancestor(g, 2)
    assert result == 1


def test_random_tree_has_core() -> None:
    """random_tree always contains the core node."""
    g = random_tree(5)
    assert 0 in g.nodes
    assert g.nodes[0].type == "CORE"


def test_random_tree_within_budget() -> None:
    """random_tree produces at most max_modules + 1 nodes (core + budget)."""
    g = random_tree(MAX_MODULES)
    assert len(g.nodes) <= MAX_NODES


def test_random_tree_is_valid() -> None:
    """random_tree produces a genome that passes validation."""
    g = random_tree(6)
    validate_genome_dict(g.to_dict())


def test_random_tree_single_module() -> None:
    """random_tree(1) produces at most 2 nodes."""
    g = random_tree(1)
    assert len(g.nodes) <= SINGLE_MODULE_MAX_NODES


def test_get_tree_depth_core_only() -> None:
    """Core-only tree has depth 0."""
    assert get_tree_depth(core_only()) == 0


def test_get_tree_depth_chain() -> None:
    """Core → hinge → brick has depth 2."""
    assert get_tree_depth(core_hinge_brick()) == CHAIN_DEPTH


def test_validate_tree_depth_within_limit() -> None:
    """Depth within limit returns True."""
    g = core_hinge_brick()
    assert validate_tree_depth(g, max_depth=5) is True


def test_validate_tree_depth_at_limit() -> None:
    """Depth exactly at limit returns True."""
    g = core_hinge_brick()
    assert validate_tree_depth(g, max_depth=2) is True


def test_validate_tree_depth_exceeds_limit() -> None:
    """Depth exceeding limit returns False."""
    g = core_hinge_brick()
    assert validate_tree_depth(g, max_depth=1) is False


def test_top_ancestor_of_the_core_is_the_core() -> None:
    """A node without a parent is its own top ancestor."""
    assert get_top_ancestor(core_hinge(), 0) == 0


def test_depth_of_an_empty_genome_is_zero() -> None:
    """No nodes means depth 0."""
    assert get_tree_depth(TreeGenome()) == 0


def test_pruning_drops_bad_edges_and_their_subtrees() -> None:
    """Edges to missing nodes, on disallowed faces, or on a taken face go."""
    g = core_hinge_brick()
    g.nodes[BAD_FACE_ID] = TreeNode(type="BRICK", rotation="DEG_0")
    g.nodes[TAKEN_FACE_ID] = TreeNode(type="BRICK", rotation="DEG_0")
    g.edges.extend([
        TreeEdge(
            parent=1, child=BAD_FACE_ID, face="BACK"
        ),  # hinge allows FRONT only
        TreeEdge(
            parent=0, child=TAKEN_FACE_ID, face="FRONT"
        ),  # core FRONT is taken
        TreeEdge(parent=0, child=99, face="LEFT"),  # child does not exist
    ])
    prune_invalid_edges(g)
    assert {(e.parent, e.child) for e in g.edges} == {(0, 1), (1, 2)}
    assert BAD_FACE_ID not in g.nodes
    assert TAKEN_FACE_ID not in g.nodes
