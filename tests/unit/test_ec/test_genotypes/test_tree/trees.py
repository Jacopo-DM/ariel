"""Small tree genomes and constants shared by the tree genotype tests."""

from ariel.ec.genotypes.tree.tree_genome import (
    TreeEdge,
    TreeGenome,
    TreeNode,
)

BRICK_ID = 2
CORE_HINGE_NODE_COUNT = 2
MAX_MODULES = 4
BRANCH_ID = 1
KEPT_ID = 2
SIBLING_ID = 3
MAX_NODES = MAX_MODULES + 1
SINGLE_MODULE_MAX_NODES = 2
CHAIN_DEPTH = 2
ORPHAN_ID = 5
ORPHAN_CHILD_ID = 6
BAD_FACE_ID = 3
TAKEN_FACE_ID = 4


def core_only() -> TreeGenome:
    g = TreeGenome()
    g.nodes = {0: TreeNode(type="CORE", rotation="DEG_0")}
    g.edges = []
    return g


def core_hinge() -> TreeGenome:
    g = TreeGenome()
    g.nodes = {
        0: TreeNode(type="CORE", rotation="DEG_0"),
        1: TreeNode(type="HINGE", rotation="DEG_0"),
    }
    g.edges = [TreeEdge(parent=0, child=1, face="FRONT")]
    return g


def core_hinge_brick() -> TreeGenome:
    g = TreeGenome()
    g.nodes = {
        0: TreeNode(type="CORE", rotation="DEG_0"),
        1: TreeNode(type="HINGE", rotation="DEG_0"),
        2: TreeNode(type="BRICK", rotation="DEG_0"),
    }
    g.edges = [
        TreeEdge(parent=0, child=1, face="FRONT"),
        TreeEdge(parent=1, child=2, face="FRONT"),
    ]
    return g


def orphan_genome() -> TreeGenome:
    # A core, plus a hinge with a brick child that is not connected to it.
    g = core_only()
    g.nodes[ORPHAN_ID] = TreeNode(type="HINGE", rotation="DEG_0")
    g.nodes[ORPHAN_CHILD_ID] = TreeNode(type="BRICK", rotation="DEG_0")
    g.edges = [TreeEdge(parent=ORPHAN_ID, child=ORPHAN_CHILD_ID, face="FRONT")]
    return g


def reject_everything(_genome: object) -> None:
    msg = "rejected"
    raise ValueError(msg)


def brick_with_two_children() -> TreeGenome:
    return TreeGenome(
        nodes={
            0: TreeNode(type="CORE", rotation="DEG_0"),
            1: TreeNode(type="BRICK", rotation="DEG_0"),
            2: TreeNode(type="BRICK", rotation="DEG_0"),
            3: TreeNode(type="BRICK", rotation="DEG_0"),
        },
        edges=[
            TreeEdge(parent=0, child=1, face="FRONT"),
            TreeEdge(parent=1, child=2, face="FRONT"),
            TreeEdge(parent=1, child=3, face="LEFT"),
        ],
    )
