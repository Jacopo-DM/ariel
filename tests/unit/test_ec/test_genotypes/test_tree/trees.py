"""Small tree genomes, strategies and constants for the tree genome tests."""

from hypothesis import strategies as st

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


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Real enum member names and junk, so generated dicts reach every check
NAMES = st.sampled_from([
    "CORE",
    "BRICK",
    "HINGE",
    "NONE",
    "DEG_0",
    "DEG_45",
    "DEG_90",
    "FRONT",
    "BACK",
    "LEFT",
    "RIGHT",
    "TOP",
    "BOTTOM",
]) | st.text(max_size=5)
IDS = st.integers(min_value=-1, max_value=6)
JSON_VALUES: st.SearchStrategy[object] = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False)
    | st.text(max_size=5),
    lambda inner: (
        st.lists(inner, max_size=3)
        | st.dictionaries(st.text(max_size=5), inner, max_size=3)
    ),
    max_leaves=8,
)
RAW_NODES = st.fixed_dictionaries(
    {"type": NAMES, "rotation": NAMES},
    optional={"id": IDS, "extra": st.text(max_size=5)},
)
RAW_EDGES = st.fixed_dictionaries(
    {"parent": IDS, "child": IDS, "face": NAMES},
    optional={"note": st.text(max_size=5)},
)
# fixed_dictionaries takes its key and value types from this mapping
NO_REQUIRED_KEYS: dict[str, st.SearchStrategy[object]] = {}
# Dicts shaped like a stored genome, near misses of it, and any JSON object
GENOME_DICTS: st.SearchStrategy[dict[str, object]] = st.one_of(
    st.fixed_dictionaries(
        NO_REQUIRED_KEYS,
        optional={
            "nodes": st.dictionaries(IDS.map(str), RAW_NODES, max_size=5)
            | st.lists(RAW_NODES, max_size=5)
            | JSON_VALUES,
            "edges": st.lists(RAW_EDGES, max_size=5) | JSON_VALUES,
        },
    ),
    st.dictionaries(st.text(max_size=5), JSON_VALUES, max_size=3),
)
GENOMES = st.builds(
    TreeGenome,
    nodes=st.dictionaries(
        st.integers(),
        st.builds(TreeNode, type=st.text(), rotation=st.text()),
        max_size=5,
    ),
    edges=st.lists(
        st.builds(
            TreeEdge, parent=st.integers(), child=st.integers(), face=st.text()
        ),
        max_size=5,
    ),
)
