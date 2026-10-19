"""Test: HighProbabilityDecoder and VectorDecoder."""

import networkx as nx
import numpy as np
import pytest
from networkx import DiGraph
from numpy.typing import NDArray

from ariel.body_phenotypes.robogen_lite.config import (
    ALLOWED_FACES,
    ALLOWED_ROTATIONS,
    NUM_OF_FACES,
    NUM_OF_ROTATIONS,
    NUM_OF_TYPES_OF_MODULES,
    ModuleRotationsIdx,
    ModuleType,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
)
from ariel.body_phenotypes.robogen_lite.decoders.vector_decoding import (
    VectorDecoder,
)

NUM_SYMBOLS = 3

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

NUM_MODULES = 4


def _type_space(num_modules: int = NUM_MODULES) -> NDArray[np.float32]:
    """Type probability space with core forced at index 0, no NONE elsewhere.

    Returns
    -------
    NDArray[np.float32]
        The probability space.
    """
    space = np.ones((num_modules, NUM_OF_TYPES_OF_MODULES), dtype=np.float32)
    space[0] = 0.0
    space[0, ModuleType.CORE.value] = 1.0
    for i in range(1, num_modules):
        space[i, ModuleType.NONE.value] = 0.0
        space[i, ModuleType.CORE.value] = 0.0
    return space


def _conn_space(num_modules: int = NUM_MODULES) -> NDArray[np.float32]:
    return np.ones((num_modules, num_modules, NUM_OF_FACES), dtype=np.float32)


def _rot_space(num_modules: int = NUM_MODULES) -> NDArray[np.float32]:
    return np.ones((num_modules, NUM_OF_ROTATIONS), dtype=np.float32)


# ---------------------------------------------------------------------------
# HighProbabilityDecoder — initialization
# ---------------------------------------------------------------------------


def test_hi_prob_decoder_initialization() -> None:
    """HighProbabilityDecoder initializes with the given num_modules."""
    decoder = HighProbabilityDecoder(num_modules=NUM_MODULES)
    assert decoder.num_modules == NUM_MODULES


# ---------------------------------------------------------------------------
# HighProbabilityDecoder — probability_matrices_to_graph
# ---------------------------------------------------------------------------


def test_hi_prob_decoder_returns_digraph() -> None:
    """probability_matrices_to_graph returns a NetworkX DiGraph."""
    decoder = HighProbabilityDecoder(num_modules=NUM_MODULES)
    graph = decoder.probability_matrices_to_graph(
        _type_space(),
        _conn_space(),
        _rot_space(),
    )
    assert isinstance(graph, DiGraph)


def test_hi_prob_decoder_core_always_present() -> None:
    """The core node (index 0) always appears in the decoded graph."""
    decoder = HighProbabilityDecoder(num_modules=NUM_MODULES)
    graph = decoder.probability_matrices_to_graph(
        _type_space(),
        _conn_space(),
        _rot_space(),
    )
    assert 0 in graph.nodes


def test_hi_prob_decoder_core_type_correct() -> None:
    """The core node carries type='CORE'."""
    decoder = HighProbabilityDecoder(num_modules=NUM_MODULES)
    graph = decoder.probability_matrices_to_graph(
        _type_space(),
        _conn_space(),
        _rot_space(),
    )
    assert graph.nodes[0]["type"] == "CORE"


def test_hi_prob_decoder_nodes_have_rotation() -> None:
    """Every node carries a 'rotation' attribute."""
    decoder = HighProbabilityDecoder(num_modules=NUM_MODULES)
    graph = decoder.probability_matrices_to_graph(
        _type_space(),
        _conn_space(),
        _rot_space(),
    )
    for node in graph.nodes:
        assert "rotation" in graph.nodes[node]


def test_hi_prob_decoder_no_self_loops() -> None:
    """The decoded graph has no self-loops."""
    decoder = HighProbabilityDecoder(num_modules=NUM_MODULES)
    graph = decoder.probability_matrices_to_graph(
        _type_space(),
        _conn_space(),
        _rot_space(),
    )
    assert not any(u == v for u, v in graph.edges)


def test_hi_prob_decoder_zero_conn_stops_gracefully() -> None:
    """All-zero connection space produces a graph with at least the core."""
    decoder = HighProbabilityDecoder(num_modules=NUM_MODULES)
    zero_conn = np.zeros(
        (NUM_MODULES, NUM_MODULES, NUM_OF_FACES), dtype=np.float32
    )
    graph = decoder.probability_matrices_to_graph(
        _type_space(),
        zero_conn,
        _rot_space(),
    )
    assert 0 in graph.nodes


def test_hi_prob_decoder_single_module() -> None:
    """Works correctly when num_modules=1 (core only)."""
    n = 1
    type_s = np.zeros((1, NUM_OF_TYPES_OF_MODULES), dtype=np.float32)
    type_s[0, ModuleType.CORE.value] = 1.0
    conn_s = np.zeros((1, 1, NUM_OF_FACES), dtype=np.float32)
    rot_s = np.ones((1, NUM_OF_ROTATIONS), dtype=np.float32)
    decoder = HighProbabilityDecoder(num_modules=n)
    graph = decoder.probability_matrices_to_graph(type_s, conn_s, rot_s)
    assert 0 in graph.nodes


# ---------------------------------------------------------------------------
# HighProbabilityDecoder — apply_connection_constraints
# ---------------------------------------------------------------------------


def test_hi_prob_apply_constraints_removes_self_connections() -> None:
    """After apply_connection_constraints, diagonal conn entries are zero."""
    decoder = HighProbabilityDecoder(num_modules=NUM_MODULES)
    decoder.conn_p_space = _conn_space()
    decoder.rot_p_space = _rot_space()
    decoder.type_p_space = _type_space()
    decoder.apply_connection_constraints()
    for face_idx in range(NUM_OF_FACES):
        for i in range(NUM_MODULES):
            assert decoder.conn_p_space[i, i, face_idx] == 0


def test_hi_prob_apply_constraints_core_never_child() -> None:
    """After constraints, nothing can connect to core as a child."""
    decoder = HighProbabilityDecoder(num_modules=NUM_MODULES)
    decoder.conn_p_space = _conn_space()
    decoder.rot_p_space = _rot_space()
    decoder.type_p_space = _type_space()
    decoder.apply_connection_constraints()
    assert np.all(decoder.conn_p_space[:, 0, :] == 0)


# ---------------------------------------------------------------------------
# HighProbabilityDecoder — set_module_types_and_rotations
# ---------------------------------------------------------------------------


def test_hi_prob_set_types_creates_type_dict() -> None:
    """set_module_types_and_rotations builds type_dict for all modules."""
    decoder = HighProbabilityDecoder(num_modules=NUM_MODULES)
    decoder.conn_p_space = _conn_space()
    decoder.rot_p_space = _rot_space()
    decoder.type_p_space = _type_space()
    decoder.set_module_types_and_rotations()
    assert len(decoder.type_dict) == NUM_MODULES


def test_hi_prob_none_module_loses_all_its_connections() -> None:
    """A module whose best type is NONE can be neither parent nor child."""
    none_module = 1
    decoder = HighProbabilityDecoder(num_modules=NUM_MODULES)
    decoder.conn_p_space = _conn_space()
    decoder.rot_p_space = _rot_space()
    decoder.type_p_space = _type_space()
    decoder.type_p_space[none_module, ModuleType.NONE.value] = 2.0
    decoder.set_module_types_and_rotations()
    assert decoder.type_dict[none_module] == ModuleType.NONE
    assert not decoder.conn_p_space[none_module].any()
    assert not decoder.conn_p_space[:, none_module].any()


def test_hi_prob_set_types_creates_rot_dict() -> None:
    """set_module_types_and_rotations builds rot_dict for all modules."""
    decoder = HighProbabilityDecoder(num_modules=NUM_MODULES)
    decoder.conn_p_space = _conn_space()
    decoder.rot_p_space = _rot_space()
    decoder.type_p_space = _type_space()
    decoder.set_module_types_and_rotations()
    assert len(decoder.rot_dict) == NUM_MODULES
    for v in decoder.rot_dict.values():
        assert isinstance(v, ModuleRotationsIdx)


# ---------------------------------------------------------------------------
# HighProbabilityDecoder — decode_probability_to_graph
# ---------------------------------------------------------------------------


def _setup_decoder(num_modules: int = NUM_MODULES) -> HighProbabilityDecoder:
    """Return a decoder fully initialized for decode_probability_to_graph.

    Returns
    -------
    HighProbabilityDecoder
        The decoder.
    """
    decoder = HighProbabilityDecoder(num_modules=num_modules)
    decoder.conn_p_space = _conn_space(num_modules)
    decoder.rot_p_space = _rot_space(num_modules)
    decoder.type_p_space = _type_space(num_modules)
    decoder.apply_connection_constraints()
    decoder.set_module_types_and_rotations()
    return decoder


def test_hi_prob_decode_builds_nodes_set() -> None:
    """decode_probability_to_graph populates the nodes set."""
    decoder = _setup_decoder()
    decoder.decode_probability_to_graph()
    assert hasattr(decoder, "nodes")
    assert 0 in decoder.nodes


def test_hi_prob_decode_builds_edges_list() -> None:
    """decode_probability_to_graph populates the edges list."""
    decoder = _setup_decoder()
    decoder.decode_probability_to_graph()
    assert hasattr(decoder, "edges")
    assert isinstance(decoder.edges, list)


# ---------------------------------------------------------------------------
# VectorDecoder — initialization
# ---------------------------------------------------------------------------


def test_vector_decoder_initialization() -> None:
    """VectorDecoder initializes with the given num_modules."""
    decoder = VectorDecoder(num_modules=NUM_MODULES)
    assert decoder.num_modules == NUM_MODULES


# ---------------------------------------------------------------------------
# VectorDecoder — assign_symbols_from_range
# ---------------------------------------------------------------------------


def test_assign_symbols_basic() -> None:
    """assign_symbols_from_range maps values to the correct symbols."""
    decoder = VectorDecoder(num_modules=NUM_MODULES)
    result = decoder.assign_symbols_from_range(
        vector=[0.0, 0.5, 1.0],
        symbols=["A", "B", "C"],
    )
    assert isinstance(result, list)
    assert len(result) == NUM_SYMBOLS
    assert all(s in {"A", "B", "C"} for s in result)


def test_assign_symbols_out_of_range_raises() -> None:
    """Values outside [0, 1] raise ValueError."""
    decoder = VectorDecoder(num_modules=NUM_MODULES)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        decoder.assign_symbols_from_range(vector=[1.5], symbols=["A", "B"])


def test_assign_symbols_negative_raises() -> None:
    """Negative values raise ValueError."""
    decoder = VectorDecoder(num_modules=NUM_MODULES)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        decoder.assign_symbols_from_range(vector=[-0.1], symbols=["A", "B"])


def test_assign_symbols_single_symbol_always_returned() -> None:
    """Single-symbol list is always returned regardless of value."""
    decoder = VectorDecoder(num_modules=NUM_MODULES)
    result = decoder.assign_symbols_from_range(
        vector=[0.0, 0.5, 1.0],
        symbols=["X"],
    )
    assert all(s == "X" for s in result)


def test_assign_symbols_boundary_zero() -> None:
    """Value 0.0 maps to the first symbol."""
    decoder = VectorDecoder(num_modules=NUM_MODULES)
    result = decoder.assign_symbols_from_range(
        vector=[0.0],
        symbols=["FIRST", "SECOND", "THIRD"],
    )
    assert result[0] == "FIRST"


def test_assign_symbols_boundary_one() -> None:
    """Value 1.0 maps to the last symbol."""
    decoder = VectorDecoder(num_modules=NUM_MODULES)
    result = decoder.assign_symbols_from_range(
        vector=[1.0],
        symbols=["FIRST", "SECOND", "THIRD"],
    )
    assert result[0] == "THIRD"


def test_assign_symbols_per_element_symbols() -> None:
    """Per-element symbol lists assign from the correct per-element set."""
    decoder = VectorDecoder(num_modules=NUM_MODULES)
    result: list[str] = decoder.assign_symbols_from_range(
        vector=[0.2, 0.8],
        symbols=[["A", "B"], ["C", "D"]],
    )
    assert result[0] in {"A", "B"}
    assert result[1] in {"C", "D"}


def test_assign_symbols_with_weights() -> None:
    """Uniform weights are equivalent to no weights."""
    decoder = VectorDecoder(num_modules=NUM_MODULES)
    vector = [0.3, 0.7]
    symbols = ["X", "Y", "Z"]
    result_no_weight = decoder.assign_symbols_from_range(
        vector=vector, symbols=symbols
    )
    result_uniform = decoder.assign_symbols_from_range(
        vector=vector,
        symbols=symbols,
        weight=[1.0, 1.0, 1.0],
    )
    assert result_no_weight == result_uniform


def test_assign_symbols_numpy_input() -> None:
    """assign_symbols_from_range accepts numpy arrays as vector input."""
    decoder = VectorDecoder(num_modules=NUM_MODULES)
    vector = np.array([0.1, 0.5, 0.9])
    result = decoder.assign_symbols_from_range(
        vector=vector, symbols=["A", "B", "C"]
    )
    assert len(result) == NUM_SYMBOLS


def test_assign_symbols_integer_symbols() -> None:
    """Integer symbols can be used as the symbol set."""
    decoder = VectorDecoder(num_modules=NUM_MODULES)
    result = decoder.assign_symbols_from_range(
        vector=[0.0, 0.5, 1.0],
        symbols=[0, 1, 2],
    )
    assert all(s in {0, 1, 2} for s in result)


# ---------------------------------------------------------------------------
# Decoding errors: the guards against corrupt probability spaces
# ---------------------------------------------------------------------------

BIG = 5.0


def test_hi_prob_decode_core_as_child_raises() -> None:
    """A connection that makes the core a child is an error."""
    decoder = _setup_decoder()
    decoder.conn_p_space[0, 0, 0] = BIG
    with pytest.raises(ValueError, match="core module as a child"):
        decoder.decode_probability_to_graph()


def test_hi_prob_decode_none_child_raises() -> None:
    """A NONE module cannot become a child."""
    decoder = _setup_decoder()
    decoder.type_dict[1] = ModuleType.NONE
    decoder.conn_p_space[0, 1, 0] = BIG
    with pytest.raises(ValueError, match="NONE module"):
        decoder.decode_probability_to_graph()


def test_hi_prob_decode_none_parent_raises() -> None:
    """A NONE module cannot be a parent."""
    decoder = _setup_decoder()
    decoder.type_dict[0] = ModuleType.NONE
    with pytest.raises(ValueError, match="NONE module"):
        decoder.decode_probability_to_graph()


# ---------------------------------------------------------------------------
# VectorDecoder — vectors_to_graph
# ---------------------------------------------------------------------------

BRICK_VALUE = 0.2  # first of three types: BRICK, HINGE, NONE
HINGE_VALUE = 0.6
NONE_VALUE = 0.9
FIRST_SLOT = 0.0
LAST_SLOT = 1.0
VECTOR_MODULES = 4
FIRST_NODES = 3


def _vectors(
    types: list[float], connections: list[float], rotations: list[float]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    return (
        np.array(types, dtype=np.float64),
        np.array(connections, dtype=np.float64),
        np.array(rotations, dtype=np.float64),
    )


def test_vectors_to_graph_builds_the_documented_example() -> None:
    """Brick, then NONE (left out), then hinge, all on the first free slot."""
    decoder = VectorDecoder(num_modules=VECTOR_MODULES)
    graph = decoder.vectors_to_graph(
        *_vectors(
            [BRICK_VALUE, NONE_VALUE, HINGE_VALUE],
            [FIRST_SLOT] * 3,
            [FIRST_SLOT] * 3,
        )
    )
    assert isinstance(graph, DiGraph)
    assert set(graph.nodes) == {0, 1, 3}
    assert graph.nodes[0]["type"] == "CORE"
    assert graph.nodes[1]["type"] == "BRICK"
    assert graph.nodes[3]["type"] == "HINGE"
    # both on the core: the first two free slots are the core's two first faces
    assert [parent for parent, _ in graph.edges] == [0, 0]
    faces = [graph.edges[edge]["face"] for edge in graph.edges]
    assert len(set(faces)) == len(faces)


def test_vectors_to_graph_makes_a_tree() -> None:
    decoder = VectorDecoder(num_modules=VECTOR_MODULES)
    graph = decoder.vectors_to_graph(
        *_vectors([BRICK_VALUE] * 3, [LAST_SLOT] * 3, [FIRST_SLOT] * 3)
    )
    assert len(graph.nodes) == VECTOR_MODULES
    assert len(graph.edges) == VECTOR_MODULES - 1
    # the last free slot of the placed modules is used each time: module 1 on
    # the core, module 2 on module 1, module 3 on module 2
    assert sorted(graph.edges) == [(0, 1), (1, 2), (2, 3)]


def test_vectors_to_graph_rotation_is_limited_by_the_type() -> None:
    decoder = VectorDecoder(num_modules=3)
    graph = decoder.vectors_to_graph(
        *_vectors([BRICK_VALUE, HINGE_VALUE], [FIRST_SLOT] * 2, [LAST_SLOT] * 2)
    )
    allowed = {r.name for r in ALLOWED_ROTATIONS[ModuleType.BRICK]}
    assert graph.nodes[1]["rotation"] in allowed
    assert graph.nodes[1]["rotation"] == ModuleRotationsIdx.DEG_90.name


def test_vectors_to_graph_always_finds_a_free_slot() -> None:
    """Hinges offer one face each, as many as they use: no module is lost."""
    modules = len(ALLOWED_FACES[ModuleType.CORE]) + 3
    decoder = VectorDecoder(num_modules=modules)
    count = modules - 1
    graph = decoder.vectors_to_graph(
        *_vectors(
            [HINGE_VALUE] * count, [FIRST_SLOT] * count, [FIRST_SLOT] * count
        )
    )
    assert len(graph.nodes) == modules
    assert nx.is_tree(graph.to_undirected())


def test_vectors_to_graph_with_only_none_modules_is_the_core() -> None:
    decoder = VectorDecoder(num_modules=3)
    graph = decoder.vectors_to_graph(
        *_vectors([NONE_VALUE] * 2, [FIRST_SLOT] * 2, [FIRST_SLOT] * 2)
    )
    assert set(graph.nodes) == {0}
    assert not graph.edges


def test_vectors_to_graph_with_a_single_module_is_the_core() -> None:
    decoder = VectorDecoder(num_modules=1)
    graph = decoder.vectors_to_graph(*_vectors([], [], []))
    assert set(graph.nodes) == {0}


def test_vectors_to_graph_refuses_a_vector_of_the_wrong_length() -> None:
    decoder = VectorDecoder(num_modules=VECTOR_MODULES)
    with pytest.raises(ValueError, match="connection vector must hold 3"):
        decoder.vectors_to_graph(
            *_vectors([BRICK_VALUE] * 3, [FIRST_SLOT] * 2, [FIRST_SLOT] * 3)
        )


def test_vectors_to_graph_refuses_values_outside_the_unit_interval() -> None:
    decoder = VectorDecoder(num_modules=3)
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        decoder.vectors_to_graph(
            *_vectors([BRICK_VALUE, 1.5], [FIRST_SLOT] * 2, [FIRST_SLOT] * 2)
        )


def test_vectors_to_graph_resets_between_calls() -> None:
    decoder = VectorDecoder(num_modules=3)
    first = decoder.vectors_to_graph(
        *_vectors([BRICK_VALUE] * 2, [FIRST_SLOT] * 2, [FIRST_SLOT] * 2)
    )
    second = decoder.vectors_to_graph(
        *_vectors([NONE_VALUE] * 2, [FIRST_SLOT] * 2, [FIRST_SLOT] * 2)
    )
    assert len(first.nodes) == FIRST_NODES
    assert len(second.nodes) == 1
