"""Test: robot builders, blueprint IO, the module base class, best-first."""

from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import mujoco
import networkx as nx
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from ariel.body_phenotypes.robogen_lite import constructor
from ariel.body_phenotypes.robogen_lite.config import (
    NUM_OF_ROTATIONS,
    NUM_OF_TYPES_OF_MODULES,
    ModuleFaces,
    ModuleType,
)
from ariel.body_phenotypes.robogen_lite.cppn_neat.genome import Genome
from ariel.body_phenotypes.robogen_lite.decoders import (
    cppn_best_first,
    draw_graph,
    load_graph_from_json,
    save_graph_as_json,
)
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule
from ariel.body_phenotypes.robogen_lite.modules.module import Module
from ariel.simulators import compile_model
from tests.unit.misuse import untyped

mpl.use("Agg")

NUM_CPPN_INPUTS = 6
NUM_CPPN_OUTPUTS = 1 + NUM_OF_TYPES_OF_MODULES + NUM_OF_ROTATIONS
MAX_MODULES = 8
BODIES_FOR_FACES = 20
ONE_SECOND = 1.0


def _branching_graph() -> nx.DiGraph[int]:
    # Four nodes: a two-node graph has a degenerate layout in networkx.
    graph: nx.DiGraph[int] = nx.DiGraph()
    graph.add_node(0, type="CORE", rotation="DEG_0")
    graph.add_node(1, type="HINGE", rotation="DEG_0")
    graph.add_node(2, type="BRICK", rotation="DEG_0")
    graph.add_node(3, type="HINGE", rotation="DEG_0")
    graph.add_edge(0, 1, face="FRONT")
    graph.add_edge(0, 2, face="LEFT")
    graph.add_edge(2, 3, face="FRONT")
    return graph


def _small_graph() -> nx.DiGraph[int]:
    # Core with a hinge on its FRONT face.
    graph: nx.DiGraph[int] = nx.DiGraph()
    graph.add_node(0, type="CORE", rotation="DEG_0")
    graph.add_node(1, type="HINGE", rotation="DEG_0")
    graph.add_edge(0, 1, face="FRONT")
    return graph


# --- blueprint IO and drawing ----------------------------------------------- #


def test_save_graph_without_a_path_does_nothing(tmp_path: Path) -> None:
    """save_graph_as_json(None) is a no-op."""
    save_graph_as_json(_small_graph(), None)
    assert list(tmp_path.iterdir()) == []


def test_graph_json_roundtrip(tmp_path: Path) -> None:
    """A saved graph loads back with its nodes, edges and attributes."""
    path = tmp_path / "graph.json"
    save_graph_as_json(_small_graph(), path)
    assert json.loads(path.read_text(encoding="utf-8"))["edges"]
    loaded = load_graph_from_json(path)
    assert set(loaded.nodes) == {0, 1}
    assert loaded.edges[0, 1]["face"] == "FRONT"
    assert loaded.nodes[1]["type"] == "HINGE"


GRAPH_NAMES = st.sampled_from(["CORE", "BRICK", "HINGE", "DEG_0", "FRONT"])


@st.composite
def graphs(draw: st.DrawFn) -> nx.DiGraph[int]:
    """Draw a directed graph with module attributes on nodes and edges.

    Returns
    -------
    nx.DiGraph[int]
        The drawn graph.
    """
    nodes = draw(st.lists(st.integers(), unique=True, max_size=5))
    graph: nx.DiGraph[int] = nx.DiGraph()
    for node in nodes:
        graph.add_node(
            node, type=draw(GRAPH_NAMES | st.text()), rotation=draw(st.text())
        )
    if nodes:
        pairs = st.tuples(st.sampled_from(nodes), st.sampled_from(nodes))
        for parent, child in draw(st.lists(pairs, max_size=5)):
            graph.add_edge(parent, child, face=draw(st.text()))
    return graph


@given(graphs())
def test_graph_json_file_round_trip(graph: nx.DiGraph[int]) -> None:
    """Loading a saved graph gives the same nodes, edges and attributes."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "graph.json"
        save_graph_as_json(graph, path)
        loaded = load_graph_from_json(path)
    assert dict(loaded.nodes.items()) == dict(graph.nodes.items())
    assert sorted(loaded.edges(data="face")) == sorted(graph.edges(data="face"))


GRAPH_FILE_VALUES = (
    st.none() | st.booleans() | st.integers() | st.text(max_size=3)
)
GRAPH_FILES = st.one_of(
    st.binary(),
    st.fixed_dictionaries(
        {},
        optional={
            "directed": st.booleans(),
            "multigraph": st.booleans(),
            "graph": st.dictionaries(st.text(max_size=3), GRAPH_FILE_VALUES),
            "nodes": st.lists(
                st.dictionaries(
                    st.sampled_from(["id", "type"]), GRAPH_FILE_VALUES
                ),
                max_size=3,
            )
            | GRAPH_FILE_VALUES,
            "edges": st.lists(
                st.dictionaries(
                    st.sampled_from(["source", "target", "face"]),
                    GRAPH_FILE_VALUES,
                ),
                max_size=3,
            )
            | GRAPH_FILE_VALUES,
        },
    ).map(lambda data: json.dumps(data).encode()),
)
VALID_GRAPH_FILE = (
    b'{"directed": true, "multigraph": false, "graph": {}, '
    b'"nodes": [{"id": 0, "type": "CORE"}, {"id": 1, "type": "HINGE"}], '
    b'"edges": [{"source": 0, "target": 1, "face": "FRONT"}]}'
)


@given(GRAPH_FILES)
@example(VALID_GRAPH_FILE)
@example(b"[]")
@example(b"{}")
@example(b'{"nodes": 1, "edges": []}')
@example(b'{"nodes": [{}], "edges": []}')
@example(b'{"directed": false, "nodes": [], "edges": []}')
@example(b'{"multigraph": true, "nodes": [], "edges": []}')
def test_load_graph_from_json_boundary(raw: bytes) -> None:
    """Any file content loads as a simple DiGraph or raises ValueError."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "graph.json"
        path.write_bytes(raw)
        try:
            graph = load_graph_from_json(path)
        except ValueError:
            return
    assert isinstance(graph, nx.DiGraph)
    assert not graph.is_multigraph()


def test_draw_graph_saves_an_image(tmp_path: Path) -> None:
    """A drawn graph is written to the given file."""
    path = tmp_path / "graph.png"
    draw_graph(_branching_graph(), save_file=path, dpi=50)
    assert path.stat().st_size > 0


def test_draw_graph_shows_when_no_file_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a file the figure is shown."""
    shown: list[bool] = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))
    draw_graph(_branching_graph())
    assert shown == [True]


# --- the module base class -------------------------------------------------- #


def test_module_subclass_must_define_required_attributes() -> None:
    """A subclass without `index` and `module_type` is rejected."""
    with pytest.raises(NotImplementedError, match="must define attribute"):
        type("_Incomplete", (Module,), {"rotate": lambda _self, _angle: None})


def test_module_rotate_default_raises() -> None:
    """The abstract rotate raises when a subclass calls it via super."""

    class _Complete(Module):
        index = 0
        module_type = ModuleType.BRICK

        def rotate(self, angle: float) -> None:
            untyped(super().rotate)(angle)  # the abstract method, on purpose

    with pytest.raises(NotImplementedError, match="does not implement"):
        _Complete().rotate(0.0)


# --- constructor ------------------------------------------------------------ #


def test_constructor_requires_a_core_at_the_root() -> None:
    """A graph whose root is not a core module is rejected."""
    graph: nx.DiGraph[int] = nx.DiGraph()
    graph.add_node(0, type="HINGE", rotation="DEG_0")
    with pytest.raises(ValueError, match="not of type CoreModule"):
        constructor.construct_mjspec_from_graph(graph)


def test_constructor_builds_core_with_children() -> None:
    """The small graph builds a core that compiles."""
    core = constructor.construct_mjspec_from_graph(_small_graph())
    assert isinstance(core, CoreModule)
    model = compile_model(core.spec)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    del core, model, data


# --- best-first decoder ----------------------------------------------------- #


def test_softmax_sums_to_one() -> None:
    """Softmax returns a probability vector."""
    probs = cppn_best_first.softmax(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert float(probs.sum()) == pytest.approx(1.0)
    assert probs[2] > probs[0]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_best_first_decoder_grows_a_valid_tree(seed: int) -> None:
    """The decoded graph is a tree rooted at the core with the budget used."""
    random.seed(seed)
    genome = Genome.random(
        NUM_CPPN_INPUTS,
        NUM_CPPN_OUTPUTS,
        NUM_CPPN_INPUTS + NUM_CPPN_OUTPUTS,
        0,
    )
    graph = cppn_best_first.MorphologyDecoderBestFirst(
        genome, max_modules=MAX_MODULES
    ).decode()
    assert len(graph) == MAX_MODULES
    assert nx.is_arborescence(graph)
    assert graph.nodes[0]["type"] == "CORE"
    core = constructor.construct_mjspec_from_graph(graph)
    del core


def test_best_first_decoder_stalls_without_free_faces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no face allowed anywhere only the core is returned."""
    monkeypatch.setattr(
        cppn_best_first,
        "ALLOWED_FACES",
        {module_type: [] for module_type in ModuleType},
    )
    genome = Genome.random(
        NUM_CPPN_INPUTS,
        NUM_CPPN_OUTPUTS,
        NUM_CPPN_INPUTS + NUM_CPPN_OUTPUTS,
        0,
    )
    graph = cppn_best_first.MorphologyDecoderBestFirst(
        genome, max_modules=MAX_MODULES
    ).decode()
    assert list(graph.nodes) == [0]


def test_best_first_decoder_stalls_without_allowed_rotations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A candidate whose rotation its type does not allow is dropped."""
    monkeypatch.setattr(
        cppn_best_first,
        "ALLOWED_ROTATIONS",
        {module_type: [] for module_type in ModuleType},
    )
    genome = Genome.random(
        NUM_CPPN_INPUTS,
        NUM_CPPN_OUTPUTS,
        NUM_CPPN_INPUTS + NUM_CPPN_OUTPUTS,
        0,
    )
    graph = cppn_best_first.MorphologyDecoderBestFirst(
        genome, max_modules=MAX_MODULES
    ).decode()
    assert list(graph.nodes) == [0]


def test_best_first_decoder_uses_every_face_of_the_parent() -> None:
    """The face is checked against the parent, so BACK and TOP are used."""
    random.seed(0)
    faces: set[str] = set()
    hinge_faces: set[str] = set()
    for _ in range(BODIES_FOR_FACES):
        genome = Genome.random(
            NUM_CPPN_INPUTS,
            NUM_CPPN_OUTPUTS,
            NUM_CPPN_INPUTS + NUM_CPPN_OUTPUTS,
            0,
        )
        graph = cppn_best_first.MorphologyDecoderBestFirst(
            genome, max_modules=MAX_MODULES
        ).decode()
        for parent, child in graph.edges:
            face = str(graph.edges[parent, child]["face"])
            faces.add(face)
            if graph.nodes[child]["type"] == "HINGE":
                hinge_faces.add(face)
    assert faces == {face.name for face in ModuleFaces}
    assert hinge_faces != {"FRONT"}


def test_brick_and_hinge_expose_sites() -> None:
    """Both submodules expose attachment sites that builders can use."""
    assert BrickModule(index=1).sites
    assert HingeModule(index=2).sites
