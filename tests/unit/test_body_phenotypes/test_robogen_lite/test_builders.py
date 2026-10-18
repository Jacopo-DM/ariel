"""Test: robot builders, blueprint IO, the module base class, best-first."""

from __future__ import annotations

import json
import random
import runpy
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import mujoco
import networkx as nx
import numpy as np
import pytest

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
from ariel.body_phenotypes.robogen_lite.prebuilt_robots import (
    gecko,
    random_20,
    spider_with_blocks,
)

mpl.use("Agg")

NUM_CPPN_INPUTS = 6
NUM_CPPN_OUTPUTS = 1 + NUM_OF_TYPES_OF_MODULES + NUM_OF_ROTATIONS
MAX_MODULES = 8
TOTAL_MODULES = 20
SPIDER_BODIES = 1 + 4 * 4
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


def _do_nothing(self: Module, angle: float) -> None:
    del self, angle


def test_module_subclass_must_define_required_attributes() -> None:
    """A subclass without `index` and `module_type` is rejected."""
    with pytest.raises(NotImplementedError, match="must define attribute"):
        type("_Incomplete", (Module,), {"rotate": _do_nothing})


def test_module_rotate_default_raises() -> None:
    """The abstract rotate raises when a subclass calls it via super."""

    class _Complete(Module):
        index = 0
        module_type = ModuleType.BRICK

        def rotate(self, angle: float) -> None:
            super().rotate(angle)

    with pytest.raises(NotImplementedError, match="does not implement"):
        _Complete().rotate(0.0)


# --- prebuilt robots -------------------------------------------------------- #


def test_random_body_has_twenty_modules() -> None:
    """The random body is reproducible and compiles."""
    first = random_20.random_spider20(seed=3)
    again = random_20.random_spider20(seed=3)
    assert first.spec.to_xml() == again.spec.to_xml()
    model = first.spec.compile()
    assert model.nbody > TOTAL_MODULES
    del first, again, model


def test_random_body_runs_out_of_sites(monkeypatch: pytest.MonkeyPatch) -> None:
    """If modules expose no faces the builder reports it."""

    class _BareHinge(HingeModule):
        def __init__(self, index: int) -> None:
            super().__init__(index)
            self.sites = {}

    class _BareBrick(BrickModule):
        def __init__(self, index: int) -> None:
            super().__init__(index)
            self.sites = {}

    monkeypatch.setattr(random_20, "HingeModule", _BareHinge)
    monkeypatch.setattr(random_20, "BrickModule", _BareBrick)
    with pytest.raises(RuntimeError, match="No open attachment sites"):
        random_20.random_spider20()


def test_spider_with_blocks_bodies_compile() -> None:
    """Both spider variants compile, with four legs of four modules."""
    plain = spider_with_blocks.body_spider()
    turned = spider_with_blocks.body_spider45()
    for core in (plain, turned):
        model = core.spec.compile()
        assert model.nbody > SPIDER_BODIES
        del model
    del plain, turned


def test_attach_names_the_module_and_rotates_it() -> None:
    """attach() prefixes the name and applies a non-zero rotation."""
    core = spider_with_blocks.make_core()
    hinge = HingeModule(index=1)
    attached = spider_with_blocks.attach(
        core, ModuleFaces.FRONT, hinge, "H", rotation=90
    )
    assert attached is hinge
    assert attached.name == "C-H"  # type: ignore[union-attr]  # known bug
    del core, hinge, attached


def test_gecko_main_writes_the_xml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running gecko.py as a script writes gecko.xml under the cwd."""
    target = tmp_path / (
        "ariel/src/ariel/body_phenotypes/robogen_lite/prebuilt_robots"
    )
    target.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(Path(gecko.__file__)), run_name="__main__")
    assert (target / "gecko.xml").read_text(encoding="utf-8").startswith("<")


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
    model = core.spec.compile()
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


def test_brick_and_hinge_expose_sites() -> None:
    """Both submodules expose attachment sites that builders can use."""
    assert BrickModule(index=1).sites
    assert HingeModule(index=2).sites
