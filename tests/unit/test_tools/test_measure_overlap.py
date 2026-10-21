"""Tests for the overlap measurement in ``tools/`` (real MuJoCo, small runs)."""

from __future__ import annotations

# Standard library
import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, cast

# Third-party libraries
import networkx as nx
import pytest

# Local libraries
from ariel.body_phenotypes import construct_mjspec_from_graph

if TYPE_CHECKING:
    from types import ModuleType

    import mujoco

SCRIPT = Path(__file__).resolve().parents[3] / "tools" / "measure_overlap.py"
SMALL = 5
HEADER_AND_SOURCES = 4
CLEAN_NODES = {
    0: ("CORE", "DEG_0"),
    1: ("BRICK", "DEG_0"),
    2: ("HINGE", "DEG_90"),
    3: ("BRICK", "DEG_0"),
}
CLEAN_EDGES = [(0, 1, "FRONT"), (1, 2, "FRONT"), (2, 3, "FRONT")]
OVERLAP_NODES = {
    0: ("CORE", "DEG_0"),
    1: ("BRICK", "DEG_45"),
    2: ("BRICK", "DEG_45"),
    3: ("HINGE", "DEG_45"),
}
OVERLAP_EDGES = [(0, 1, "FRONT"), (1, 2, "TOP"), (2, 3, "LEFT")]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_overlap", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _graph(
    nodes: dict[int, tuple[str, str]], edges: list[tuple[int, int, str]]
) -> nx.DiGraph[int]:
    graph: nx.DiGraph[int] = nx.DiGraph()
    for node, (kind, rotation) in nodes.items():
        graph.add_node(node, type=kind, rotation=rotation)
    for parent, child, face in edges:
        graph.add_edge(parent, child, face=face)
    return graph


def test_a_clean_body_has_no_overlap() -> None:
    script = _load()
    graph = _graph(CLEAN_NODES, CLEAN_EDGES)
    assert not script.has_overlap(graph, construct_mjspec_from_graph(graph))


def test_a_body_with_modules_inside_each_other_is_counted() -> None:
    script = _load()
    graph = _graph(OVERLAP_NODES, OVERLAP_EDGES)
    assert script.has_overlap(graph, construct_mjspec_from_graph(graph))


def test_module_sizes_read_the_hinge_and_the_brick() -> None:
    script = _load()
    graph = _graph(CLEAN_NODES, CLEAN_EDGES)
    model = cast(
        "mujoco.MjModel", construct_mjspec_from_graph(graph).spec.compile()
    )
    sizes = script.module_sizes(model)
    brick_volume, brick_side = sizes[1]
    hinge_volume, hinge_side = sizes[2]
    assert brick_side == pytest.approx(0.075)
    assert brick_volume == pytest.approx(0.075**3)
    # The fork, from the plate to the axis (31.74 mm, drawn 1% smaller)
    assert hinge_side == pytest.approx(0.03174 * 0.99)
    assert hinge_volume < brick_volume


def test_the_geoms_of_a_hinge_belong_to_the_hinge() -> None:
    script = _load()
    model = cast(
        "mujoco.MjModel",
        construct_mjspec_from_graph(
            _graph(CLEAN_NODES, CLEAN_EDGES)
        ).spec.compile(),
    )
    owners = [script.owner_of(model, geom) for geom in range(model.ngeom)]
    assert owners == [0, 1, 2, 2, 3]  # core, brick, stator, rotor, brick


def test_measure_gives_one_row_per_source() -> None:
    script = _load()
    rows = script.measure(bodies=SMALL)
    assert [row.source for row in rows] == [
        "tree 10 modules",
        "tree 20 modules",
        "probability 20",
    ]
    for row in rows:
        assert row.bodies == SMALL
        assert 0 <= row.overlapping <= SMALL
        assert row.repaired_overlapping == 0  # the repair leaves none
        assert 0 <= row.identical <= row.clean <= SMALL
        assert row.identical == row.clean  # clean bodies are unchanged
        assert row.build_ms > 0
        assert row.repair_ms > 0


def test_measure_is_repeatable() -> None:
    script = _load()
    first = script.measure(bodies=SMALL)
    second = script.measure(bodies=SMALL)
    keys = ("overlapping", "repaired_overlapping", "removed", "clean")
    assert [[getattr(r, k) for k in keys] for r in first] == [
        [getattr(r, k) for k in keys] for r in second
    ]


def test_main_prints_the_table(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _load()
    monkeypatch.setattr(script, "BODIES", SMALL)
    assert script.main() == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("source")
    assert len(lines) == HEADER_AND_SOURCES
