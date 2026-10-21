"""Test: pruning overlapping modules, and the repair flag of the constructor."""

from __future__ import annotations

import logging
import random

import networkx as nx
import pytest

from ariel.body_phenotypes import (
    CoreModule,
    construct_mjspec_from_graph,
    find_overlaps,
    repair_overlaps,
)
from ariel.body_phenotypes.robogen_lite import overlap
from ariel.body_phenotypes.robogen_lite.overlap import (
    Overlap,
    depth_of,
    later_module,
    prune_subtree,
)
from ariel.ec.genotypes.tree import random_tree
from tests.unit.test_body_phenotypes.test_robogen_lite import (
    test_constructor_xml as pinned,
)

RANDOM_BODIES = 25
SEED = 1
CORE_AND_BRICK_GEOMS = 2
DEEP = 2  # ids used in the small hand-built trees
SIDE = 3
FAR = 6


def _tree(edges: list[tuple[int, int, str]]) -> nx.DiGraph[int]:
    graph: nx.DiGraph[int] = nx.DiGraph()
    graph.add_node(0, type="CORE", rotation="DEG_0")
    for parent, child, face in edges:
        graph.add_node(child, type="BRICK", rotation="DEG_0")
        graph.add_edge(parent, child, face=face)
    return graph


def test_depth_counts_the_edges_from_the_core() -> None:
    graph = _tree([(0, 5, "FRONT"), (5, 2, "FRONT")])
    assert [depth_of(graph, node) for node in (0, 5, DEEP)] == [0, 1, 2]


def test_the_deeper_module_is_later_whatever_the_ids() -> None:
    graph = _tree([(0, 5, "FRONT"), (5, 2, "FRONT"), (0, 3, "RIGHT")])
    assert later_module(graph, DEEP, SIDE) == DEEP
    assert later_module(graph, SIDE, DEEP) == DEEP
    assert later_module(graph, 5, DEEP) == DEEP  # a lower id does not matter


def test_at_equal_depth_the_later_face_is_later() -> None:
    graph = _tree([(0, 1, "FRONT"), (0, 2, "RIGHT")])
    assert later_module(graph, 1, DEEP) == DEEP
    assert later_module(graph, DEEP, 1) == DEEP
    swapped = _tree([(0, 1, "RIGHT"), (0, 2, "FRONT")])
    assert later_module(swapped, 1, DEEP) == 1  # the face decides, not the id


def test_siblings_deeper_down_are_ordered_by_their_own_face() -> None:
    graph = _tree([(0, 1, "FRONT"), (1, 2, "RIGHT"), (1, 3, "LEFT")])
    assert later_module(graph, DEEP, SIDE) == SIDE


def test_the_first_edge_where_the_paths_differ_decides() -> None:
    # Node 5 hangs below FRONT then BOTTOM, node 6 below BACK then FRONT.
    # The last faces say 5 is later, the first differing edge says 6 is.
    graph = _tree([
        (0, 1, "FRONT"),
        (0, 4, "BACK"),
        (1, 5, "BOTTOM"),
        (4, 6, "FRONT"),
    ])
    assert later_module(graph, 5, FAR) == FAR


def test_prune_subtree_removes_the_module_and_what_hangs_on_it() -> None:
    graph = _tree([(0, 1, "FRONT"), (1, 2, "FRONT"), (0, 3, "RIGHT")])
    pruned = prune_subtree(graph, 1)
    assert sorted(pruned.nodes) == [0, 3]
    assert sorted(graph.nodes) == [0, 1, 2, 3]  # the argument is untouched


def test_the_core_cannot_be_pruned() -> None:
    with pytest.raises(ValueError, match="core"):
        prune_subtree(_tree([(0, 1, "FRONT")]), 0)


def test_the_deepest_later_module_goes_first_and_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    graph = pinned.graph_from(pinned.SPECS["overlapping"])
    caplog.set_level(logging.DEBUG, logger="rich")
    repaired = repair_overlaps(graph, construct_mjspec_from_graph)
    messages = [r.getMessage() for r in caplog.records if r.name == "rich"]
    assert messages == [
        "pruned 1 modules (ids 3) for overlap with 0",
        "pruned 1 modules (ids 2) for overlap with 0",
    ]
    assert sorted(repaired.nodes) == [0, 1]
    assert find_overlaps(repaired, construct_mjspec_from_graph(repaired)) == []


def test_the_smaller_id_goes_first_among_equally_deep_modules(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    graph = _tree([(0, 1, "FRONT"), (0, 2, "RIGHT")])

    def fake(found: nx.DiGraph[int], _core: object) -> list[Overlap]:
        return [
            Overlap(first=0, second=node, depth=1.0, tolerance=0.1)
            for node in (1, 2)
            if node in found
        ]

    monkeypatch.setattr(overlap, "find_overlaps", fake)
    caplog.set_level(logging.DEBUG, logger="rich")
    repaired = repair_overlaps(graph, lambda _graph: CoreModule(index=0))
    assert [r.getMessage() for r in caplog.records if r.name == "rich"] == [
        "pruned 1 modules (ids 1) for overlap with 0",
        "pruned 1 modules (ids 2) for overlap with 0",
    ]
    assert list(repaired.nodes) == [0]


def test_a_removed_module_takes_its_subtree_with_it(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    graph = _tree([(0, 1, "FRONT"), (1, 2, "FRONT"), (2, 3, "FRONT")])

    def fake(found: nx.DiGraph[int], _core: object) -> list[Overlap]:
        return (
            [Overlap(first=0, second=DEEP, depth=1.0, tolerance=0.1)]
            if DEEP in found
            else []
        )

    monkeypatch.setattr(overlap, "find_overlaps", fake)
    caplog.set_level(logging.DEBUG, logger="rich")
    repaired = repair_overlaps(graph, lambda _graph: CoreModule(index=0))
    assert [r.getMessage() for r in caplog.records if r.name == "rich"] == [
        "pruned 2 modules (ids 2, 3) for overlap with 0"
    ]
    assert sorted(repaired.nodes) == [0, 1]


@pytest.mark.parametrize("name", ["core only", "snake", "cross"])
def test_a_body_without_overlap_is_identical_with_the_repair_on(
    name: str,
) -> None:
    graph = pinned.graph_from(pinned.SPECS[name])
    xml = construct_mjspec_from_graph(graph, repair_overlaps=True).spec.to_xml()
    plain = construct_mjspec_from_graph(graph).spec.to_xml()
    assert xml == plain
    assert nx.utils.graphs_equal(
        repair_overlaps(graph, construct_mjspec_from_graph), graph
    )
    assert pinned.XML_HASHES[name]  # the plain body is the pinned one


def test_the_flag_is_off_by_default() -> None:
    graph = pinned.graph_from(pinned.SPECS["overlapping"])
    plain = construct_mjspec_from_graph(graph).spec.to_xml()
    assert plain.count("<geom") > CORE_AND_BRICK_GEOMS  # nothing pruned


def test_the_flag_builds_the_body_without_the_overlap() -> None:
    graph = pinned.graph_from(pinned.SPECS["overlapping"])
    before = nx.DiGraph(graph)
    core = construct_mjspec_from_graph(graph, repair_overlaps=True)
    assert core.spec.to_xml().count("<geom") == CORE_AND_BRICK_GEOMS
    assert (
        find_overlaps(pinned.graph_from(pinned.SPECS["overlapping"]), core)
        is not None
    )
    assert nx.utils.graphs_equal(graph, before)  # the argument is untouched


def test_random_bodies_are_repaired_for_good() -> None:
    random.seed(SEED)
    for _ in range(RANDOM_BODIES):
        graph = random_tree(12).to_networkx()
        before = nx.DiGraph(graph)
        repaired = repair_overlaps(graph, construct_mjspec_from_graph)
        assert nx.utils.graphs_equal(graph, before)
        assert 0 in repaired
        assert set(repaired.nodes) <= set(graph.nodes)
        assert (
            find_overlaps(repaired, construct_mjspec_from_graph(repaired)) == []
        )
        assert nx.utils.graphs_equal(
            repair_overlaps(repaired, construct_mjspec_from_graph), repaired
        )
