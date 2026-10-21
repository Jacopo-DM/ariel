"""Test: the count measures keep the values they had before geometry-measures.

``golden_counts.json`` holds every measure that reads only the graph, for 60
random trees of up to 20 modules (``random.seed(42)``, as
``tools/measure_overlap.py``) and the 13 robots of ``john_set``, recorded with
the grid-based ``MorphologicalMeasures`` of commit ``ef74198``. The shape
measures (bounding box, coverage, proportion, symmetry, ``is_2d``) are left
out on purpose: their meaning changes with the goal geometry-measures.
"""

from __future__ import annotations

# Standard library
import json
import random
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

# Third-party libraries
import networkx as nx
import pytest

# Local libraries
from ariel.body_phenotypes import MorphologicalMeasures
from ariel.body_phenotypes.robogen_lite.prebuilt_robots import john_tables
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.parts import build
from ariel.ec.genotypes.tree import random_tree

if TYPE_CHECKING:
    from ariel.body_phenotypes.robogen_lite.prebuilt_robots.parts import Part

GOLDEN = Path(__file__).with_name("golden_counts.json")
SEED = 42
BODIES = 60
MODULES = 20
TABLES = (
    "BABY_A",
    "BABY_B",
    "CENTIPEDE_3",
    "CENTIPEDE_4",
    "CENTIPEDE_5",
    "GECKO",
    "IGUANA",
    "LINKIN_MODIFIED",
    "SNAKE",
    "SPIDER_8",
    "SPIDER_12",
    "SPIDER_16",
    "TURTLE",
)
COUNTS = (
    "num_modules",
    "num_bricks",
    "num_active_hinges",
    "num_joints",
    "num_filled_bricks",
    "num_filled_active_hinges",
    "num_filled_modules",
    "num_single_neighbour_modules",
    "num_double_neighbour_bricks",
    "num_double_neighbour_active_hinges",
    "max_potentionally_filled_core_and_bricks",
    "max_potential_single_neighbour_modules",
    "max_potential_joints",
    "potential_double_neighbour_bricks_and_active_hinges",
    "core_is_filled",
    "filled_core_and_bricks_proportion",
    "double_neighbour_brick_and_active_hinge_proportion",
    "branching",
    "limbs",
    "length_of_limbs",
    "module_diversity",
    "joints",
    "brick_cluster_ratio",
    "B",
    "L",
    "E",
    "J",
    "D",
)

type Graph = nx.DiGraph[int]
type Values = dict[str, float | int | bool]


def table_graph(parts: tuple[Part, ...]) -> Graph:
    """Turn a table of parts into the graph it describes.

    Parameters
    ----------
    parts : tuple[Part, ...]
        A table from ``john_tables``.

    Returns
    -------
    nx.DiGraph[int]
        Nodes with ``type`` and ``rotation``, edges with ``face``.
    """
    graph: Graph = nx.DiGraph()
    graph.add_node(0, type="CORE", rotation="DEG_0")
    ids = {"core": 0}
    seen = {"hinge": 0, "brick": 0}
    for part in parts:
        name = f"{part.kind}_{seen[part.kind]}"
        seen[part.kind] += 1
        ids[name] = part.index
        graph.add_node(part.index, type=part.kind.upper(), rotation="DEG_0")
        graph.add_edge(ids[part.parent], part.index, face=part.face.name)
    return graph


@cache
def bodies() -> dict[str, Graph]:
    """Name every body the golden values cover.

    Returns
    -------
    dict[str, nx.DiGraph[int]]
        Body name to its graph.
    """
    random.seed(SEED)
    named = {
        f"tree {i}": random_tree(MODULES).to_networkx() for i in range(BODIES)
    }
    for table in TABLES:
        named[table] = table_graph(getattr(john_tables, table))
    return named


def counts(graph: Graph) -> Values:
    """Read every count measure of one body.

    Parameters
    ----------
    graph : nx.DiGraph[int]
        The body.

    Returns
    -------
    dict[str, float | int | bool]
        Measure name to value.
    """
    measures = MorphologicalMeasures(graph)
    return {name: getattr(measures, name) for name in COUNTS}


GOLDEN_VALUES: dict[str, Values] = json.loads(GOLDEN.read_text())


@pytest.mark.parametrize("name", sorted(GOLDEN_VALUES))
def test_the_counts_match_the_golden_values(name: str) -> None:
    assert counts(bodies()[name]) == GOLDEN_VALUES[name]


def test_the_golden_file_covers_every_body() -> None:
    assert set(GOLDEN_VALUES) == set(bodies())


@pytest.mark.parametrize("table", TABLES)
def test_a_built_table_robot_keeps_the_golden_graph(table: str) -> None:
    graph = build(getattr(john_tables, table)).graph
    assert graph is not None
    assert counts(graph) == GOLDEN_VALUES[table]
