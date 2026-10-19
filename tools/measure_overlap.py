"""Count robot bodies whose modules sit inside each other.

Run from the repo root: ``uv run --no-sync python tools/measure_overlap.py``.
Read-only. It builds 60 bodies from each of three sources with seed 42 (a
random tree of 10 modules, a random tree of 20, and the probability decoder
with 20), builds each with ``construct_mjspec_from_graph``, compiles it, and
measures every pair of geoms at the rest pose. A body counts as overlapping when
two modules that are not parent and child touch deeper than the tolerance:
1% of the thinnest side of the smaller of the two modules, so the rule does not
depend on the size of the robot. MuJoCo does the counting on purpose: the
repair finds overlaps with Trimesh and python-fcl, and this is the
independent check.

Notes
-----
    * A module is a graph node. Its geoms are named
      ``<parent>-<node>-<face>-<kind>``; the core's geom is ``core``.
    * The size of a module is the volume of its geoms, its thinnest side is
      twice the smallest half-size among them.
"""

import random
import re
import sys
import time
from collections.abc import Callable

import mujoco
import networkx as nx
import numpy as np
from pydantic import BaseModel, ConfigDict

# Local libraries
from ariel.body_phenotypes import (
    NUM_OF_FACES,
    NUM_OF_ROTATIONS,
    NUM_OF_TYPES_OF_MODULES,
    CoreModule,
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders import HighProbabilityDecoder
from ariel.ec.genotypes.tree import random_tree

SEED = 42
BODIES = 60
TOLERANCE = 0.01  # share of the thinnest side of the smaller module
CORE_NODE = 0
GEOM_OWNER = re.compile(r"^\d+-(\d+)-\d+-")

type Graph = nx.DiGraph[int]
type Builder = Callable[[Graph], CoreModule]


class Row(BaseModel):
    """The numbers measured for one source of bodies."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str
    bodies: int
    overlapping: int
    mean_ms: float


def owner_of(model: mujoco.MjModel, geom: int) -> int:
    """Find the graph node that owns a geom.

    Parameters
    ----------
    model : mujoco.MjModel
        The compiled body.
    geom : int
        The geom index.

    Returns
    -------
    int
        The node id; the core's geom gives 0.
    """
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
    match = GEOM_OWNER.match(name)
    return int(match.group(1)) if match else CORE_NODE


def module_sizes(
    model: mujoco.MjModel,
) -> dict[int, tuple[float, float]]:
    """Measure every module: its volume and its thinnest side.

    Parameters
    ----------
    model : mujoco.MjModel
        The compiled body; every geom must be a box.

    Returns
    -------
    dict[int, tuple[float, float]]
        Node id to (volume, thinnest side) in cubic metres and metres.
    """
    volumes: dict[int, float] = {}
    thinnest: dict[int, float] = {}
    for geom in range(model.ngeom):
        node = owner_of(model, geom)
        half = model.geom_size[geom]
        volumes[node] = volumes.get(node, 0.0) + float(8 * np.prod(half))
        thinnest[node] = min(thinnest.get(node, np.inf), float(2 * half.min()))
    return {node: (volumes[node], thinnest[node]) for node in volumes}


def has_overlap(graph: Graph, core: CoreModule) -> bool:
    """Tell whether two non-adjacent modules penetrate beyond the tolerance.

    Every pair of geoms that belong to different modules is measured with
    ``mj_geomDistance``, which gives a negative distance for a penetration.
    The contacts MuJoCo reports are not used: it skips contacts between
    bodies that are rigidly welded together, which is every brick on the core.

    Parameters
    ----------
    graph : nx.DiGraph[int]
        The graph the body was built from.
    core : CoreModule
        The body.

    Returns
    -------
    bool
        True when a distance is more negative than the tolerance of its pair.
    """
    model = core.spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sizes = module_sizes(model)
    owners = [owner_of(model, geom) for geom in range(model.ngeom)]
    fromto = np.zeros(6)
    for first_geom in range(model.ngeom):
        for second_geom in range(first_geom + 1, model.ngeom):
            first, second = owners[first_geom], owners[second_geom]
            if first == second or graph.has_edge(first, second):
                continue
            if graph.has_edge(second, first):
                continue
            smaller = min((first, second), key=lambda node: sizes[node][0])
            distance = mujoco.mj_geomDistance(
                model, data, first_geom, second_geom, 1.0, fromto
            )
            if -distance > TOLERANCE * sizes[smaller][1]:
                return True
    return False


def sources() -> dict[str, Callable[[], Graph]]:
    """Name the three sources of random bodies.

    Returns
    -------
    dict[str, Callable[[], Graph]]
        Source name to a function that draws one graph (seeded globally).
    """
    rng = np.random.default_rng(SEED)

    def probability(modules: int) -> Graph:
        decoder = HighProbabilityDecoder(modules)
        return decoder.probability_matrices_to_graph(
            rng.random((modules, NUM_OF_TYPES_OF_MODULES), dtype=np.float32),
            rng.random((modules, modules, NUM_OF_FACES), dtype=np.float32),
            rng.random((modules, NUM_OF_ROTATIONS), dtype=np.float32),
        )

    return {
        "tree 10 modules": lambda: random_tree(10).to_networkx(),
        "tree 20 modules": lambda: random_tree(20).to_networkx(),
        "probability 20": lambda: probability(20),
    }


def measure(builder: Builder, bodies: int = BODIES) -> list[Row]:
    """Build bodies from every source and count the overlapping ones.

    Parameters
    ----------
    builder : Callable
        Turns a graph into a body, for example ``construct_mjspec_from_graph``.
    bodies : int
        How many bodies to draw from each source.

    Returns
    -------
    list[Row]
        One row per source.
    """
    rows: list[Row] = []
    for name, draw in sources().items():
        random.seed(SEED)
        graphs = [draw() for _ in range(bodies)]
        overlapping = 0
        start = time.perf_counter()
        for graph in graphs:
            core = builder(graph)
            overlapping += has_overlap(graph, core)
        elapsed = time.perf_counter() - start
        rows.append(
            Row(
                source=name,
                bodies=bodies,
                overlapping=overlapping,
                mean_ms=1000 * elapsed / bodies,
            )
        )
    return rows


def main() -> int:
    """Print one line per source.

    Returns
    -------
    int
        Always 0.
    """
    sys.stdout.write("source             bodies  overlapping  ms per body\n")
    for row in measure(construct_mjspec_from_graph, BODIES):
        sys.stdout.write(
            f"{row.source:<18} {row.bodies:>6}  {row.overlapping:>11}  "
            f"{row.mean_ms:>11.1f}\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
