"""Find modules of a robot body that sit inside each other.

The body is built and compiled, every geom becomes a Trimesh box, and
python-fcl (through ``trimesh.collision``) reports the penetration depth of
every colliding pair. Two modules overlap when their depth is more than
``TOLERANCE`` times the thinnest side of the smaller of the two, so the rule
does not depend on the size of the robot. Parent and child modules share a
face on purpose and are never compared.

Notes
-----
    * A module is a graph node. Its geoms are named
        ``<parent>-<node>-<face>-<kind>``; the core's geom is ``core``.
    * The size of a module is the total volume of its geoms; its thinnest
        side is twice the smallest half-size among them.
    * Poses are at joint angle zero, as compiled.
    * The later module of an overlapping pair is the deeper one in the tree;
        at equal depth, the one whose path from the core takes the later face
        (``ModuleFaces`` order) at the first edge where the two paths differ.
        The repair removes it with everything attached beyond it.
"""

# Standard library
import re
from collections.abc import Callable
from functools import cache

# Third-party libraries
import mujoco
import networkx as nx
import numpy as np
import trimesh
from numpy.typing import NDArray
from trimesh.collision import CollisionManager

# Local libraries
from ariel import log
from ariel.body_phenotypes.robogen_lite.config import IDX_OF_CORE, ModuleFaces
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.parameters import FrozenModel

# Global constants
TOLERANCE = 0.01  # share of the thinnest side of the smaller module
GEOM_OWNER = re.compile(r"^\d+-(\d+)-\d+-")

type Graph = nx.DiGraph[int]
type Builder = Callable[[Graph], CoreModule]
type Floats = NDArray[np.float64]


class Overlap(FrozenModel):
    """Two modules that penetrate each other beyond the tolerance."""

    first: int
    second: int
    depth: float
    tolerance: float


def pair_tolerance(
    first: tuple[float, float], second: tuple[float, float]
) -> float:
    """Work out how deep two modules may penetrate each other.

    Parameters
    ----------
    first : tuple[float, float]
        Volume and thinnest side of the first module.
    second : tuple[float, float]
        Volume and thinnest side of the second module.

    Returns
    -------
    float
        ``TOLERANCE`` times the thinnest side of the smaller module (by
        volume; the first when the volumes are equal), in the unit of the
        sides.
    """
    smaller = first if first[0] <= second[0] else second
    return TOLERANCE * smaller[1]


def module_sizes(
    owners: list[int], halves: Floats
) -> dict[int, tuple[float, float]]:
    """Measure every module from its boxes.

    Parameters
    ----------
    owners : list[int]
        The node that owns each box.
    halves : NDArray[np.float64]
        The half sizes of the boxes, one row of three per box.

    Returns
    -------
    dict[int, tuple[float, float]]
        Node id to (total volume, thinnest side).
    """
    volumes: dict[int, float] = {}
    thinnest: dict[int, float] = {}
    for owner, half in zip(owners, halves, strict=True):
        volumes[owner] = volumes.get(owner, 0.0) + float(8 * np.prod(half))
        thinnest[owner] = min(
            thinnest.get(owner, float("inf")), float(2 * half.min())
        )
    return {node: (volumes[node], thinnest[node]) for node in volumes}


@cache
def box_mesh(extents: tuple[float, float, float]) -> trimesh.Trimesh:
    """Make the Trimesh box of a size, once.

    Trimesh tests every mesh for convexity when it is added to a collision
    manager and keeps the answer on the mesh, so reusing one mesh for every
    box of a size removes most of the time of a collision check.

    Parameters
    ----------
    extents : tuple[float, float, float]
        The full side lengths.

    Returns
    -------
    trimesh.Trimesh
        The box, centred on the origin. It is shared: do not change it.
    """
    mesh: trimesh.Trimesh = trimesh.creation.box(extents=extents)
    return mesh


def box_overlaps(
    graph: Graph,
    owners: list[int],
    halves: Floats,
    poses: Floats,
) -> list[Overlap]:
    """Find the overlapping modules among a set of boxes.

    Parameters
    ----------
    graph : nx.DiGraph[int]
        The robot graph; modules that are parent and child are skipped.
    owners : list[int]
        The node that owns each box.
    halves : NDArray[np.float64]
        Half sizes, one row of three per box.
    poses : NDArray[np.float64]
        One 4 x 4 pose matrix per box.

    Returns
    -------
    list[Overlap]
        Pairs deeper than the tolerance, ordered by node ids. The depth of a
        pair is the deepest contact between any of their boxes.
    """
    manager = CollisionManager()  # type: ignore[no-untyped-call]  # trimesh.collision is untyped
    for index, (half, pose) in enumerate(zip(halves, poses, strict=True)):
        manager.add_object(  # type: ignore[no-untyped-call]  # trimesh.collision is untyped
            str(index), box_mesh(tuple(float(side) for side in 2 * half)), pose
        )
    _, contacts = manager.in_collision_internal(return_data=True)  # type: ignore[no-untyped-call]  # trimesh.collision is untyped
    depths: dict[tuple[int, int], float] = {}
    for contact in contacts:
        names = sorted(int(name) for name in contact.names)
        first, second = sorted((owners[names[0]], owners[names[1]]))
        if first == second or graph.has_edge(first, second):
            continue
        if graph.has_edge(second, first):
            continue
        depths[first, second] = max(
            depths.get((first, second), 0.0), float(contact.depth)
        )
    sizes = module_sizes(owners, halves)
    found = []
    for (first, second), depth in sorted(depths.items()):
        tolerance = pair_tolerance(sizes[first], sizes[second])
        if depth > tolerance:
            found.append(
                Overlap(
                    first=first,
                    second=second,
                    depth=depth,
                    tolerance=tolerance,
                )
            )
    return found


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
        The node id; the core's geom gives the index of the core.
    """
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
    match = GEOM_OWNER.match(name)
    return int(match.group(1)) if match else IDX_OF_CORE


def find_overlaps(graph: Graph, core: CoreModule) -> list[Overlap]:
    """Find the modules of a built body that sit inside each other.

    Parameters
    ----------
    graph : nx.DiGraph[int]
        The graph the body was built from.
    core : CoreModule
        The body built from ``graph``.

    Returns
    -------
    list[Overlap]
        Pairs deeper than the tolerance, ordered by node ids.

    Raises
    ------
    ValueError
        If a geom of the body is not a box.
    """
    model = core.spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    box = mujoco.mjtGeom.mjGEOM_BOX
    if any(kind != box for kind in model.geom_type):
        msg = "Overlap detection only handles box geoms."
        raise ValueError(msg)
    owners = [owner_of(model, geom) for geom in range(model.ngeom)]
    poses = np.tile(np.eye(4), (model.ngeom, 1, 1))
    poses[:, :3, :3] = data.geom_xmat.reshape(-1, 3, 3)
    poses[:, :3, 3] = data.geom_xpos
    return box_overlaps(graph, owners, np.array(model.geom_size), poses)


def depth_of(graph: Graph, node: int) -> int:
    """Count the edges between the core and a module.

    Parameters
    ----------
    graph : nx.DiGraph[int]
        The robot graph.
    node : int
        The module.

    Returns
    -------
    int
        The number of edges on the path from the core.
    """
    return int(nx.shortest_path_length(graph, IDX_OF_CORE, node))


def later_module(graph: Graph, first: int, second: int) -> int:
    """Pick which of two overlapping modules counts as the later one.

    Parameters
    ----------
    graph : nx.DiGraph[int]
        The robot graph.
    first : int
        One module.
    second : int
        The other module.

    Returns
    -------
    int
        The deeper module; at equal depth, the one whose path from the core
        leaves the common part through the later face. The answer does not
        depend on the order of the arguments.
    """
    depths = depth_of(graph, first), depth_of(graph, second)
    if depths[0] != depths[1]:
        return first if depths[0] > depths[1] else second
    paths = [
        nx.shortest_path(graph, IDX_OF_CORE, node) for node in (first, second)
    ]
    step = next(
        index
        for index, (one, other) in enumerate(zip(*paths, strict=True))
        if one != other
    )
    faces = [
        ModuleFaces[graph.edges[path[step - 1], path[step]]["face"]].value
        for path in paths
    ]
    return first if faces[0] > faces[1] else second


def prune_subtree(graph: Graph, node: int) -> Graph:
    """Remove a module and everything attached beyond it.

    Parameters
    ----------
    graph : nx.DiGraph[int]
        The robot graph; it is not changed.
    node : int
        The module to remove.

    Returns
    -------
    nx.DiGraph[int]
        A copy without the module and its descendants.

    Raises
    ------
    ValueError
        If the module is the core.
    """
    if node == IDX_OF_CORE:
        msg = "The core cannot be pruned."
        raise ValueError(msg)
    pruned = graph.copy()
    pruned.remove_nodes_from([node, *nx.descendants(graph, node)])
    return pruned


def repair_overlaps(graph: Graph, build: Builder) -> Graph:
    """Prune modules until no two modules overlap beyond the tolerance.

    Each round builds the body, finds the overlapping pairs, and removes the
    later module of the pair whose later module is deepest (ties: the smaller
    node id), with its subtree; removing the deepest first keeps the most of
    the body. Every round removes at least one module, so the loop ends.

    Parameters
    ----------
    graph : nx.DiGraph[int]
        The robot graph; it is not changed.
    build : Callable
        Builds the body of a graph, for example the constructor.

    Returns
    -------
    nx.DiGraph[int]
        A copy of the graph without overlaps; the graph itself when there
        were none.
    """
    repaired = graph.copy()
    while found := find_overlaps(repaired, build(repaired)):
        victims = {
            overlap: later_module(repaired, overlap.first, overlap.second)
            for overlap in found
        }
        overlap, victim = min(
            victims.items(),
            key=lambda item: (-depth_of(repaired, item[1]), item[1]),
        )
        other = overlap.first if victim == overlap.second else overlap.second
        gone = sorted(
            {victim, *nx.descendants(repaired, victim)},
        )
        log.debug(
            "pruned %d modules (ids %s) for overlap with %d",
            len(gone),
            ", ".join(str(node) for node in gone),
            other,
        )
        repaired = prune_subtree(repaired, victim)
    return repaired
