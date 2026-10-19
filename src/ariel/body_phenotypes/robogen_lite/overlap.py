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
"""

# Standard library
import re

# Third-party libraries
import mujoco
import networkx as nx
import numpy as np
import trimesh
from numpy.typing import NDArray
from trimesh.collision import CollisionManager

# Local libraries
from ariel.body_phenotypes.robogen_lite.config import IDX_OF_CORE
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.parameters import FrozenModel

# Global constants
TOLERANCE = 0.01  # share of the thinnest side of the smaller module
GEOM_OWNER = re.compile(r"^\d+-(\d+)-\d+-")

type Graph = nx.DiGraph[int]
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
            str(index), trimesh.creation.box(extents=2 * half), pose
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
