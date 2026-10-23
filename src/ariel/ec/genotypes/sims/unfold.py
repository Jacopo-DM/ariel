"""Unfold a Sims genome into the body graph the builder takes.

Unfolding starts at the core and grows the body in rounds: in each round
every module placed in the round before grows the children its template's
connections ask for, in the order the connections are listed. A template
may appear at most its copy limit times along one path from the core, and
never more than a fifth of the maximum number of modules. A connection
marked ``terminal_only`` grows only from the last copy of a chain. A face
that already holds a module keeps it: the first connection wins.

Mirroring. A module whose mirror image (through the body's centre plane,
the plane through the core from front to back) is the module itself lies
on that plane: the core, and chains grown straight along it. A connection
with ``reflect`` set, from such a module, also grows a mirrored twin of its
child on the opposite face, and the twin's whole subtree is the mirror
image of the child's. An original and its twin are placed together or not
at all, so a symmetric genotype stays symmetric when the maximum number of
modules cuts it short.

The faces and tilts of a mirrored module are found from the real module
frames: each placed module carries the reflection, in its own coordinates,
that maps it onto its mirror image, and the child's face and tilt are the
ones whose site and frame match that reflection (DESIGN.md Decisions,
"Mirroring modules"). A brick keeps its tilt; a hinge at 45 degrees
becomes one at -45.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import mujoco
import networkx as nx
import numpy as np
import numpy.typing as npt

from ariel.body_phenotypes import (
    ALLOWED_FACES,
    IDX_OF_CORE,
    BrickModule,
    CoreModule,
    HingeModule,
    ModuleFaces,
    ModuleRotationsTheta,
    ModuleType,
)
from ariel.parameters import FrozenModel

if TYPE_CHECKING:
    from .genome import SimsConnection, SimsGenome, SimsNode

type Matrix = npt.NDArray[np.float64]
type Flat = tuple[float, ...]  # a 3 x 3 matrix, row by row

MODULE_SHARE = 5  # the copy limit is at most a fifth of the module maximum
DEFAULT_MAX_MODULES = 30  # as the other decoders
ANGLE_TOLERANCE = 0.01  # the modules round their quaternions to 3 decimals
POSITION_TOLERANCE = 1e-6  # metres
# Tilts tried for a mirrored module, its own first; only a hinge needs -45
TILTS = {
    ModuleType.BRICK: ("DEG_0", "DEG_45", "DEG_90"),
    ModuleType.HINGE: ("DEG_0", "DEG_45", "DEG_90", "DEG_MINUS_45"),
}


def max_copy_limit(max_modules: int) -> int:
    """Return the highest copy limit allowed in a body of `max_modules`.

    Returns
    -------
    int
        A fifth (20%) of `max_modules`, at least 1: 6 for 30, 20 for 100.
    """
    return max(1, max_modules // MODULE_SHARE)


def _reflection(normal: tuple[float, float, float]) -> Matrix:
    n = np.asarray(normal, dtype=np.float64)
    n /= np.linalg.norm(n)
    return np.eye(3) - 2 * np.outer(n, n)


# Reflections that map a module's shape onto itself, in its own frame: the
# module grows along its y axis and has a square cross-section; a hinge
# must also keep its joint axis (z) on its own line
SYMMETRIES = {
    ModuleType.BRICK: tuple(
        _reflection(n) for n in ((1, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, -1))
    ),
    ModuleType.HINGE: tuple(_reflection(n) for n in ((1, 0, 0), (0, 0, 1))),
}


def _rotation(quat: npt.ArrayLike) -> Matrix:
    unit = np.asarray(quat, dtype=np.float64)
    unit /= np.linalg.norm(unit)
    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, unit)
    return matrix.reshape(3, 3)


def _flat(matrix: Matrix) -> Flat:
    return tuple(float(value) for value in matrix.ravel())


def _matrix(flat: Flat) -> Matrix:
    return np.asarray(flat, dtype=np.float64).reshape(3, 3)


def _module(kind: ModuleType) -> BrickModule | HingeModule:
    if kind is ModuleType.BRICK:
        return BrickModule(index=1)
    return HingeModule(index=1)


@cache
def _site(kind: ModuleType, face: ModuleFaces) -> tuple[Matrix, Matrix]:
    """Return the frame and position of a face's site in the module's frame.

    Returns
    -------
    tuple[Matrix, Matrix]
        Rotation of the site, and its position, relative to the module.
    """
    if kind is ModuleType.CORE:
        site = CoreModule(index=IDX_OF_CORE).sites[face]
        return _rotation(site.quat), np.asarray(site.pos, dtype=np.float64)
    module = _module(kind)
    site = module.sites[face]
    rotation = _rotation(site.quat)
    position = np.asarray(site.pos, dtype=np.float64)
    if isinstance(module, HingeModule):  # the site sits on the rotor
        rotor = next(b for b in module.body.bodies if b.name == "rotor")
        rotor_rotation = _rotation(rotor.quat)
        rotation = rotor_rotation @ rotation
        position = np.asarray(rotor.pos) + rotor_rotation @ position
    return rotation, position


@cache
def _tilt(kind: ModuleType, tilt: str) -> Matrix:
    """Return the frame of a module tilted by `tilt`, relative to its site.

    Returns
    -------
    Matrix
        The rotation the builder gives the module's body.
    """
    module = _module(kind)
    module.rotate(ModuleRotationsTheta[tilt].value)
    return _rotation(module.body.quat)


@cache
def core_mirror() -> Flat:
    """Return the core's centre-plane reflection, in the core's frame.

    Returns
    -------
    Flat
        The reflection through the plane halfway between the core's LEFT
        and RIGHT faces.
    """
    right = _site(ModuleType.CORE, ModuleFaces.RIGHT)[1]
    left = _site(ModuleType.CORE, ModuleFaces.LEFT)[1]
    x, y, z = (float(v) for v in right - left)
    return _flat(_reflection((x, y, z)))


class Placement(FrozenModel):
    """Where a mirrored module goes and how it relates to its original."""

    face: ModuleFaces
    tilt: str
    mirror: Flat  # maps the module onto its original, in its own frame


def mirror_child(
    parent: ModuleType,
    parent_mirror: Flat,
    face: ModuleFaces,
    tilt: str,
    child: ModuleType,
) -> Placement:
    """Find the mirror image of a child on `face` of a mirrored parent.

    Parameters
    ----------
    parent
        Type of the parent module.
    parent_mirror
        Reflection that maps the parent onto its mirror image, in the
        parent's frame.
    face
        Face of the parent the original child grows on.
    tilt
        ``ModuleRotationsTheta`` name of the original child's tilt.
    child
        Type of the child module.

    Returns
    -------
    Placement
        The face and tilt of the mirror image on the mirrored parent, and
        its own reflection.

    Raises
    ------
    ValueError
        If no face and tilt give the mirror image. Every reflection that
        unfolding passes maps the parent's faces onto faces it has, so this
        means the module frames changed under it.
    """
    reflect = _matrix(parent_mirror)
    rotation, position = _site(parent, face)
    target = reflect @ rotation @ _tilt(child, tilt)
    target_position = reflect @ position
    tilts = (tilt, *(t for t in TILTS[child] if t != tilt))
    for new_face in ALLOWED_FACES[parent]:
        new_rotation, new_position = _site(parent, new_face)
        if not np.allclose(
            new_position, target_position, atol=POSITION_TOLERANCE
        ):
            continue
        for new_tilt in tilts:
            relation = (new_rotation @ _tilt(child, new_tilt)).T @ target
            for symmetry in SYMMETRIES[child]:
                if np.allclose(relation, symmetry, atol=ANGLE_TOLERANCE):
                    return Placement(
                        face=new_face, tilt=new_tilt, mirror=_flat(symmetry)
                    )
    msg = f"No mirror image of a {child.name} on the {face.name} face of a"
    msg += f" {parent.name} at {tilt}"
    raise ValueError(msg)


class _Module(FrozenModel):
    """A module placed in the body."""

    id: int
    kind: ModuleType
    template: int
    counts: tuple[int, ...]  # copies of each template on the path here
    flip: Flat | None  # maps it onto the unmirrored module, if mirrored
    on_plane: Flat | None  # maps it onto itself, if it lies on the plane


class _Child(FrozenModel):
    """A module about to be placed on a face of a placed one."""

    parent: _Module
    template: int
    kind: ModuleType
    face: ModuleFaces
    tilt: str
    counts: tuple[int, ...]
    flip: Flat | None
    on_plane: Flat | None


def _limit(node: SimsNode, cap: int) -> int:
    return min(node.copy_limit, cap)


def _child(
    genome: SimsGenome, module: _Module, connection: SimsConnection, cap: int
) -> _Child | None:
    """Return the child `connection` grows on `module`, if it may grow.

    Returns
    -------
    _Child | None
        The child, or ``None`` past its template's copy limit or for a
        terminal connection before the last copy.
    """
    node = genome.nodes[connection.child]
    count = module.counts[connection.child] + 1
    if count > _limit(node, cap):
        return None
    last = _limit(genome.nodes[module.template], cap)
    if connection.terminal_only and module.counts[module.template] < last:
        return None
    face, tilt, flip = connection.face, node.rotation.name, None
    if module.flip is not None:
        placement = mirror_child(
            module.kind, module.flip, face, tilt, node.type
        )
        face, tilt, flip = placement.face, placement.tilt, placement.mirror
    on_plane = None
    if module.on_plane is not None:
        image = mirror_child(
            module.kind, module.on_plane, face, tilt, node.type
        )
        if (image.face, image.tilt) == (face, tilt):
            on_plane = image.mirror
    counts = list(module.counts)
    counts[connection.child] = count
    return _Child(
        parent=module,
        template=connection.child,
        kind=node.type,
        face=face,
        tilt=tilt,
        counts=tuple(counts),
        flip=flip,
        on_plane=on_plane,
    )


def _twin(child: _Child) -> _Child | None:
    """Return the mirrored twin of a child of a module on the plane.

    Returns
    -------
    _Child | None
        The twin on the opposite face, or ``None`` when the child's face
        lies on the plane (its mirror image is on the same face).
    """
    plane = child.parent.on_plane
    if plane is None:
        return None
    image = mirror_child(
        child.parent.kind, plane, child.face, child.tilt, child.kind
    )
    if image.face is child.face:
        return None
    return child.model_copy(
        update={
            "face": image.face,
            "tilt": image.tilt,
            "flip": image.mirror,
            "on_plane": None,
        }
    )


def _children(
    genome: SimsGenome,
    group: tuple[_Module, ...],
    connection: SimsConnection,
    cap: int,
) -> list[_Child]:
    """Return what `connection` grows on a module and its mirrored twin.

    Returns
    -------
    list[_Child]
        One child per module of the group that may grow it, plus a new twin
        when the connection reflects and the module lies on the plane.
    """
    children = [
        child
        for module in group
        if (child := _child(genome, module, connection, cap)) is not None
    ]
    if connection.reflect and len(group) == 1 and children:
        twin = _twin(children[0])
        if twin is not None:
            children.append(twin)
    return children


def unfold(
    genome: SimsGenome, max_modules: int = DEFAULT_MAX_MODULES
) -> nx.DiGraph[int]:
    """Unfold `genome` into a body graph of at most `max_modules` modules.

    Parameters
    ----------
    genome
        The genotype.
    max_modules
        Most modules the body may have, the core included.

    Returns
    -------
    nx.DiGraph[int]
        The body as ``construct_mjspec_from_graph`` takes it: ``type`` and
        ``rotation`` (``ModuleRotationsTheta`` names) on nodes, ``face`` on
        edges; each node also carries ``template``, the genome node it was
        copied from. Node ids count up from 0, the core, in the order the
        modules were placed.
    """
    cap = max_copy_limit(max_modules)
    graph: nx.DiGraph[int] = nx.DiGraph()
    graph.add_node(IDX_OF_CORE, type="CORE", rotation="DEG_0", template=0)
    counts = [0] * len(genome.nodes)
    counts[IDX_OF_CORE] = 1
    core = _Module(
        id=IDX_OF_CORE,
        kind=ModuleType.CORE,
        template=IDX_OF_CORE,
        counts=tuple(counts),
        flip=None,
        on_plane=core_mirror(),
    )
    taken: dict[int, set[ModuleFaces]] = {IDX_OF_CORE: set()}
    rounds: list[tuple[_Module, ...]] = [(core,)]
    while rounds:
        placed: list[tuple[_Module, ...]] = []
        for group in rounds:
            for connection in genome.connections:
                if connection.parent != group[0].template:
                    continue
                children = [
                    child
                    for child in _children(genome, group, connection, cap)
                    if child.face not in taken[child.parent.id]
                ]
                if not children or len(graph) + len(children) > max_modules:
                    continue
                placed.append(
                    tuple(_place(graph, taken, child) for child in children)
                )
        rounds = placed
    return graph


def _place(
    graph: nx.DiGraph[int], taken: dict[int, set[ModuleFaces]], child: _Child
) -> _Module:
    node = len(graph)
    graph.add_node(
        node, type=child.kind.name, rotation=child.tilt, template=child.template
    )
    graph.add_edge(child.parent.id, node, face=child.face.name)
    taken[child.parent.id].add(child.face)
    taken[node] = set()
    return _Module(
        id=node,
        kind=child.kind,
        template=child.template,
        counts=child.counts,
        flip=child.flip,
        on_plane=child.on_plane,
    )
