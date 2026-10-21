"""Describe a robot body as a table of parts and build it.

Each part is one line: the kind of module, its index, the module it hangs on,
the face of that module, and for a hinge the rotation in degrees. A part is
named after its kind and its place among the parts of that kind
(``hinge_0``, ``brick_0``, ``hinge_1``, ...); a table names the parent by that
name, and the core is ``core``.

Every builder in this package also writes the body down as a graph on
``CoreModule.graph`` (``start_graph`` and ``attach``): nodes with ``type``,
``rotation`` (degrees) and ``prefix`` (the start of the names of the module's
geoms; the core's is empty), edges with ``face``. The measures in
``ariel.body_phenotypes`` read it.
"""

# Standard library
from typing import Literal, cast

# Third-party libraries
import networkx as nx

# Local libraries
from ariel.body_phenotypes.robogen_lite.config import ModuleFaces
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule
from ariel.parameters import FrozenModel

type Piece = HingeModule | BrickModule
type Graph = nx.DiGraph[int]


class Part(FrozenModel):
    """One module of a body and where it is attached."""

    kind: Literal["hinge", "brick"]
    index: int
    parent: str
    face: ModuleFaces
    rotation: int | None = None


def hinge(
    index: int,
    parent: str,
    face: ModuleFaces,
    rotation: int | None = None,
) -> Part:
    """Describe a hinge.

    Parameters
    ----------
    index : int
        The index of the module in the body.
    parent : str
        The name of the module it hangs on (``core``, ``hinge_0``, ...).
    face : ModuleFaces
        The face of the parent that holds it.
    rotation : int | None
        The rotation of the hinge in degrees, if it is rotated.

    Returns
    -------
    Part
        The description.
    """
    return Part(
        kind="hinge",
        index=index,
        parent=parent,
        face=face,
        rotation=rotation,
    )


def brick(index: int, parent: str, face: ModuleFaces) -> Part:
    """Describe a brick.

    Parameters
    ----------
    index : int
        The index of the module in the body.
    parent : str
        The name of the module it hangs on (``core``, ``hinge_0``, ...).
    face : ModuleFaces
        The face of the parent that holds it.

    Returns
    -------
    Part
        The description.
    """
    return Part(kind="brick", index=index, parent=parent, face=face)


def start_graph(core: CoreModule) -> Graph:
    """Give a new body a graph that holds only its core.

    Parameters
    ----------
    core : CoreModule
        The core of the body being built.

    Returns
    -------
    nx.DiGraph[int]
        The graph, also stored on ``core.graph``.
    """
    graph: Graph = nx.DiGraph()
    graph.add_node(cast("int", core.index), type="CORE", rotation=0, prefix="")
    core.graph = graph
    return graph


def attach(
    core: CoreModule,
    parent: CoreModule | Piece,
    face: ModuleFaces,
    child: Piece,
    prefix: str,
    rotation: float = 0,
) -> None:
    """Attach ``child`` to ``parent`` and write it in ``core.graph``.

    Parameters
    ----------
    core : CoreModule
        The core of the body; its graph is started on the first call.
    parent : CoreModule | HingeModule | BrickModule
        The module it hangs on.
    face : ModuleFaces
        The face of ``parent`` that holds it.
    child : HingeModule | BrickModule
        The module attached.
    prefix : str
        The prefix of the names of its bodies and geoms.
    rotation : float
        The rotation of ``child`` in degrees, as recorded (the caller turns
        the module itself).
    """
    parent.sites[face].attach_body(body=child.body, prefix=prefix)
    graph = core.graph if core.graph is not None else start_graph(core)
    node = cast("int", child.index)
    graph.add_node(
        node, type=child.module_type.name, rotation=rotation, prefix=prefix
    )
    graph.add_edge(cast("int", parent.index), node, face=face.name)


def build(parts: tuple[Part, ...]) -> CoreModule:
    """Build a body from its table of parts.

    Parameters
    ----------
    parts : tuple[Part, ...]
        The parts, in the order they are attached.

    Returns
    -------
    CoreModule
        The core module of the finished robot body.
    """
    core = CoreModule(index=0)
    start_graph(core)
    named: dict[str, CoreModule | Piece] = {"core": core}
    # Steps from the core: a hinge's servo cable runs back to the core
    steps = {"core": 0}
    counts = {"hinge": 0, "brick": 0}
    for part in parts:
        name = f"{part.kind}_{counts[part.kind]}"
        counts[part.kind] += 1
        steps[name] = steps[part.parent] + 1
        module: Piece
        if part.kind == "hinge":
            module = HingeModule(index=part.index, steps_from_core=steps[name])
        else:
            module = BrickModule(index=part.index)
        if part.rotation is not None:
            module.rotate(part.rotation)
        attach(
            core,
            named[part.parent],
            part.face,
            module,
            name,
            part.rotation or 0,
        )
        named[name] = module
    return core
