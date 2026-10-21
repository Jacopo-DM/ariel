"""Build a MuJoCo robot specification from a graph of modules."""

# Handle forward references in type hints
from __future__ import annotations

# Standard library
from typing import TYPE_CHECKING, cast

# Third-party libraries
import networkx as nx

# Local libraries
from ariel.body_phenotypes.robogen_lite.config import (
    IDX_OF_CORE,
    ModuleFaces,
    ModuleRotationsTheta,
    ModuleType,
)
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule
from ariel.body_phenotypes.robogen_lite.overlap import (
    repair_overlaps as repair_graph_overlaps,
)

# Type checking
if TYPE_CHECKING:
    from networkx import DiGraph

    from ariel.body_phenotypes.robogen_lite.modules.module import Module


def construct_mjspec_from_graph(
    graph: DiGraph[int], *, repair_overlaps: bool = False
) -> CoreModule:
    """
    Construct a MuJoCo specification from a graph representation.

    Can be used for constructing the body of a robot after crossover using the
    graph representation.

    Parameters
    ----------
    graph : Graph
        A graph representation of the robot's structure.
    repair_overlaps : bool
        When True, modules that sit inside each other are pruned first (the
        deeper module of an overlapping pair goes, with everything attached
        beyond it) and the body is built from the pruned copy; the graph
        itself is not changed. See the ``overlap`` module of this package.

    Returns
    -------
    CoreModule
        The core module of the robot, which contains all other modules;
        an unknown module type raises ``ValueError``.
    """
    if repair_overlaps:
        graph = repair_graph_overlaps(graph, _construct)
    return _construct(graph)


def _construct(graph: DiGraph[int]) -> CoreModule:
    """
    Build the body of a graph as it is.

    Parameters
    ----------
    graph : Graph
        A graph representation of the robot's structure.

    Returns
    -------
    CoreModule
        The core module of the robot, which contains all other modules.

    Raises
    ------
    ValueError
        If the graph contains unknown module types.
    """
    # Steps from the core: a hinge's servo cable runs back to the core
    steps = (
        nx.single_source_shortest_path_length(graph, IDX_OF_CORE)
        if IDX_OF_CORE in graph
        else {}
    )
    modules: dict[int, Module | None] = {}
    for node in graph.nodes:
        # Extract module type and rotation from the graph node
        module_type = graph.nodes[node]["type"]
        module_rotation = graph.nodes[node]["rotation"]

        # Create the module based on its type
        module: Module | None
        match module_type:
            case ModuleType.CORE.name:
                module = CoreModule(index=IDX_OF_CORE)
            case ModuleType.HINGE.name:
                module = HingeModule(
                    index=node, steps_from_core=max(steps.get(node, 1), 1)
                )
            case ModuleType.BRICK.name:
                module = BrickModule(index=node)
            case ModuleType.NONE.name:
                module = None
            case _:
                msg = f"Unknown module type: {module_type}"
                raise ValueError(msg)

        # Check that the module is not None
        if module is not None:
            rotation_angle = ModuleRotationsTheta[module_rotation].value
            module.rotate(rotation_angle)
            modules[node] = module
        else:
            modules[node] = None

    # Attach bodies to modules based on the graph edges
    for edge in graph.edges:
        # Extract the from and to modules and the face from the edge
        from_module = edge[0]
        to_module = edge[1]
        face = graph.edges[edge]["face"]

        # Check if both modules exist (not 'None')
        if modules[to_module] is not None:
            # A graph from a decoder has parts below the core and no edge
            # from an empty node; the casts leave any other graph failing
            # with AttributeError, as before.
            parent = cast("Module", modules[from_module])
            child = cast("BrickModule | HingeModule", modules[to_module])
            parent.sites[ModuleFaces[face]].attach_body(
                body=child.body,
                prefix=f"{parent.index}-{child.index}-{ModuleFaces[face].value}-",
            )

    core_module = modules[IDX_OF_CORE]
    if isinstance(core_module, CoreModule):
        core_module.graph = _with_prefixes(graph)
        return core_module

    msg = "The core module is not of type CoreModule."
    raise ValueError(msg)


def _with_prefixes(graph: DiGraph[int]) -> DiGraph[int]:
    """Copy a graph and give every node the prefix of its geom names.

    Parameters
    ----------
    graph : Graph
        The graph a body was built from.

    Returns
    -------
    Graph
        A copy whose nodes carry ``prefix``: ``<parent>-<node>-<face>-`` as
        ``_construct`` names them, empty for the core.
    """
    copy = graph.copy()
    for node in copy.nodes:
        copy.nodes[node]["prefix"] = ""
    for parent, child in copy.edges:
        face = ModuleFaces[copy.edges[parent, child]["face"]]
        copy.nodes[child]["prefix"] = f"{parent}-{child}-{face.value}-"
    return copy
