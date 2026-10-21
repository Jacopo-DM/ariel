"""Graph blueprint for ARIEL-robots.

Note
-----
* Graphs are represented as directed graphs (DiGraph) using NetworkX.
* Graphs are saved as JSON [1]_.

References
----------
.. [1] `NetworkX JSON Graph <https://networkx.org/documentation/stable/reference/readwrite/generated/networkx.readwrite.json_graph.tree_data.html#networkx.readwrite.json_graph.tree_data>`_

Todo
----
- [ ] DiGraph positioning use cartesian coordinates instead of spring layout
"""

# Evaluate type annotations in a deferred manner (ruff: UP037)
from __future__ import annotations

# Standard library
import json
from pathlib import Path

# Third-party libraries
import matplotlib.pyplot as plt
import networkx as nx
from networkx import DiGraph
from networkx.readwrite import json_graph

# Local libraries
from ariel.body_phenotypes.robogen_lite.config import (
    ModuleFaces,
    ModuleInstance,
    ModuleRotationsIdx,
    ModuleType,
)


class Blueprint:
    """Implements the high-probability-decoding algorithm."""

    # Set by each decoder before generate_networkx_graph runs
    nodes: set[int]
    edges: list[tuple[int, int, int]]
    type_dict: dict[int, ModuleType]
    rot_dict: dict[int, ModuleRotationsIdx]

    def __init__(self, num_modules: int) -> None:
        """
        Initialize the high-probability-decoding algorithm.

        Parameters
        ----------
        num_modules : int
            Number of modules to be decoded.
        """
        self.num_modules = num_modules

        # Data structure to hold the decoded graph (not networkx graph)
        self._graph: dict[int, ModuleInstance] = {}

        # NetworkX graph
        self.graph: DiGraph[int] = nx.DiGraph()

    def generate_networkx_graph(self) -> None:
        """Generate a NetworkX graph from the decoded graph."""
        for node in self.nodes:
            self.graph.add_node(
                node,
                type=self.type_dict[node].name,
                rotation=self.rot_dict[node].name,
            )
        for edges in self.edges:
            parent, child, face = edges
            self.graph.add_edge(
                parent,
                child,
                face=ModuleFaces(face).name,
            )


def save_graph_as_json(
    graph: DiGraph[int],
    save_file: Path | str | None = None,
) -> None:
    """
    Save a directed graph as a JSON file.

    Parameters
    ----------
    graph : DiGraph
        The directed graph to save.
    save_file : Path | str | None, optional
        The file path to save the graph JSON, by default None
    """
    if save_file is None:
        return

    # The networkx stub's deprecated overload (link=) matches every call,
    # also this one, which uses edges= as networkx asks
    data = json_graph.node_link_data(graph, edges="edges")  # ty: ignore[deprecated]  # pyright: ignore[reportDeprecated]
    json_string = json.dumps(data, indent=4)

    Path(save_file).write_text(json_string, encoding="utf-8")


def load_graph_from_json(
    load_file: Path | str,
) -> DiGraph[int]:
    """
    Load a directed graph from a JSON file.

    Parameters
    ----------
    load_file : Path | str
        The file path to load the graph JSON.

    Returns
    -------
    DiGraph
        The loaded directed graph.

    Raises
    ------
    ValueError
        If the file is not UTF-8 JSON in node-link form, or it declares
        an undirected graph or a multigraph.
    """
    with Path(load_file).open("r", encoding="utf-8") as f:
        data = json.load(f)
    try:
        graph: DiGraph[int] = json_graph.node_link_graph(
            data,
            directed=True,
            multigraph=False,
            edges="edges",
        )
    except (KeyError, TypeError, AttributeError) as error:
        msg = f"not a node-link graph file: {load_file}"
        raise ValueError(msg) from error
    # The file's own "directed" and "multigraph" keys override the
    # arguments above, so check what came back.
    if not graph.is_directed() or graph.is_multigraph():
        msg = f"not a simple directed graph file: {load_file}"
        raise ValueError(msg)
    return graph


def draw_graph(
    graph: DiGraph[int],
    title: str = "NetworkX Directed Graph",
    save_file: Path | str | None = None,
    *,
    seed: int = 42,
    dpi: int = 300,
) -> None:
    """
    Draw a directed graph.

    Parameters
    ----------
    graph : DiGraph
        The directed graph to draw.
    title : str
        The title of the graph.
    save_file : Path | str | None, optional
        The file path to save the graph image, by default None
    """
    plt.figure()

    pos = nx.spectral_layout(graph)

    pos = nx.spring_layout(graph, pos=pos, k=1, iterations=20, seed=seed)

    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_size=150,
        node_color="#FFFFFF00",
        edgecolors="blue",
        font_size=8,
        width=0.5,
    )

    edge_labels = nx.get_edge_attributes(graph, "face")

    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        font_color="red",
        font_size=8,
    )

    plt.title(title)

    # Save the graph visualization
    if save_file:
        plt.savefig(save_file, dpi=dpi)
    else:
        # Show the plot
        plt.show()
