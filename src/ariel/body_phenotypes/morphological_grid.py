"""Grid of a robot graph and the helpers behind the morphological measures."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    import networkx as nx
    from numpy.typing import NDArray


_FACES = ("FRONT", "BACK", "RIGHT", "LEFT", "TOP", "BOTTOM")

_ALLOWED_FACES = {
    "CORE": ["FRONT", "BACK", "RIGHT", "LEFT", "TOP", "BOTTOM"],
    "BRICK": ["FRONT", "RIGHT", "LEFT", "TOP", "BOTTOM"],
    "HINGE": ["FRONT"],
}


class MorphologicalGrid:
    """Modular robot morphological measures computed from a tree (blueprint).

    Works with a NetworkX directed graph representation of a robot.

    Inspired by the measures defined in:
        Miras, K., Haasdijk, E., Glette, K., Eiben, A.E. (2018).
        Search Space Analysis of Evolvable Robot Morphologies.
        EvoApplications 2018. LNCS vol 10784. Springer, Cham.
        https://doi.org/10.1007/978-3-319-77538-8_47

    Implementation adapted from the revolve2 implementation, with
    some measures renamed and others added for clarity and
    relevance to our experiments.

    Parameters
    ----------
    robot_graph : nx.DiGraph
        Directed graph of the robot phenotype.  Nodes require ``'type'`` and
        ``'rotation'`` attributes; edges require a ``'face'`` attribute.

    Attributes
    ----------
    graph : nx.DiGraph
        The input robot graph.
    grid : NDArray
        3D object array placing each node at its integer grid position.
    symmetry_grid : NDArray
        Padded version of grid centred on the core, used for symmetry checks.
    core_grid_position : NDArray[np.float64]
        Integer (x, y, z) index of the core module inside grid.
    is_2d : bool
        ``True`` when every module's rotation is a multiple of 90° in the XY
        plane (i.e. the robot has no vertical extent beyond its flat layout).
    core_node : Any
        The unique root node of graph (in-degree == 0).
    modules : list
        All nodes in graph.
    bricks : list
        Nodes whose ``'type'`` is ``'BRICK'``.
    active_hinges : list
        Nodes whose ``'type'`` is ``'HINGE'``.
    core_is_filled : bool
        ``True`` if every allowed face of the core has a child module.
    filled_bricks : list
        Bricks that have all their allowed faces occupied by children.
    filled_active_hinges : list
        Active hinges that have their one allowed face occupied by a child.
    single_neighbour_modules : list
        Non-core modules connected to exactly one other module (leaf nodes).
    single_neighbour_bricks : list
        Bricks with no children (leaf bricks).
    double_neighbour_bricks : list
        Bricks with exactly one child (internal chain nodes).
    double_neighbour_active_hinges : list
        Active hinges with exactly one child (internal chain nodes).
    xy_symmetry : float
        Degree of symmetry across the XY plane (Z-axis mirror), in [0, 1].
    xz_symmetry : float
        Degree of symmetry across the XZ plane (Y-axis mirror), in [0, 1].
    yz_symmetry : float
        Degree of symmetry across the YZ plane (X-axis mirror), in [0, 1].

    Raises
    ------
    ValueError
        If robot_graph is empty or contains more than one root node.
    """

    def __init__(self, robot_graph: nx.DiGraph[int]) -> None:
        """Create the measures for `robot_graph`; raise if it is empty.

        Raises
        ------
        ValueError
            If the robot graph is empty.
        """
        if robot_graph.number_of_nodes() == 0:
            msg = "Cannot analyze empty robot graph"
            raise ValueError(msg)
        for parent, child, data in robot_graph.edges(data=True):
            face = data.get("face")
            if face not in _FACES:
                msg = (
                    f"Edge {parent} -> {child} has no valid face ({face!r}); "
                    f"valid faces are {', '.join(_FACES)}"
                )
                raise ValueError(msg)

        self.graph = robot_graph
        self.grid, self.core_grid_position = self._graph_to_grid(robot_graph)
        self.core_node = self._find_core_node()
        self.is_2d = self._calculate_is_2d()
        self.modules = list(robot_graph.nodes())
        self.bricks = self._get_nodes_by_type("BRICK")
        self.active_hinges = self._get_nodes_by_type("HINGE")
        self.core_is_filled = self._calculate_core_is_filled()
        self.filled_bricks = self._calculate_filled_bricks()
        self.filled_active_hinges = self._calculate_filled_active_hinges()
        self.single_neighbour_bricks = self._calculate_single_neighbour_bricks()
        self.single_neighbour_modules = (
            self._calculate_single_neighbour_modules()
        )
        self.double_neighbour_bricks = self._calculate_double_neighbour_bricks()
        self.double_neighbour_active_hinges = (
            self._calculate_double_neighbour_active_hinges()
        )
        self._pad_grid()
        self.xy_symmetry = self._calculate_xy_symmetry()
        self.xz_symmetry = self._calculate_xz_symmetry()
        self.yz_symmetry = self._calculate_yz_symmetry()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_core_node(self) -> int:
        roots = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]
        if len(roots) != 1:
            msg = f"Expected exactly one root node, found {len(roots)}"
            raise ValueError(msg)
        return roots[0]

    def _get_nodes_by_type(self, module_type: str) -> list[int]:
        return [
            n
            for n in self.graph.nodes()
            if self.graph.nodes[n].get("type") == module_type
        ]

    def _calculate_is_2d(self) -> bool:
        valid = {"DEG_0", "DEG_90", "DEG_180", "DEG_270"}
        return all(
            self.graph.nodes[n].get("rotation", "DEG_0") in valid
            for n in self.graph.nodes()
        )

    def _get_node_type(self, node: int) -> str:
        return cast("str", self.graph.nodes[node].get("type", "UNKNOWN"))

    def _get_allowed_faces(self, node: int) -> list[str]:
        return _ALLOWED_FACES.get(self._get_node_type(node), [])

    def _get_node_connections(self, node: int) -> list[str]:
        return [
            self.graph.edges[node, successor]["face"]
            for successor in self.graph.successors(node)
        ]

    def _count_neighbors(self, node: int) -> int:
        return self.graph.in_degree(node) + self.graph.out_degree(node)

    def _graph_to_grid(
        self,
        robot_graph: nx.DiGraph[int],
    ) -> tuple[NDArray[np.object_], NDArray[np.float64]]:
        positions: dict[int, NDArray[np.float64]] = {}
        core_node = self._find_core_node()
        self._calculate_graph_positions(
            core_node, positions, np.array([0, 0, 0])
        )

        pos_array = np.array(list(positions.values()))
        min_pos = pos_array.min(axis=0)

        grid = np.full(pos_array.max(axis=0) - min_pos + 1, None, dtype=object)
        for node in robot_graph.nodes():
            grid[tuple(positions[node] - min_pos)] = node

        return grid, positions[core_node] - min_pos

    def _calculate_graph_positions(
        self,
        node: int,
        positions: dict[int, NDArray[np.float64]],
        pos: NDArray[np.float64],
    ) -> None:
        positions[node] = pos.copy()
        face_directions = {
            "FRONT": np.array([1, 0, 0]),
            "BACK": np.array([-1, 0, 0]),
            "RIGHT": np.array([0, 1, 0]),
            "LEFT": np.array([0, -1, 0]),
            "TOP": np.array([0, 0, 1]),
            "BOTTOM": np.array([0, 0, -1]),
        }
        for child in self.graph.successors(node):
            face = self.graph.edges[node, child]["face"]
            if child not in positions:
                self._calculate_graph_positions(
                    child, positions, pos + face_directions[face]
                )

    def _calculate_core_is_filled(self) -> bool:
        return len(self._get_node_connections(self.core_node)) == len(
            self._get_allowed_faces(self.core_node),
        )

    def _calculate_filled_bricks(self) -> list[int]:
        return [
            b
            for b in self.bricks
            if len(self._get_node_connections(b))
            == len(self._get_allowed_faces(b))
        ]

    def _calculate_filled_active_hinges(self) -> list[int]:
        return [
            h
            for h in self.active_hinges
            if len(self._get_node_connections(h))
            == len(self._get_allowed_faces(h))
        ]

    def _calculate_single_neighbour_bricks(self) -> list[int]:
        return [b for b in self.bricks if self.graph.out_degree(b) == 0]

    def _calculate_single_neighbour_modules(self) -> list[int]:
        return [
            n
            for n in self.modules
            if self._get_node_type(n) != "CORE"
            and self._count_neighbors(n) == 1
        ]

    def _calculate_double_neighbour_bricks(self) -> list[int]:
        return [b for b in self.bricks if self.graph.out_degree(b) == 1]

    def _calculate_double_neighbour_active_hinges(self) -> list[int]:
        return [h for h in self.active_hinges if self.graph.out_degree(h) == 1]

    def _pad_grid(self) -> None:
        x, y, z = self.grid.shape
        xo, yo, zo = self.core_grid_position
        self.symmetry_grid = np.full(
            (x + xo, y + yo, z + zo),
            None,
            dtype=object,
        )
        self.symmetry_grid[:x, :y, :z] = self.grid

    def _mirror_symmetry(self, axis: int) -> float:
        """Share of off-plane modules that have a matching mirror module.

        Parameters
        ----------
        axis : int
            The axis the mirror plane is perpendicular to (0 x, 1 y, 2 z).

        Returns
        -------
        float
            The number of mirrored modules of the same type divided by the
            number of modules off the plane through the core; 0.0 when every
            module is on the plane.
        """
        grid = np.moveaxis(self.symmetry_grid, axis, 0)
        centre = int(self.core_grid_position[axis])
        reach = max(centre, int(self.grid.shape[axis]) - 1 - centre)
        on_plane = sum(cell is not None for cell in grid[centre].ravel())
        off_sym = 0
        for layer in range(1, min(reach, centre) + 1):
            for pos, neg in zip(
                grid[centre + layer].ravel(),
                grid[centre - layer].ravel(),
                strict=True,
            ):
                if (
                    pos is not None
                    and neg is not None
                    and self._get_node_type(pos) == self._get_node_type(neg)
                ):
                    off_sym += 2
        diff = self.num_modules - on_plane
        return off_sym / diff if diff > 0 else 0.0

    def _calculate_xy_symmetry(self) -> float:
        return self._mirror_symmetry(2)

    def _calculate_xz_symmetry(self) -> float:
        return self._mirror_symmetry(1)

    def _calculate_yz_symmetry(self) -> float:
        return self._mirror_symmetry(0)

    # ------------------------------------------------------------------
    # Bounding-box geometry
    # ------------------------------------------------------------------

    @property
    def bounding_box_depth(self) -> int:
        """Extent along the forward/backward (X) axis in module-grid units.

        Returns
        -------
        int
        """
        return cast("int", self.grid.shape[0])

    @property
    def bounding_box_width(self) -> int:
        """Extent along the left/right (Y) axis in module-grid units.

        Returns
        -------
        int
        """
        return cast("int", self.grid.shape[1])

    @property
    def bounding_box_height(self) -> int:
        """Extent along the up/down (Z) axis in module-grid units.

        Returns
        -------
        int
        """
        return cast("int", self.grid.shape[2])

    @property
    def bounding_box_volume(self) -> int:
        """Product of all three bounding-box dimensions (m_area in the paper).

        Returns
        -------
        int
        """
        return (
            self.bounding_box_depth
            * self.bounding_box_width
            * self.bounding_box_height
        )

    # ------------------------------------------------------------------
    # Raw counts
    # ------------------------------------------------------------------

    @property
    def num_modules(self) -> int:
        """Total number of modules.

        Returns
        -------
        int
        """
        return len(self.modules)
