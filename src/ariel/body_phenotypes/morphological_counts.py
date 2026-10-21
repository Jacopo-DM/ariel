"""Counts of a robot graph, the base of the morphological measures."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from ariel.body_phenotypes.robogen_lite import CoreModule


_FACES = ("FRONT", "BACK", "RIGHT", "LEFT", "TOP", "BOTTOM")

_ALLOWED_FACES = {
    "CORE": ["FRONT", "BACK", "RIGHT", "LEFT", "TOP", "BOTTOM"],
    "BRICK": ["FRONT", "RIGHT", "LEFT", "TOP", "BOTTOM"],
    "HINGE": ["FRONT"],
}


class MorphologicalCounts:
    """Module counts and potential maxima of a built robot's graph.

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
    robot : CoreModule
        A built robot that carries its graph (``CoreModule.graph``): nodes
        with ``'type'``, edges with ``'face'``.

    Attributes
    ----------
    graph : nx.DiGraph
        The input robot graph.
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

    Raises
    ------
    ValueError
        If the robot has no graph, or its graph is empty or has more than
        one root node.
    """

    def __init__(self, robot: CoreModule) -> None:
        """Create the measures for `robot`; raise if its graph is unusable.

        Raises
        ------
        ValueError
            If the robot has no graph or its graph is empty.
        """
        if robot.graph is None:
            msg = "The robot has no graph; build it with a builder of ARIEL."
            raise ValueError(msg)
        robot_graph = robot.graph
        self.robot = robot
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
        self.core_node = self._find_core_node()
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

    @property
    def num_bricks(self) -> int:
        """Number of BRICK-type modules.

        Returns
        -------
        int
        """
        return len(self.bricks)

    @property
    def num_active_hinges(self) -> int:
        """Number of HINGE-type modules.

        Returns
        -------
        int
        """
        return len(self.active_hinges)

    @property
    def num_joints(self) -> int:
        """Alias for ``num_active_hinges``.

        Returns
        -------
        int
        """
        return self.num_active_hinges

    @property
    def num_filled_bricks(self) -> int:
        """Number of bricks with every allowed face occupied.

        Returns
        -------
        int
        """
        return len(self.filled_bricks)

    @property
    def num_filled_active_hinges(self) -> int:
        """Number of active hinges with their allowed face occupied.

        Returns
        -------
        int
        """
        return len(self.filled_active_hinges)

    @property
    def num_filled_modules(self) -> int:
        """
        Total fully-saturated modules (filled bricks, hinges and core).

        Returns
        -------
        int
        """
        return (
            self.num_filled_bricks
            + self.num_filled_active_hinges
            + (1 if self.core_is_filled else 0)
        )

    @property
    def num_single_neighbour_modules(self) -> int:
        """Number of non-core leaf modules (connected to exactly one module).

        Returns
        -------
        int
        """
        return len(self.single_neighbour_modules)

    @property
    def num_double_neighbour_bricks(self) -> int:
        """Number of bricks connected to exactly two modules.

        Returns
        -------
        int
        """
        return len(self.double_neighbour_bricks)

    @property
    def num_double_neighbour_active_hinges(self) -> int:
        """Number of active hinges connected to exactly two modules.

        Returns
        -------
        int
        """
        return len(self.double_neighbour_active_hinges)

    # ------------------------------------------------------------------
    # Theoretical maxima
    # ------------------------------------------------------------------

    @property
    def max_potentionally_filled_core_and_bricks(self) -> int:
        """Upper bound on filled core+bricks given this module set (b_max).

        Returns
        -------
        int
        """
        return min(max(0, (self.num_modules - 2) // 3), 1 + self.num_bricks)

    @property
    def max_potential_single_neighbour_modules(self) -> int:
        """
        Upper bound on single-neighbour modules given this module set (l_max).

        Returns
        -------
        int
        """
        return self.num_modules - 1 - max(0, (self.num_modules - 3) // 3)

    @property
    def max_potential_joints(self) -> int:
        """
        Maximum possible joints if every inter-module connection were a hinge.

        Returns
        -------
        int
            ``num_modules - 1``, or 0 for a single-module robot.
        """
        return max(0, self.num_modules - 1)

    @property
    def potential_double_neighbour_bricks_and_active_hinges(self) -> int:
        """
        Upper bound on double-neighbour bricks+hinges (e_max).

        Returns
        -------
        int
        """
        return max(0, self.num_bricks + self.num_active_hinges - 1)
