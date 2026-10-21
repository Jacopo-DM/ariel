"""Counts of a robot graph."""

from __future__ import annotations

from ariel.body_phenotypes.morphological_grid import MorphologicalGrid


class MorphologicalCounts(MorphologicalGrid):
    """Module counts and potential maxima of a robot graph."""

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
