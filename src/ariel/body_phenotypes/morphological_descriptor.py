"""
MorphologicalMeasures class for robot phenotype digraph analysis.

Mostly based on the revolve2 implementation:
https://github.com/ci-group/revolve2/blob/master/standards/revolve2/standards/morphological_measures.py
"""

from __future__ import annotations

from ariel.body_phenotypes.morphological_counts import MorphologicalCounts


class MorphologicalMeasures(MorphologicalCounts):  # ruff: ignore[too-many-public-methods]  # a collection of measures, so many properties are expected
    """Modular robot morphological measures computed from a tree (blueprint).

    The grid and its helpers are in ``MorphologicalGrid``, the counts in
    ``MorphologicalCounts``; this class adds the proportions and the
    single-letter aliases (see those classes for the attributes).
    """

    # ------------------------------------------------------------------
    # Named proportions (intermediate, kept for backwards compatibility)
    # ------------------------------------------------------------------

    @property
    def filled_core_and_bricks_proportion(self) -> float:
        """Ratio of filled core+bricks to their theoretical maximum (branching).

        Returns
        -------
        float
            Value in [0, 1].
        """
        if self.max_potentionally_filled_core_and_bricks == 0:
            return 0.0
        return (
            len(self.filled_bricks) + (1 if self.core_is_filled else 0)
        ) / self.max_potentionally_filled_core_and_bricks

    @property
    def double_neighbour_brick_and_active_hinge_proportion(self) -> float:
        """Ratio of double-neighbour bricks+hinges to their theoretical maximum.

        Returns
        -------
        float
            Value in [0, 1].
        """
        if self.potential_double_neighbour_bricks_and_active_hinges == 0:
            return 0.0
        return (
            self.num_double_neighbour_bricks
            + self.num_double_neighbour_active_hinges
        ) / self.potential_double_neighbour_bricks_and_active_hinges

    @property
    def bounding_box_volume_coverage(self) -> float:
        """Fraction of the bounding-box volume occupied by modules.

        Returns
        -------
        float
            Value in (0, 1].
        """
        return self.num_modules / self.bounding_box_volume

    @property
    def branching(self) -> float:
        """Alias for ``filled_core_and_bricks_proportion``.

        Returns
        -------
        float
        """
        return self.filled_core_and_bricks_proportion

    @property
    def limbs(self) -> float:
        """Fraction of non-core leaf modules, normalised by l_max.

        Returns
        -------
        float
            Value in [0, 1].
        """
        if self.max_potential_single_neighbour_modules == 0:
            return 0.0
        return (
            self.num_single_neighbour_modules
            / self.max_potential_single_neighbour_modules
        )

    @property
    def length_of_limbs(self) -> float:
        """Alias for ``double_neighbour_brick_and_active_hinge_proportion``.

        Returns
        -------
        float
        """
        return self.double_neighbour_brick_and_active_hinge_proportion

    @property
    def coverage(self) -> float:
        """Alias for ``bounding_box_volume_coverage``.

        Returns
        -------
        float
        """
        return self.bounding_box_volume_coverage

    @property
    def proportion_2d(self) -> float:
        """Proportion measure for 2D robots only.

        Returns
        -------
        float
            min(depth, width) / max(depth, width), in (0, 1].
        """
        return min(self.bounding_box_depth, self.bounding_box_width) / max(
            self.bounding_box_depth,
            self.bounding_box_width,
        )

    @property
    def proportion(self) -> float:
        """Ground-plane footprint proportion; delegates to ``P``.

        Returns
        -------
        float
            min(depth, width) / max(depth, width), in (0, 1].
        """
        return self.P

    @property
    def symmetry(self) -> float:
        """Best-of-three plane symmetry score.

        Returns
        -------
        float
            max(xy_symmetry, xz_symmetry, yz_symmetry).
        """
        return max(self.xy_symmetry, self.xz_symmetry, self.yz_symmetry)

    @property
    def module_diversity(self) -> float:
        """Fraction of BRICK/HINGE edges that alternate module type.

        Returns
        -------
        float
            Value in [0, 1]; 0.0 if no BRICK/HINGE edge exists.
        """
        alt = total = 0
        for u, v in self.graph.edges():
            t1, t2 = self._get_node_type(u), self._get_node_type(v)
            if t1 in {"BRICK", "HINGE"} and t2 in {"BRICK", "HINGE"}:
                total += 1
                if t1 != t2:
                    alt += 1
        return alt / total if total > 0 else 0.0

    @property
    def joints(self) -> float:
        """Joint ratio J = num_joints / max_potential_joints.

        Returns
        -------
        float
            Value in [0, 1].
        """
        if self.max_potential_joints == 0:
            return 0.0
        return self.num_joints / self.max_potential_joints

    @property
    def size(self) -> float:
        """Size = num_modules / bounding_box_volume.

        Returns
        -------
        float
            Value in (0, 1].
        """
        # TODO: verify that bounding_box_volume is the right normaliser here  # ruff: ignore[line-contains-todo, missing-todo-author]
        return self.num_modules / self.bounding_box_volume

    @property
    def brick_cluster_ratio(self) -> float:
        """Fraction of brick-involved edges that connect two bricks.

        Complements ``D``: ``D`` measures global alternation; this penalises
        dense brick-to-brick clusters.

        Returns
        -------
        float
            0.0 = every brick connects only to hinges.
            1.0 = every brick connects only to other bricks.
        """
        brick_set = set(self.bricks)
        brick_total = brick_brick = 0
        for u, v in self.graph.edges():
            u_b, v_b = u in brick_set, v in brick_set
            if u_b or v_b:
                brick_total += 1
                if u_b and v_b:
                    brick_brick += 1
        return brick_brick / brick_total if brick_total > 0 else 0.0

    # ------------------------------------------------------------------
    # Paper-letter aliases
    # ------------------------------------------------------------------

    @property
    def B(self) -> float:  # ruff: ignore[invalid-function-name]
        """Branching B = b / b_max."""
        return self.branching

    @property
    def L(self) -> float:  # ruff: ignore[invalid-function-name]
        """Limbs L = l / l_max."""
        return self.limbs

    @property
    def E(self) -> float:  # ruff: ignore[invalid-function-name]
        """Extensiveness (length of limbs) E = e / e_max."""
        return self.length_of_limbs

    @property
    def C(self) -> float:  # ruff: ignore[invalid-function-name]
        """Coverage C = num_modules / bounding_box_volume."""
        return self.coverage

    @property
    def J(self) -> float:  # ruff: ignore[invalid-function-name]
        """Joints J = j / j_max."""
        return self.joints

    @property
    def S(self) -> float:  # ruff: ignore[invalid-function-name]
        """Symmetry S = max(xy, xz, yz symmetry)."""
        return self.symmetry

    @property
    def D(self) -> float:  # ruff: ignore[invalid-function-name]
        """Module diversity D = alternating BRICK/HINGE fraction."""
        return self.module_diversity

    @property
    def P(self) -> float:  # ruff: ignore[invalid-function-name]
        """Proportion P of the ground-plane (depth x width) footprint.

        Uses depth and width only, valid for both 2D and 3D robots.

        Returns
        -------
        float
            min(depth, width) / max(depth, width), in (0, 1].
        """
        d, w = self.bounding_box_depth, self.bounding_box_width
        return min(d, w) / max(d, w) if max(d, w) > 0 else 0.0
