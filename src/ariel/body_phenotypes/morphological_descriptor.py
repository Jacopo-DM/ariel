"""
MorphologicalMeasures class for robot phenotype digraph analysis.

Mostly based on the revolve2 implementation:
https://github.com/ci-group/revolve2/blob/master/standards/revolve2/standards/morphological_measures.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ariel.body_phenotypes.morphological_counts import MorphologicalCounts
from ariel.body_phenotypes.robogen_lite import (
    IDX_OF_CORE,
    module_sizes,
    rest_boxes,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ariel.body_phenotypes.robogen_lite import CoreModule

# Axes of the core's frame: FRONT points along +y, RIGHT along x, TOP along +z
DEPTH_AXIS = 1
WIDTH_AXIS = 0
HEIGHT_AXIS = 2
FLAT_TOLERANCE = 1e-3  # metres a module may reach past the core and stay flat


class MorphologicalMeasures(MorphologicalCounts):  # ruff: ignore[too-many-public-methods]  # a collection of measures, so many properties are expected
    """Morphological measures of a built robot.

    The counts come from the graph the robot was built from
    (``CoreModule.graph``, see ``MorphologicalCounts``); the size measures
    from its geoms at joint angle zero, in metres, along the core's own
    axes: depth from FRONT to BACK, width from LEFT to RIGHT, height from
    TOP to BOTTOM. This class adds the proportions and the single-letter
    aliases.

    Symmetry, per plane through the centre of the core: a module is on the
    plane when its centre (the volume-weighted centre of its geoms) is
    within half its thinnest side of it; an off-plane module is mirrored
    when a module of the same type sits at its mirror point, within half the
    thinner of the two thinnest sides. The score is the share of off-plane
    modules that are mirrored, 0.0 when no module is off the plane.

    Parameters
    ----------
    robot : CoreModule
        A robot from ``construct_mjspec_from_graph`` or a prebuilt builder.
    """

    def __init__(self, robot: CoreModule) -> None:
        """Measure ``robot``."""
        super().__init__(robot)
        boxes = rest_boxes(robot)
        centres = boxes.poses[:, :3, 3]
        reach = np.einsum(
            "nij,nj->ni", np.abs(boxes.poses[:, :3, :3]), boxes.halves
        )
        low, high = centres - reach, centres + reach
        self.extent = high.max(axis=0) - low.min(axis=0)
        volumes = np.prod(boxes.halves, axis=1) * 8
        self.module_volume = float(volumes.sum())
        self.centres: dict[int, NDArray[np.float64]] = {}
        for node in set(boxes.owners):
            mine = np.array(boxes.owners) == node
            self.centres[node] = (
                volumes[mine] @ centres[mine] / volumes[mine].sum()
            )
        self.thinnest = {
            node: thinnest
            for node, (_, thinnest) in module_sizes(
                boxes.owners, boxes.halves
            ).items()
        }
        self.xy_symmetry = self._mirror_symmetry(HEIGHT_AXIS)
        self.xz_symmetry = self._mirror_symmetry(DEPTH_AXIS)
        self.yz_symmetry = self._mirror_symmetry(WIDTH_AXIS)
        core = np.array(boxes.owners) == IDX_OF_CORE
        self.reaches_past_core = bool(
            high[:, HEIGHT_AXIS].max()
            > high[core, HEIGHT_AXIS].max() + FLAT_TOLERANCE
            or low[:, HEIGHT_AXIS].min()
            < low[core, HEIGHT_AXIS].min() - FLAT_TOLERANCE
        )

    def _mirror_symmetry(self, axis: int) -> float:
        """Share of off-plane modules that have a mirror partner.

        Parameters
        ----------
        axis : int
            The axis the mirror plane is perpendicular to (0 x, 1 y, 2 z).

        Returns
        -------
        float
            Mirrored off-plane modules over off-plane modules, in [0, 1].
        """
        plane = self.centres[IDX_OF_CORE][axis]
        off = [
            node
            for node, centre in self.centres.items()
            if abs(centre[axis] - plane) > self.thinnest[node] / 2
        ]
        mirrored = 0
        for node in off:
            image = self.centres[node].copy()
            image[axis] = 2 * plane - image[axis]
            mirrored += any(
                other != node
                and self._get_node_type(other) == self._get_node_type(node)
                and np.linalg.norm(centre - image)
                <= min(self.thinnest[node], self.thinnest[other]) / 2
                for other, centre in self.centres.items()
            )
        return mirrored / len(off) if off else 0.0

    # ------------------------------------------------------------------
    # Bounding box, in metres
    # ------------------------------------------------------------------

    @property
    def bounding_box_depth(self) -> float:
        """Extent from FRONT to BACK of the core, in metres.

        Returns
        -------
        float
        """
        return float(self.extent[DEPTH_AXIS])

    @property
    def bounding_box_width(self) -> float:
        """Extent from LEFT to RIGHT of the core, in metres.

        Returns
        -------
        float
        """
        return float(self.extent[WIDTH_AXIS])

    @property
    def bounding_box_height(self) -> float:
        """Extent from TOP to BOTTOM of the core, in metres.

        Returns
        -------
        float
        """
        return float(self.extent[HEIGHT_AXIS])

    @property
    def bounding_box_volume(self) -> float:
        """Depth times width times height, in cubic metres.

        Returns
        -------
        float
        """
        return (
            self.bounding_box_depth
            * self.bounding_box_width
            * self.bounding_box_height
        )

    @property
    def is_2d(self) -> bool:
        """Whether no module reaches above or below the core.

        Returns
        -------
        bool
        """
        return not self.reaches_past_core

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
        """Fraction of the bounding-box volume the modules fill.

        Returns
        -------
        float
            The summed volume of the geoms over ``bounding_box_volume``,
            in (0, 1].
        """
        return self.module_volume / self.bounding_box_volume

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
        """Size, the same as ``coverage``.

        Returns
        -------
        float
            Value in (0, 1].
        """
        return self.coverage

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
        """Coverage C = module volume / bounding_box_volume."""
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
        return min(d, w) / max(d, w)
