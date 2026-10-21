"""Simulation tasks for evaluating robot performance."""

from ariel.simulation.tasks.gait_learning import (
    x_speed,
    xy_displacement,
    y_speed,
)
from ariel.simulation.tasks.targeted_locomotion import distance_to_target
from ariel.simulation.tasks.turning_in_place import turning_in_place

__all__ = [
    "distance_to_target",
    "turning_in_place",
    "x_speed",
    "xy_displacement",
    "y_speed",
]
