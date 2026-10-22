"""Robot Olympics Competition template code."""

# Standard library
import argparse
from pathlib import Path

# Chooses the OpenGL backend before MuJoCo loads (CLAUDE.md, Things that bite)
import ariel.rendering  # ruff: ignore[unused-import]  # imported for its side effect

import mujoco as mj
import numpy as np

# Local libraries
from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders import (
    load_graph_from_json,
)
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.tracker import Tracker

# --- DATA SETUP --- #
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__"
DATA.mkdir(parents=True, exist_ok=True)

# Global variables
SPAWN_POS = [-0.8, 0, 0.1]
NUM_OF_MODULES = 30
TARGET_POSITION = [5, 0, 0.5]
TIME_OUT = 500

# --- COMMAND LINE --- #
parser = argparse.ArgumentParser(description="Score one robot in the arena.")
parser.add_argument(
    "--graph",
    type=Path,
    default=CWD / "YOUR_ROBOT_GRAPH.json",
    help="Robot graph to load (default: YOUR_ROBOT_GRAPH.json here)",
)
parser.add_argument(
    "--duration",
    type=int,
    default=TIME_OUT,
    help="Seconds to simulate and record",
)
args = parser.parse_args()

# Local scripts
from A3_plot_function import show_xpos_history
from A3_template import experiment, fitness_function


    # ==================================================================== #


def main() -> None:
    """Entry point."""
    # Load your robot graph here
    robot_graph = load_graph_from_json(args.graph)

    # Construct the robot from the graph
    core = construct_mjspec_from_graph(robot_graph)

    # Define a tracker to track the x-position of the robot
    mujoco_type_to_find = mj.mjtObj.mjOBJ_GEOM
    name_to_bind = "core"
    tracker = Tracker(
        mujoco_obj_to_find=mujoco_type_to_find,
        name_to_bind=name_to_bind,
    )

    # !!! Replace this with your controller function !!!
    # It takes (model, data) and returns one target angle per actuator;
    # this placeholder holds every joint at 0 so the file runs as given.
    def your_controller(model: mj.MjModel, _data: mj.MjData) -> np.ndarray:
        return np.zeros(model.nu)

    # Create the controller
    ctrl = Controller(
        controller_callback_function=your_controller,
        tracker=tracker,
    )

    # Simulate the robot
    experiment(
        robot=core, controller=ctrl, duration=args.duration, mode="video"
    )

    # Plot the x-position history
    show_xpos_history(
        tracker.history["xpos"][0],
        spawn_position=SPAWN_POS,
        target_position=TARGET_POSITION,
    )

    # Calculate and print the fitness of your robot
    fitness = fitness_function(tracker.history["xpos"][0])
    msg = f"Fitness of generated robot: {fitness}"
    console.log(msg)


if __name__ == "__main__":
    main()
