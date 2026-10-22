"""Launching the MuJoCo viewer."""

# Standard library
import argparse
from pathlib import Path

# Third-party libraries
import mujoco
import numpy as np
from mujoco import viewer
from rich.console import Console
from rich.traceback import install

# Local libraries
from ariel.simulation.environments import OlympicArena as World

# --- DATA SETUP --- #
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__"
DATA.mkdir(parents=True, exist_ok=True)

# --- RANDOM GENERATOR SETUP --- #
SEED = 42
RNG = np.random.default_rng(SEED)

# --- TERMINAL OUTPUT SETUP --- #
install(show_locals=False)
console = Console()

# --- COMMAND LINE --- #
parser = argparse.ArgumentParser(description="Open the MuJoCo viewer.")
parser.add_argument(
    "--visualize",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Open the viewer (--no-visualize steps 5 s without a window)",
)
args = parser.parse_args()


def main() -> None:
    """Entry point."""
    # Base world
    world = World()

    # Spawn a test object to validate the environment
    xml = r"""
    <mujoco>
    <worldbody>
        <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
        <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
    </worldbody>
    </mujoco>
    """
    test_object = mujoco.MjSpec.from_string(xml)
    world.spawn(test_object, correct_collision_with_floor=True)

    # Compile the model and create data
    model = world.compile()
    data = mujoco.MjData(model)

    # Open the viewer, or step five seconds without a window
    if args.visualize:
        viewer.launch(model, data)
    else:
        mujoco.mj_step(model, data, nstep=int(5 / model.opt.timestep))


if __name__ == "__main__":
    main()
