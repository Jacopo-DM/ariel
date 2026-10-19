"""Decode a CPPN genome into a robot body and simulate it."""

from pathlib import Path

import mujoco
import numpy as np
from mujoco import viewer
from rich.console import Console

from ariel.body_phenotypes import (
    NUM_OF_ROTATIONS,
    NUM_OF_TYPES_OF_MODULES,
    CoreModule,
    Genome,
    IdManager,
    MorphologyDecoderBestFirst,
    construct_mjspec_from_graph,
)
from ariel.rendering import single_frame_renderer
from ariel.simulation import (
    Controller,
    NaCPG,
    SimpleFlatWorld,
    create_fully_connected_adjacency,
)

SCRIPT_NAME = Path(__file__).stem
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True, parents=True)
SEED = 42
console = Console()
RNG = np.random.default_rng(SEED)
np.set_printoptions(precision=3, suppress=True)


def run(robot: CoreModule, *, with_viewer: bool = False) -> None:
    """Spawn `robot` in a flat world and simulate it, optionally in a viewer."""
    world = SimpleFlatWorld()
    world.spawn(robot.spec)
    model = world.spec.compile()
    data = mujoco.MjData(model)

    # with (DATA / f"{SCRIPT_NAME}.xml").open("w", encoding="utf-8") as f:

    mujoco.mj_resetData(model, data)
    single_frame_renderer(model, data, steps=10)

    adj_dict = create_fully_connected_adjacency(len(data.ctrl.copy()))
    na_cpg_mat = NaCPG(adj_dict)

    def callback(
        _model: mujoco.MjModel,
        data: mujoco.MjData,
        /,
        *_args: object,
        **_kwargs: object,
    ) -> object:
        return na_cpg_mat.forward(data.time)

    ctrl = Controller(controller_callback_function=callback)
    ctrl.tracker.setup(world.spec, data)

    mujoco.set_mjcb_control(ctrl.set_control)
    mujoco.mj_resetData(model, data)

    if with_viewer:
        viewer.launch(model=model, data=data)


if __name__ == "__main__":
    DECODER_TYPE = "best_first"
    MAX_MODULES = 5
    num_initial_mutations = 5

    T, R = NUM_OF_TYPES_OF_MODULES, NUM_OF_ROTATIONS
    NUM_CPPN_INPUTS, NUM_CPPN_OUTPUTS = 6, 1 + T + R

    # 1. Define the starting innovation ID for the first genome as 0.
    initial_innov_id = 0

    # 2. Calculate how many connections will be created.
    num_initial_conns = NUM_CPPN_INPUTS * NUM_CPPN_OUTPUTS

    # 3. Initialize the IdManager. Its last-used ID is (num_conns - 1).
    id_manager = IdManager(
        node_start=NUM_CPPN_INPUTS + NUM_CPPN_OUTPUTS - 1,
        innov_start=num_initial_conns
        - 1,  # we have 78 conns (6 inputs * 13 outputs) --> last ID is 77
    )

    # 4. Create the initial random genome.
    my_cppn_genome = Genome.random(
        num_inputs=NUM_CPPN_INPUTS,
        num_outputs=NUM_CPPN_OUTPUTS,
        next_node_id=(NUM_CPPN_INPUTS + NUM_CPPN_OUTPUTS),
        next_innov_id=initial_innov_id,  # This is now 0
    )

    # 5. Apply initial mutations to the genome.
    # (this was done in old revolve2, and it makes sense to keep it but it's
    # optional)
    for _ in range(num_initial_mutations):
        my_cppn_genome.mutate(
            1.0,  # Use floats for rates
            1.0,  # Use floats for rates
            id_manager.get_next_innov_id,
            id_manager.get_next_node_id,
        )

    # 6. Decode the genome into a robot morphology.

    decoder = MorphologyDecoderBestFirst(
        cppn_genome=my_cppn_genome, max_modules=MAX_MODULES
    )
    decoded_robot_graph = decoder.decode()

    final_robot_graph = decoded_robot_graph

    # 7. Construct and run the final robot.
    core = construct_mjspec_from_graph(final_robot_graph)
    run(core, with_viewer=True)
