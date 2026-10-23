"""Example: parallel robot evaluation with ``ariel.simulation.process_pool``.

Scores robots on a targeted-locomotion task with a CMA-ES-optimised ANN
controller, each robot in its own worker process. The function the pool
runs, ``evaluate``, lives at the top level of this file: a spawned worker
imports it by module name, which a function defined in a notebook cell
does not allow.
"""

import mujoco as mj
import nevergrad as ng
import numpy as np
import torch
from numpy.typing import NDArray

from ariel.parameters import FrozenModel
from ariel.simulation import process_pool
from ariel.simulators import compile_model

# ---------------------------------------------------------------------------
# Minimal ANN controller (experiment-specific)
# ---------------------------------------------------------------------------


class _Network:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
    ) -> None:
        self.W1 = torch.randn(hidden_size, input_size)
        self.b1 = torch.zeros(hidden_size)
        self.W2 = torch.randn(output_size, hidden_size)
        self.b2 = torch.zeros(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.W1 @ x + self.b1)
        return torch.tanh(self.W2 @ x + self.b2)

    def parameters(self) -> list[torch.Tensor]:
        return [self.W1, self.b1, self.W2, self.b2]


def _fill_parameters(net: _Network, params: NDArray[np.float64]) -> None:
    offset = 0
    for p in net.parameters():
        numel = p.numel()
        p.data = torch.tensor(
            params[offset : offset + numel], dtype=torch.float32
        ).reshape(p.shape)
        offset += numel


# ---------------------------------------------------------------------------
# What each job carries
# ---------------------------------------------------------------------------


class LocomotionConfig(FrozenModel):
    """Settings of one evaluation: where the robot starts and aims, and CMA-ES."""

    spawn_position: tuple[float, float, float]
    target_position: tuple[float, float, float]
    seed: int | None = None
    cma_generations: int = 20
    cma_pop_size: int = 5
    hidden_size: int = 16


# ---------------------------------------------------------------------------
# The function each worker runs
# ---------------------------------------------------------------------------


def evaluate(job: tuple[str, LocomotionConfig]) -> float:
    """Build the robot, optimise a controller with CMA-ES, return the best.

    Fitness is the distance to the target after the controller has run
    (lower is better).

    Returns
    -------
    float
        The best fitness found.
    """
    xml, config = job
    model = compile_model(mj.MjSpec.from_string(xml))
    data = mj.MjData(model)
    rng = np.random.default_rng(config.seed)

    state_size = len(data.qpos) + len(data.qvel) + 3
    net = _Network(state_size, config.hidden_size, model.nu)
    num_vars = sum(p.numel() for p in net.parameters())

    opt = ng.optimizers.CMA(
        parametrization=num_vars,
        budget=config.cma_generations * config.cma_pop_size,
    )

    min_fitness = float("inf")

    for _ in range(config.cma_generations):
        candidates = [opt.ask() for _ in range(config.cma_pop_size)]
        for cand in candidates:
            _fill_parameters(net, cand.value)
            mj.mj_resetData(model, data)

            # Warmup
            data.ctrl[:] = rng.normal(scale=0.1, size=model.nu)
            mj.mj_step(model, data, nstep=300)

            displacement = data.qpos[:3].copy()
            target = tuple(np.array(config.target_position) + displacement)

            def cb(m: mj.MjModel, d: mj.MjData) -> None:
                pos = d.qpos[:3].copy()
                vec = np.array(target) - pos  # ruff: ignore[function-uses-loop-variable]
                dist = np.linalg.norm(vec) + 1e-6
                state = np.concatenate([
                    d.qpos.copy(),
                    d.qvel.copy(),
                    vec / dist,
                ])
                ctrl = net.forward(torch.tensor(state, dtype=torch.float32))
                d.ctrl[:] = ctrl.detach().numpy()[: m.nu]

            mj.set_mjcb_control(cb)
            mj.mj_step(model, data, nstep=1_000)
            mj.set_mjcb_control(None)

            fitness = float(
                np.linalg.norm(np.array(target) - data.qpos[:3])
            )
            opt.tell(cand, fitness)
            min_fitness = min(min_fitness, fitness)

    return min_fitness


# ---------------------------------------------------------------------------
# How to use it in an evolution loop
# ---------------------------------------------------------------------------


def evaluate_population(
    jobs: list[tuple[str, LocomotionConfig]],
    num_workers: int = 8,
) -> list[float]:
    """Run the jobs through a process pool and return fitness values.

    Returns
    -------
    list[float]
        The fitness of each job, in the order of the jobs.
    """
    with process_pool(workers=num_workers) as pool:
        return list(pool.map(evaluate, jobs))


if __name__ == "__main__":
    # Dummy smoke-test: two identical placeholder XML strings.
    # Replace with real robot XML from construct_mjspec_from_graph +
    # SimpleFlatWorld.
    dummy_xml = """
    <mujoco>
        <worldbody>
        <body name="core" pos="0 0 0.1">
            <joint type="free"/>
            <geom type="box" size="0.1 0.1 0.05" mass="1"/>
            <body name="leg" pos="0.1 0 0">
            <joint name="hinge" type="hinge" axis="0 1 0"/>
            <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.1"/>
            </body>
        </body>
        </worldbody>
        <actuator>
        <motor joint="hinge" gear="10"/>
        </actuator>
    </mujoco>
    """

    config = LocomotionConfig(
        spawn_position=(0.0, 0.0, 0.1),
        target_position=(2.0, 0.0, 0.1),
        seed=42,
    )

    jobs = [(dummy_xml, config), (dummy_xml, config)]
    results = evaluate_population(jobs, num_workers=2)
    print("Fitness values:", results)
