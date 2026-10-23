"""Companion module for process_pool.ipynb.

Holds ``evaluate_robot``, the function the notebook sends to
``ariel.simulation.process_pool``. A spawned worker process starts as a
fresh Python interpreter and imports the function it runs by its module
name; a function defined in a notebook cell lives in the kernel's
``__main__``, which a worker cannot import, so it lives here.
"""

import mujoco as mj
import nevergrad as ng
import numpy as np
import torch

from ariel.simulators import compile_model

SPAWN = (0.0, 0.0, 0.1)
TARGET = (2.0, 0.0, 0.1)
CMA_GEN = 10
CMA_POP = 5


class _Network:
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        self.W1 = torch.randn(hidden_size, input_size)
        self.b1 = torch.zeros(hidden_size)
        self.W2 = torch.randn(output_size, hidden_size)
        self.b2 = torch.zeros(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.W1 @ x + self.b1)
        return torch.tanh(self.W2 @ x + self.b2)

    def parameters(self) -> list[torch.Tensor]:
        return [self.W1, self.b1, self.W2, self.b2]


def _fill(net: _Network, params: np.ndarray) -> None:
    offset = 0
    for p in net.parameters():
        n = p.numel()
        p.data = torch.tensor(params[offset : offset + n], dtype=torch.float32).reshape(p.shape)
        offset += n


def evaluate_robot(job: tuple[str, int | None]) -> float:
    """Evaluate one robot: optimise a controller, return the best distance.

    ``job`` is the robot's MuJoCo XML and a seed. Fitness is the final
    distance to the target (lower is better).
    """
    xml, seed = job
    local_rng = np.random.default_rng(seed)
    try:
        model = compile_model(mj.MjSpec.from_string(xml))
        data = mj.MjData(model)
    except Exception:
        return float(np.linalg.norm(np.array(TARGET) - np.array(SPAWN)) * 10)

    state_size = len(data.qpos) + len(data.qvel) + 3
    net = _Network(state_size, 16, model.nu)
    num_vars = sum(p.numel() for p in net.parameters())
    opt = ng.optimizers.CMA(parametrization=num_vars, budget=CMA_GEN * CMA_POP)
    best = float("inf")

    for _ in range(CMA_GEN):
        candidates = [opt.ask() for _ in range(CMA_POP)]
        for cand in candidates:
            _fill(net, cand.value)
            mj.mj_resetData(model, data)
            data.ctrl[:] = local_rng.normal(scale=0.1, size=model.nu)
            mj.mj_step(model, data, nstep=300)
            displacement = data.qpos[:3].copy()
            target = tuple(np.array(TARGET) + displacement)

            def cb(m, d, _target=target, _net=net) -> None:
                pos = d.qpos[:3].copy()
                vec = np.array(_target) - pos
                dist = np.linalg.norm(vec) + 1e-6
                state = np.concatenate([d.qpos.copy(), d.qvel.copy(), vec / dist])
                ctrl = _net.forward(torch.tensor(state, dtype=torch.float32))
                d.ctrl[:] = ctrl.detach().numpy()[: m.nu]

            mj.set_mjcb_control(cb)
            mj.mj_step(model, data, nstep=1_000)
            mj.set_mjcb_control(None)

            fitness = float(np.linalg.norm(np.array(target) - data.qpos[:3]))
            opt.tell(cand, fitness)
            best = min(best, fitness)

    return best
