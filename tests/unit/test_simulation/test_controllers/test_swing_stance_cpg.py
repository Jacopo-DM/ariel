"""Test: SwingStanceCPG (Righetti and Ijspeert, 2008)."""

import math

import numpy as np
import pytest
import torch

from ariel.simulation import SwingStanceCPG

OMEGA = 2 * math.pi
LOCKED = 0.95  # correlation of two locked joints


def _network(
    n: int, phases: list[float], swing: float, stance: float
) -> SwingStanceCPG:
    adjacency = {i: [j for j in range(n) if j != i] for i in range(n)}
    cpg = SwingStanceCPG(adjacency, angle_limits=None, angle_tracking=True)
    cpg.set_param_with_dict({
        "phase": phases,
        "w_swing": [swing] * n,
        "w_stance": [stance] * n,
        "amplitudes": [1.0] * n,
        "offset": [0.0] * n,
    })
    cpg.reset_state()
    return cpg


def test_the_frequency_switches_with_the_sign_of_y() -> None:
    cpg = _network(2, [0.0, 0.0], swing=3.0, stance=1.0)
    cpg.xy = torch.tensor([[0.0, 0.5], [0.0, -0.5]])
    assert cpg.frequency().tolist() == pytest.approx([1.0, 3.0], abs=1e-6)


def test_the_stance_share_follows_the_two_frequencies() -> None:
    cpg = _network(1, [0.0], swing=3 * OMEGA, stance=OMEGA)
    stance = []
    for step in range(1, 1001):
        cpg.forward(step * 0.01)
        stance.append(cpg.xy[0, 1].item() > 0)
    share = np.mean(stance[200:])
    # pi / w_stance out of pi / w_stance + pi / w_swing
    assert share == pytest.approx(0.75, abs=0.02)


def test_half_a_cycle_apart_means_anti_phase() -> None:
    cpg = _network(2, [0.0, math.pi], swing=OMEGA, stance=OMEGA)
    cpg.xy = torch.tensor([[1.0, 0.0], [0.9, 0.1]])  # nearly in phase
    for _ in range(1000):
        cpg.forward()
    history = np.array(cpg.angle_history)[-200:]
    assert np.corrcoef(history.T)[0, 1] < -LOCKED


def test_the_same_phase_means_in_phase() -> None:
    cpg = _network(2, [0.0, 0.0], swing=OMEGA, stance=OMEGA)
    cpg.xy = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # a quarter cycle apart
    for _ in range(1000):
        cpg.forward()
    history = np.array(cpg.angle_history)[-200:]
    assert np.corrcoef(history.T)[0, 1] > LOCKED
