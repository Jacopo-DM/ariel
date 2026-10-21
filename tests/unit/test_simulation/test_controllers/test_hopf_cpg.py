"""Test: HopfCPG and AdaptiveHopfCPG (Righetti and Ijspeert, 2006)."""

import math

import numpy as np
import pytest
import torch

from ariel.simulation import AdaptiveHopfCPG, HopfCPG

OMEGA = 2 * math.pi
TEACHING_HZ = 1.5


def _single[T: HopfCPG](model: type[T], w: float = OMEGA) -> T:
    cpg = model({0: []}, angle_limits=None, angle_tracking=True)
    cpg.set_param_with_dict({
        "phase": [0.0],
        "w": [w],
        "amplitudes": [1.0],
        "offset": [0.0],
    })
    cpg.reset_state()
    return cpg


def test_the_start_state_is_on_the_circle_at_the_phase() -> None:
    cpg = HopfCPG({0: []})
    cpg.set_params_by_group("phase", [math.pi / 2])
    cpg.reset_state()
    assert cpg.xy[0].tolist() == pytest.approx([0.0, 1.0], abs=1e-6)


def test_a_perturbed_oscillator_returns_to_the_circle() -> None:
    cpg = _single(HopfCPG)
    cpg.xy = torch.tensor([[0.2, 0.0]])
    for _ in range(100):
        cpg.forward()
    radius = torch.linalg.vector_norm(cpg.xy[0]).item()
    assert radius == pytest.approx(1.0, abs=0.01)


def test_the_offset_and_amplitude_shape_the_output() -> None:
    cpg = _single(HopfCPG)
    cpg.set_param_with_dict({"offset": [0.3], "amplitudes": [0.5]})
    assert cpg.forward(0.0).item() == pytest.approx(0.3 + 0.5)


def test_without_input_the_adaptive_frequency_stays() -> None:
    cpg = _single(AdaptiveHopfCPG)
    for _ in range(300):
        cpg.forward()
    assert cpg.frequency.item() == pytest.approx(OMEGA)
    assert np.allclose(
        cpg.angle_history, _history_of(_single(HopfCPG), 300), atol=1e-5
    )


def _history_of(cpg: HopfCPG, steps: int) -> list[list[float]]:
    for _ in range(steps):
        cpg.forward()
    return cpg.angle_history


def test_the_frequency_learns_a_periodic_input() -> None:
    cpg = AdaptiveHopfCPG({0: []}, epsilon=2.0, angle_limits=None)
    cpg.set_param_with_dict({
        "phase": [0.0],
        "w": [OMEGA],
        "amplitudes": [1.0],
        "offset": [0.0],
    })
    for step in range(20_001):
        time = step * 0.01
        cpg.set_input([math.sin(2 * math.pi * TEACHING_HZ * time)])
        cpg.forward(time)
    learned_hz = cpg.frequency.item() / (2 * math.pi)
    assert learned_hz == pytest.approx(TEACHING_HZ, abs=0.05)


def test_a_reset_restores_the_set_frequency_and_clears_the_input() -> None:
    cpg = _single(AdaptiveHopfCPG)
    cpg.set_input(np.ones(1))
    for _ in range(50):
        cpg.forward()
    assert cpg.frequency.item() != pytest.approx(OMEGA)
    cpg.reset_state()
    assert cpg.frequency.item() == pytest.approx(OMEGA)
    assert cpg.inputs.tolist() == [0.0]
