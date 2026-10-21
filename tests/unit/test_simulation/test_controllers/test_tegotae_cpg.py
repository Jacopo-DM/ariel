"""Test: TegotaeCPG and LearnableTegotaeCPG (Owaki; Herneth et al., 2023)."""

import math

import pytest
import torch

from ariel.simulation import LearnableTegotaeCPG, TegotaeCPG
from ariel.simulation.controllers.cpg.tegotae import mlp

OMEGA = 2 * math.pi
LOAD = 5.0  # newtons
HIDDEN = 4
NETWORK_WEIGHTS = 51  # (2 + 2) 4 + 1, (1 + 2) 4 + 1, (3 + 2) 4 + 1


def _single(sigma: float) -> TegotaeCPG:
    cpg = TegotaeCPG({0: []}, angle_limits=None)
    cpg.set_param_with_dict({
        "phase": [0.0],
        "w": [OMEGA],
        "amplitudes": [1.0],
        "offset": [0.0],
        "sigma": [sigma],
    })
    return cpg


def _frequency_under_a_body(cpg: TegotaeCPG, body_hz: float) -> float:
    """Feed the load of a limb that is down every other half cycle.

    Returns
    -------
    float
        The oscillator's mean frequency over the last 15 of 30 s, in Hz.
    """
    phases = []
    for step in range(3001):
        time = step * 0.01
        down = max(0.0, -math.sin(2 * math.pi * body_hz * time))
        cpg.set_input([LOAD * down])
        cpg.forward(time)
        phases.append(cpg.theta.item())
    return (phases[-1] - phases[1500]) / (15 * 2 * math.pi)


def test_the_feedback_is_minus_sigma_load_cos_phase() -> None:
    cpg = TegotaeCPG({0: [], 1: []})
    cpg.set_params_by_group("sigma", [0.5, 2.0])
    cpg.theta = torch.tensor([0.0, math.pi / 3])
    cpg.set_input([2.0, 3.0])
    expected = [-0.5 * 2.0, -2.0 * 3.0 * 0.5]
    assert cpg.feedback().tolist() == pytest.approx(expected)


@pytest.mark.parametrize("body_hz", [0.8, 1.2])
def test_the_oscillator_entrains_to_the_body(body_hz: float) -> None:
    assert _frequency_under_a_body(_single(2.0), body_hz) == pytest.approx(
        body_hz, abs=0.01
    )


def test_without_a_gain_the_body_has_no_say() -> None:
    assert _frequency_under_a_body(_single(0.0), 0.8) == pytest.approx(
        1.0, abs=0.01
    )


def test_a_negative_amplitude_flips_the_swing() -> None:
    cpg = _single(1.0)
    cpg.set_params_by_group("amplitudes", [-0.7])
    assert cpg.forward(0.0).item() == pytest.approx(-0.7)


def test_a_coupling_pulls_the_phases_apart() -> None:
    cpg = TegotaeCPG({0: [1], 1: [0]}, coupling=2.0, angle_limits=None)
    cpg.set_param_with_dict({
        "phase": [0.0, 0.0],
        "w": [OMEGA, OMEGA],
        "amplitudes": [1.0, 1.0],
        "offset": [0.0, 0.0],
        "sigma": [0.0, 0.0],
    })
    cpg.reset_state()
    cpg.set_params_by_group("phase", [0.0, 1.0])
    for _ in range(500):
        cpg.forward()
    difference = (cpg.theta[1] - cpg.theta[0]).item() % (2 * math.pi)
    assert difference == pytest.approx(1.0, abs=0.01)


def test_the_networks_hold_the_expected_number_of_weights() -> None:
    cpg = LearnableTegotaeCPG({0: [1], 1: [0]}, hidden=HIDDEN)
    assert sum(cpg.shared_group_sizes().values()) == NETWORK_WEIGHTS
    assert cpg.num_of_parameters == 4 * 2 + NETWORK_WEIGHTS
    lower, upper = cpg.parameter_bounds()
    assert lower.numel() == cpg.num_of_parameters
    assert upper[-NETWORK_WEIGHTS:].tolist() == [1.0] * NETWORK_WEIGHTS


def test_mlp_is_one_tanh_layer_and_a_linear_output() -> None:
    # 2 inputs, 3 hidden units: 6 input weights, 3 biases, 3 output
    # weights and 1 output bias
    first = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    bias = torch.tensor([0.0, -0.1, 0.2])
    second = torch.tensor([1.0, -2.0, 0.5])
    weights = torch.cat([first.flatten(), bias, second, torch.tensor([0.3])])
    inputs = torch.tensor([[1.0, -1.0], [0.5, 0.0]])
    expected = torch.tanh(inputs @ first + bias) @ second + 0.3
    assert mlp(weights, inputs, 3).tolist() == pytest.approx(expected.tolist())


def test_no_load_means_no_learned_feedback() -> None:
    cpg = LearnableTegotaeCPG({0: [], 1: [], 2: []}, seed=0, angle_limits=None)
    cpg.theta = torch.tensor([0.1, 2.0, 4.0])
    cpg.set_input([0.0, 0.0, 0.0])
    assert cpg.feedback().abs().max().item() == pytest.approx(0.0)
    cpg.set_input([LOAD, LOAD, LOAD])
    assert cpg.feedback().abs().max().item() > 0
