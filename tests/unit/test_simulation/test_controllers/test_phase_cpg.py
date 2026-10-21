"""Test: PhaseCPG, amplitude-controlled phase oscillators (Ijspeert, 2007)."""

import math

import numpy as np
import pytest

from ariel.simulation import PhaseCPG

OMEGA = 2 * math.pi
ROUNDING = 1e-6
# One step of a 1 Hz unit cycle moves at most 2 pi 0.01 = 0.063
LARGEST_STEP = 0.07


def _single() -> PhaseCPG:
    cpg = PhaseCPG({0: []}, angle_limits=None, angle_tracking=True)
    cpg.set_param_with_dict({
        "phase": [0.0],
        "w": [OMEGA],
        "amplitudes": [1.0],
        "offset": [0.2],
    })
    cpg.reset_state()
    return cpg


def test_the_amplitude_starts_at_zero_and_glides_to_its_target() -> None:
    cpg = _single()
    assert cpg.forward(0.0).item() == pytest.approx(0.2)
    cpg.forward(0.5)
    # critically damped: r = R (1 - exp(-a t / 2) (1 + a t / 2)), a = 20
    assert cpg.r.item() == pytest.approx(1 - 6 * math.exp(-5), abs=0.005)
    cpg.forward(1.0)
    assert cpg.r.item() == pytest.approx(1.0, abs=0.002)


def test_the_amplitude_never_overshoots() -> None:
    cpg = _single()
    radii = []
    for step in range(1, 101):
        cpg.forward(step * 0.01)
        radii.append(cpg.r.item())
    assert max(radii) <= 1.0 + 1e-3
    assert np.all(np.diff(radii) >= -ROUNDING)


def test_a_new_amplitude_mid_run_causes_no_jump() -> None:
    cpg = _single()
    for step in range(101):
        cpg.forward(step * 0.01)
    cpg.set_params_by_group("amplitudes", [0.2])
    for step in range(101, 301):
        cpg.forward(step * 0.01)
    history = np.array(cpg.angle_history)[:, 0]
    assert np.abs(np.diff(history)).max() < LARGEST_STEP
    assert cpg.r.item() == pytest.approx(0.2, abs=0.01)


def test_the_phase_starts_at_its_target() -> None:
    cpg = PhaseCPG({0: [], 1: []})
    cpg.set_params_by_group("phase", [0.3, 1.2])
    cpg.reset_state()
    assert cpg.theta.tolist() == pytest.approx([0.3, 1.2])
