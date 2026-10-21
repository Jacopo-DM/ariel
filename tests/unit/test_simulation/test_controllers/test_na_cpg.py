"""Test: NaCPG, the normalized asymmetric CPG of Tong et al. (2022)."""

import math

import numpy as np
import pytest
import torch

from ariel.simulation import NaCPG
from ariel.simulation.controllers.cpg import na_cpg

OMEGA = 2 * math.pi
STEPS = 400
SETTLED = 100  # steps before the output is measured
AMPLITUDE_TOLERANCE = 0.06  # the paper reports 5.3 % at alpha = 30
CHAIN_LENGTH = 4
HA = 0.4
HALF = 0.5
# Share of a period in the half with zeta = 1 - HA, out of 2 halves
SHORT_HALF_SHARE = (1 - HA) / 2


def _chain(n: int) -> dict[int, list[int]]:
    return {i: [i - 1] if i > 0 else [] for i in range(n)}


def _unit_cpg(n: int = 1, ha: float = 0.0) -> NaCPG:
    """Build a chain with w = 2 pi, A = 1, no offset and no angle limits.

    Returns
    -------
    NaCPG
        The network, reset to its start state.
    """
    cpg = NaCPG(_chain(n), seed=1, angle_limits=None, angle_tracking=True)
    cpg.set_param_with_dict({
        "ha": [ha] * n,
        "w": [OMEGA] * n,
        "amplitudes": [1.0] * n,
        "b": [0.0] * n,
        "phase": [0.5 * i for i in range(n)],
    })
    cpg.reset_state()
    return cpg


def _history(cpg: NaCPG, steps: int = STEPS) -> torch.Tensor:
    for _ in range(steps):
        cpg.forward()
    return torch.tensor(cpg.angle_history)


def test_the_groups_are_the_papers() -> None:
    cpg = NaCPG(_chain(2))
    assert set(cpg.parameter_groups) == {"phase", "w", "amplitudes", "ha", "b"}
    assert (cpg.ha.abs() < 1).all()


def test_the_amplitude_only_scales_the_output() -> None:
    small = _unit_cpg()
    large = _unit_cpg()
    large.set_params_by_group("amplitudes", [2.5])
    ratio = _history(large)[SETTLED:] / _history(small)[SETTLED:]
    assert ratio.mean().item() == pytest.approx(2.5, rel=1e-3)


def test_the_offset_shifts_the_cycle() -> None:
    cpg = _unit_cpg()
    cpg.set_params_by_group("b", [0.5])
    cpg.reset_state()
    history = _history(cpg)[SETTLED:, 0]
    assert history.mean().item() == pytest.approx(HALF, abs=0.02)
    assert (history.max() - history.min()).item() == pytest.approx(
        2.0, abs=2 * AMPLITUDE_TOLERANCE
    )


def test_the_asymmetry_parameter_changes_the_time_in_each_half() -> None:
    symmetric = _history(_unit_cpg())[SETTLED:, 0]
    skewed = _history(_unit_cpg(ha=HA))[SETTLED:, 0]
    assert (symmetric < 0).float().mean().item() == pytest.approx(
        HALF, abs=0.02
    )
    assert (skewed < 0).float().mean().item() == pytest.approx(
        SHORT_HALF_SHARE, abs=0.04
    )
    # the shape of the cycle is unchanged
    assert skewed.max().item() == pytest.approx(
        symmetric.max().item(), abs=0.02
    )


def test_coupled_oscillators_keep_the_set_amplitude() -> None:
    history = _history(_unit_cpg(CHAIN_LENGTH))[SETTLED:]
    assert history.max(dim=0).values.tolist() == pytest.approx(
        [1.0] * CHAIN_LENGTH, abs=AMPLITUDE_TOLERANCE
    )


def test_a_sudden_parameter_change_stays_bounded() -> None:
    cpg = _unit_cpg(CHAIN_LENGTH)
    for _ in range(SETTLED):
        cpg.forward()
    cpg.set_param_with_dict({
        "w": [3 * OMEGA] * CHAIN_LENGTH,
        "ha": [HA] * CHAIN_LENGTH,
    })
    history = _history(cpg)
    assert not torch.isnan(history).any()
    assert history.abs().max().item() < 1 + 2 * AMPLITUDE_TOLERANCE


def test_the_constraint_bounds_the_change_of_the_derivative() -> None:
    cpg = NaCPG({0: []}, alpha=1000.0, angle_limits=None)
    cpg.set_state([5.0, 5.0], [0.0, 0.0])
    cpg.forward()
    limit = cpg.constraint_function(cpg.dt)[0].item()
    assert cpg.xy_dot_old.abs().max().item() <= limit + 1e-6


def test_zeta_follows_the_sign_of_the_old_derivative() -> None:
    cpg = NaCPG({0: [], 1: [], 2: []})
    cpg.set_params_by_group("ha", [HA, HA, HA])
    cpg.set_state([0, 0, 0, 0, 0, 0], [1.0, 0, -1.0, 0, 0, 0])
    assert cpg.zeta().tolist() == pytest.approx([1 - HA, 1 + HA, 1.0])


def test_zeta_never_reaches_zero() -> None:
    cpg = NaCPG({0: []})
    cpg.set_params_by_group("ha", [1.0])
    cpg.set_state([0, 0], [1.0, 0.0])
    assert cpg.zeta().item() > 0


def test_the_default_constraint_grows_with_w_squared_and_the_step() -> None:
    cpg = NaCPG({0: [], 1: []})
    cpg.set_params_by_group("w", [1.0, 2.0])
    low, high = cpg.constraint_function(0.01).tolist()
    assert high == pytest.approx(4 * low)
    assert low == pytest.approx(na_cpg.CONSTRAINT_MARGIN * 0.01)
    assert cpg.constraint_function(0.02)[0].item() == pytest.approx(2 * low)


def test_a_given_polynomial_is_the_constraint() -> None:
    cpg = NaCPG({0: [], 1: []})
    cpg.coefficients = (0.0, 0.0, 2.0, 1.0)
    cpg.set_params_by_group("w", [1.0, -5.0])
    # 2 w + 1 at w = 1 is 3; at w = -5 it is -9, which is cut to 0
    assert cpg.constraint_function(0.01).tolist() == [3.0, 0.0]


def test_set_state_accepts_every_input_type() -> None:
    cpg = NaCPG({0: [], 1: []})
    cpg.set_state([1.0, 2.0, 3.0, 4.0], np.zeros(4))
    assert cpg.xy.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    cpg.set_state(torch.ones(2, 2), torch.zeros(2, 2))
    assert cpg.xy_dot_old.tolist() == [[0.0, 0.0], [0.0, 0.0]]


def test_reset_state_starts_on_the_limit_cycle_at_the_phase() -> None:
    cpg = NaCPG({0: []})
    cpg.set_params_by_group("phase", [math.pi / 2])
    cpg.set_params_by_group("b", [0.25])
    cpg.forward()
    cpg.reset_state()
    assert cpg.xy[0, 0].item() == pytest.approx(0.0, abs=1e-6)
    assert cpg.xy[0, 1].item() == pytest.approx(1.25)
    assert cpg.xy_dot_old.abs().sum().item() == pytest.approx(0.0)
