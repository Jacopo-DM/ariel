"""Test: NaCPG, the normalized asymmetric CPG of Tong et al. (2022)."""

import inspect
import math
import runpy
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from ariel.simulation.controllers import NaCPG, na_cpg
from ariel.simulation.controllers.na_cpg import (
    BetaParams,
    create_fully_connected_adjacency,
    scaled_beta,
)

OMEGA = 2 * math.pi
STEPS = 400
SETTLED = 100  # steps before the output is measured
AMPLITUDE_TOLERANCE = 0.06  # the paper reports 5.3 % at alpha = 30
CHAIN_LENGTH = 4
NUM_NODES = 3
HA = 0.4
TRACKED_STEPS = 7
STEPS_AFTER_RESET = 0.5
FLAT_NODES = 5
HALF = 0.5
NUM_GROUPS = 5
# Share of a period in the half with zeta = 1 - HA, out of 2 halves
SHORT_HALF_SHARE = (1 - HA) / 2


def _chain(n: int) -> dict[int, list[int]]:
    return {i: [i - 1] if i > 0 else [] for i in range(n)}


def _unit_cpg(
    n: int = 1,
    ha: float = 0.0,
    phases: list[float] | None = None,
    *,
    track: bool = True,
) -> NaCPG:
    """Build a chain with w = 2 pi, A = 1, no offset and no angle limits.

    Returns
    -------
    NaCPG
        The network, reset to its start state.
    """
    cpg = NaCPG(_chain(n), seed=1, angle_limits=None, angle_tracking=track)
    cpg.set_param_with_dict({
        "ha": [ha] * n,
        "w": [OMEGA] * n,
        "amplitudes": [1.0] * n,
        "b": [0.0] * n,
        "phase": phases if phases is not None else [0.5 * i for i in range(n)],
    })
    cpg.reset_state()
    return cpg


def _history(cpg: NaCPG, steps: int = STEPS) -> torch.Tensor:
    for _ in range(steps):
        cpg.forward()
    return torch.tensor(cpg.angle_history)


# ---------------------------------------------------------------------------
# Construction and parameters
# ---------------------------------------------------------------------------


def test_the_package_exports_the_class() -> None:
    assert na_cpg.NaCPG is NaCPG


def test_parameter_count_and_groups() -> None:
    cpg = NaCPG(create_fully_connected_adjacency(CHAIN_LENGTH))
    assert cpg.n == CHAIN_LENGTH
    assert cpg.num_of_parameters == NUM_GROUPS * CHAIN_LENGTH
    assert cpg.num_of_parameter_groups == NUM_GROUPS
    assert set(cpg.parameter_groups) == {"phase", "w", "amplitudes", "ha", "b"}


def test_the_same_seed_gives_the_same_parameters() -> None:
    first = NaCPG(_chain(NUM_NODES), seed=3).get_flat_params()
    second = NaCPG(_chain(NUM_NODES), seed=3).get_flat_params()
    other = NaCPG(_chain(NUM_NODES), seed=4).get_flat_params()
    assert torch.equal(first, second)
    assert not torch.equal(first, other)


def test_without_a_seed_the_network_still_builds() -> None:
    assert NaCPG(_chain(NUM_NODES)).n == NUM_NODES


def test_keys_must_be_the_node_numbers() -> None:
    with pytest.raises(ValueError, match=r"must be 0 \.\. n - 1"):
        NaCPG({0: [], 2: []})


def test_neighbours_must_be_nodes() -> None:
    with pytest.raises(ValueError, match="unknown neighbour 5"):
        NaCPG({0: [5], 1: []})


def test_scaled_beta_stays_inside_its_bounds() -> None:
    params = BetaParams(a=2, b=3, c=1, d=-1)
    sample = scaled_beta(params, 200, np.random.default_rng(0))
    assert sample.shape == (200,)
    assert sample.min() >= -1
    assert sample.max() <= 1


def test_initial_parameters_are_inside_their_ranges() -> None:
    cpg = NaCPG(create_fully_connected_adjacency(20), seed=0)
    assert (cpg.phase >= 0).all()
    assert (cpg.phase <= 2 * math.pi).all()
    assert (cpg.w >= 0).all()
    assert (cpg.amplitudes <= math.pi / 2).all()
    assert (cpg.ha.abs() < 1).all()
    assert (cpg.b.abs() <= HALF).all()


def test_flat_params_roundtrip_from_every_input_type() -> None:
    cpg = NaCPG(_chain(NUM_NODES), seed=0)
    size = cpg.num_of_parameters
    for params in (
        torch.arange(size, dtype=torch.float32),
        list(map(float, range(size))),
        np.arange(size, dtype=np.float64),
    ):
        cpg.set_flat_params(params)
        assert torch.equal(
            cpg.get_flat_params(), torch.arange(size, dtype=torch.float32)
        )


def test_flat_params_of_the_wrong_size_are_refused() -> None:
    cpg = NaCPG(_chain(NUM_NODES))
    with pytest.raises(ValueError, match="incorrect size"):
        cpg.set_flat_params(torch.zeros(1))


def test_group_setters() -> None:
    cpg = NaCPG(_chain(NUM_NODES))
    cpg.set_params_by_group("w", [1.0, 2.0, 3.0])
    assert cpg.w.tolist() == [1.0, 2.0, 3.0]
    cpg.set_param_with_dict({"b": np.zeros(NUM_NODES)})
    assert cpg.b.tolist() == [0.0] * NUM_NODES
    with pytest.raises(ValueError, match="does not exist"):
        cpg.set_params_by_group("nope", [0.0] * NUM_NODES)
    with pytest.raises(ValueError, match="incorrect size for group 'w'"):
        cpg.set_params_by_group("w", [0.0])


# ---------------------------------------------------------------------------
# The paper's properties
# ---------------------------------------------------------------------------


def test_a_single_oscillator_has_the_set_amplitude_and_period() -> None:
    history = _history(_unit_cpg())[SETTLED:, 0]
    assert history.max().item() == pytest.approx(1.0, abs=AMPLITUDE_TOLERANCE)
    assert history.min().item() == pytest.approx(-1.0, abs=AMPLITUDE_TOLERANCE)
    crossings = ((history[:-1] < 0) & (history[1:] >= 0)).sum().item()
    seconds = len(history) * 0.01
    assert crossings == round(seconds * OMEGA / (2 * math.pi))


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


def test_oscillators_lock_to_the_phase_differences() -> None:
    delta = 0.8
    cpg = NaCPG({0: [1], 1: []}, seed=0, angle_limits=None)
    cpg.set_param_with_dict({
        "ha": [0.0, 0.0],
        "w": [OMEGA, OMEGA],
        "amplitudes": [1.0, 1.0],
        "b": [0.0, 0.0],
        "phase": [delta, 0.0],
    })
    cpg.reset_state()
    for _ in range(STEPS):
        cpg.forward()
    angle_0 = math.atan2(cpg.xy[0, 1].item(), cpg.xy[0, 0].item())
    angle_1 = math.atan2(cpg.xy[1, 1].item(), cpg.xy[1, 0].item())
    assert (angle_0 - angle_1) % (2 * math.pi) == pytest.approx(delta, abs=0.05)


def test_a_sudden_parameter_change_stays_bounded() -> None:
    cpg = _unit_cpg(CHAIN_LENGTH, ha=0.0)
    for _ in range(SETTLED):
        cpg.forward()
    cpg.set_param_with_dict({
        "w": [3 * OMEGA] * CHAIN_LENGTH,
        "ha": [HA] * CHAIN_LENGTH,
    })
    for _ in range(STEPS):
        cpg.forward()
    history = torch.tensor(cpg.angle_history)
    assert not torch.isnan(history).any()
    assert history.abs().max().item() < 1 + 2 * AMPLITUDE_TOLERANCE


def test_the_constraint_bounds_the_change_of_the_derivative() -> None:
    cpg = NaCPG({0: []}, alpha=1000.0, angle_limits=None)
    cpg.set_state([5.0, 5.0], [0.0, 0.0])
    cpg.forward()
    limit = cpg.constraint_function()[0].item()
    assert cpg.xy_dot_old.abs().max().item() <= limit + 1e-6


# ---------------------------------------------------------------------------
# zeta, constraint function, state
# ---------------------------------------------------------------------------


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


def test_the_default_constraint_grows_with_the_square_of_the_frequency() -> (
    None
):
    cpg = NaCPG({0: [], 1: []}, dt=0.01)
    cpg.set_params_by_group("w", [1.0, 2.0])
    low, high = cpg.constraint_function().tolist()
    assert high == pytest.approx(4 * low)
    assert low == pytest.approx(na_cpg.CONSTRAINT_MARGIN * 0.01)


def test_a_given_polynomial_is_the_constraint() -> None:
    cpg = NaCPG({0: [], 1: []})
    cpg.coefficients = (0.0, 0.0, 2.0, 1.0)
    cpg.set_params_by_group("w", [1.0, -5.0])
    # 2 w + 1 at w = 1 is 3; at w = -5 it is -9, which is cut to 0
    assert cpg.constraint_function().tolist() == [3.0, 0.0]


def test_set_state_accepts_every_input_type() -> None:
    cpg = NaCPG({0: [], 1: []})
    cpg.set_state([1.0, 2.0, 3.0, 4.0], np.zeros(4))
    assert cpg.xy.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    cpg.set_state(torch.ones(2, 2), torch.zeros(2, 2))
    assert cpg.xy_dot_old.tolist() == [[0.0, 0.0], [0.0, 0.0]]


def test_reset_state_starts_on_the_limit_cycle_at_the_phase() -> None:
    cpg = NaCPG({0: []}, angle_tracking=True)
    cpg.set_params_by_group("phase", [math.pi / 2])
    cpg.set_params_by_group("b", [0.25])
    cpg.forward()
    cpg.reset_state()
    assert cpg.xy[0, 0].item() == pytest.approx(0.0, abs=1e-6)
    assert cpg.xy[0, 1].item() == pytest.approx(1.25)
    assert cpg.xy_dot_old.abs().sum().item() == pytest.approx(0.0)
    assert cpg.angle_history == []


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------


def test_forward_returns_one_angle_per_oscillator() -> None:
    cpg = NaCPG(create_fully_connected_adjacency(FLAT_NODES), seed=0)
    assert cpg.forward().shape == (FLAT_NODES,)


def test_forward_at_time_zero_resets_the_state() -> None:
    cpg = NaCPG(_chain(NUM_NODES), seed=0, angle_tracking=True)
    for _ in range(5):
        cpg.forward()
    cpg.forward(0.0)
    assert len(cpg.angle_history) == 1
    cpg.forward(STEPS_AFTER_RESET)
    assert len(cpg.angle_history) == 1 + 1


def test_clipping_limits_the_angles_and_records_the_overshoot() -> None:
    cpg = NaCPG({0: []}, angle_limits=(-0.5, 0.5))
    cpg.set_params_by_group("amplitudes", [2.0])
    cpg.set_state([0.0, 1.0], [0.0, 0.0])
    angle = cpg.forward()
    assert angle.item() == pytest.approx(0.5)
    assert cpg.clamping_error == pytest.approx(2.0 * cpg.xy[0, 1].item() - 0.5)


def test_without_clipping_the_angles_pass_but_the_error_is_recorded() -> None:
    cpg = NaCPG({0: []}, angle_limits=(-0.5, 0.5), clipping=False)
    cpg.set_params_by_group("amplitudes", [2.0])
    cpg.set_state([0.0, 1.0], [0.0, 0.0])
    angle = cpg.forward()
    assert angle.item() == pytest.approx(2.0 * cpg.xy[0, 1].item())
    assert cpg.clamping_error == pytest.approx(angle.item() - 0.5)


def test_no_limits_means_no_clamping_error() -> None:
    cpg = NaCPG({0: []}, angle_limits=None)
    cpg.set_params_by_group("amplitudes", [5.0])
    for _ in range(50):
        cpg.forward()
    assert cpg.clamping_error == pytest.approx(0.0)


def test_nan_in_the_parameters_is_an_error() -> None:
    cpg = NaCPG({0: []})
    cpg.set_params_by_group("w", [float("nan")])
    with pytest.raises(ValueError, match="NaN values detected"):
        cpg.forward()


def test_angle_tracking_records_every_step() -> None:
    cpg = NaCPG(_chain(NUM_NODES), angle_tracking=True)
    for _ in range(TRACKED_STEPS):
        cpg.forward()
    assert len(cpg.angle_history) == TRACKED_STEPS
    assert len(cpg.angle_history[0]) == NUM_NODES


# ---------------------------------------------------------------------------
# save, load, demo
# ---------------------------------------------------------------------------


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    first = NaCPG(_chain(NUM_NODES), seed=0)
    path = tmp_path / "params.pt"
    first.save(path)
    second = NaCPG(_chain(NUM_NODES), seed=1)
    second.load(path)
    assert torch.equal(first.get_flat_params(), second.get_flat_params())


def test_demo_plots_and_saves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mpl.use("Agg")
    monkeypatch.setattr(na_cpg, "DATA", tmp_path / "__data__")
    monkeypatch.setattr(plt, "show", lambda: None)
    na_cpg.main()
    assert (tmp_path / "__data__" / "na_cpg_angles.png").exists()
    assert (tmp_path / "__data__" / "na_cpg_params.pt").exists()


def test_demo_runs_as_a_script(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mpl.use("Agg")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt, "show", lambda: None)
    runpy.run_path(inspect.getfile(na_cpg), run_name="__main__")
    assert (tmp_path / "__data__" / "na_cpg_angles.png").exists()
