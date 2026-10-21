"""Test: the behaviour every CPG in ariel.simulation.controllers.cpg shares."""

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from ariel.simulation import (
    CPG,
    AdaptiveHopfCPG,
    HopfCPG,
    LearnableTegotaeCPG,
    NaCPG,
    PhaseCPG,
    SwingStanceCPG,
    TegotaeCPG,
    create_fully_connected_adjacency,
)
from ariel.simulation.controllers.cpg import BetaParams, scaled_beta

if TYPE_CHECKING:
    from ariel.simulation.controllers.cpg._base import ParamInput

ALL = [
    HopfCPG,
    AdaptiveHopfCPG,
    PhaseCPG,
    SwingStanceCPG,
    NaCPG,
    TegotaeCPG,
    LearnableTegotaeCPG,
]
# Models whose coupling holds any phase difference (Tegotae has none)
ANY_PHASE = [HopfCPG, AdaptiveHopfCPG, PhaseCPG, NaCPG]
OMEGA = 2 * math.pi
NUM_NODES = 3
TRACKED_STEPS = 7
CONTROL_PERIOD = 0.125
SUB_STEPS = 13  # ceil(0.125 / 0.01)
AMPLITUDE_TOLERANCE = 0.06
NODES_IN_GROUP_TEST = 4
LIMIT = 0.5

type Model = type[CPG]


def _unit(model: Model, n: int = 1, phases: list[float] | None = None) -> CPG:
    """Build a network at 1 Hz, amplitude 1, no offset and no limits.

    Returns
    -------
    CPG
        The network, reset to its start state.
    """
    cpg = model(
        create_fully_connected_adjacency(n),
        seed=1,
        angle_limits=None,
        angle_tracking=True,
    )
    values: dict[str, ParamInput] = {
        "phase": phases if phases is not None else [0.5 * i for i in range(n)],
        "amplitudes": [1.0] * n,
    }
    for name in cpg.parameter_groups:
        if name.startswith("w"):
            values[name] = [OMEGA] * n
        elif name in {"offset", "b", "ha"}:
            values[name] = [0.0] * n
    cpg.set_param_with_dict(values)
    cpg.reset_state()
    return cpg


def _run(cpg: CPG, seconds: float) -> NDArray[np.float64]:
    for step in range(round(seconds / cpg.dt) + 1):
        cpg.forward(step * cpg.dt)
    return np.array(cpg.angle_history)


def _phase(signal: NDArray[np.float64]) -> float:
    """Return the phase of the 1 Hz component of a signal sampled at 100 Hz.

    Returns
    -------
    float
        The phase, in radians.
    """
    times = np.arange(len(signal)) * 0.01
    return float(np.angle(np.sum(signal * np.exp(-1j * OMEGA * times))))


# ---------------------------------------------------------------------------
# Construction and parameters
# ---------------------------------------------------------------------------


def test_the_base_class_cannot_be_built() -> None:
    with pytest.raises(TypeError, match="abstract"):
        CPG({0: []})  # type: ignore[abstract]  # ty: ignore[call-non-callable]  # the point of the test


@pytest.mark.parametrize("model", ALL)
def test_one_value_per_node_in_every_group(model: Model) -> None:
    cpg = model(create_fully_connected_adjacency(NODES_IN_GROUP_TEST))
    assert cpg.n == NODES_IN_GROUP_TEST
    assert cpg.num_of_parameter_groups == len(model.PARAMETER_RANGES)
    shared = cpg.shared_group_sizes()
    per_joint = cpg.num_of_parameter_groups - len(shared)
    assert cpg.num_of_parameters == (
        NODES_IN_GROUP_TEST * per_joint + sum(shared.values())
    )
    assert cpg.get_flat_params().numel() == cpg.num_of_parameters
    assert list(cpg.parameter_groups) == list(model.PARAMETER_RANGES)


@pytest.mark.parametrize("model", ALL)
def test_the_same_seed_gives_the_same_parameters(model: Model) -> None:
    adjacency = create_fully_connected_adjacency(NUM_NODES)
    first = model(adjacency, seed=3).get_flat_params()
    second = model(adjacency, seed=3).get_flat_params()
    other = model(adjacency, seed=4).get_flat_params()
    assert torch.equal(first, second)
    assert not torch.equal(first, other)


@pytest.mark.parametrize("model", ALL)
def test_initial_parameters_are_inside_their_bounds(model: Model) -> None:
    cpg = model(create_fully_connected_adjacency(20), seed=0)
    lower, upper = cpg.parameter_bounds()
    flat = cpg.get_flat_params()
    assert lower.shape == flat.shape == upper.shape
    assert (lower <= flat).all()
    assert (flat <= upper).all()


@pytest.mark.parametrize("model", ALL)
def test_the_adjacency_is_checked(model: Model) -> None:
    with pytest.raises(ValueError, match=r"must be 0 \.\. n - 1"):
        model({0: [], 2: []})
    with pytest.raises(ValueError, match="unknown neighbour 5"):
        model({0: [5], 1: []})


def test_scaled_beta_stays_inside_its_bounds() -> None:
    params = BetaParams(a=2, b=3, c=1, d=-1)
    sample = scaled_beta(params, 200, np.random.default_rng(0))
    assert sample.shape == (200,)
    assert sample.min() >= -1
    assert sample.max() <= 1


def test_the_full_adjacency_links_every_pair() -> None:
    assert create_fully_connected_adjacency(3) == {
        0: [1, 2],
        1: [0, 2],
        2: [0, 1],
    }


@pytest.mark.parametrize("model", ALL)
def test_flat_params_roundtrip_from_every_input_type(model: Model) -> None:
    cpg = model(create_fully_connected_adjacency(NUM_NODES), seed=0)
    size = cpg.num_of_parameters
    expected = torch.arange(size, dtype=torch.float32)
    for params in (
        torch.arange(size, dtype=torch.float32),
        list(map(float, range(size))),
        np.arange(size, dtype=np.float64),
    ):
        cpg.set_flat_params(params)
        assert torch.equal(cpg.get_flat_params(), expected)
    with pytest.raises(ValueError, match="incorrect size"):
        cpg.set_flat_params(torch.zeros(1))


@pytest.mark.parametrize("model", ALL)
def test_group_setters(model: Model) -> None:
    cpg = model(create_fully_connected_adjacency(NUM_NODES))
    cpg.set_params_by_group("amplitudes", [1.0, 2.0, 3.0])
    assert cpg.group("amplitudes").tolist() == [1.0, 2.0, 3.0]
    cpg.set_param_with_dict({"phase": np.zeros(NUM_NODES)})
    assert cpg.group("phase").tolist() == [0.0] * NUM_NODES
    with pytest.raises(ValueError, match="does not exist"):
        cpg.set_params_by_group("nope", [0.0] * NUM_NODES)
    with pytest.raises(ValueError, match="incorrect size for group 'phase'"):
        cpg.set_params_by_group("phase", [0.0])


@pytest.mark.parametrize("model", ALL)
def test_save_and_load_roundtrip(model: Model, tmp_path: Path) -> None:
    adjacency = create_fully_connected_adjacency(NUM_NODES)
    first = model(adjacency, seed=0)
    path = tmp_path / "params.pt"
    first.save(path)
    second = model(adjacency, seed=1)
    second.load(path)
    assert torch.equal(first.get_flat_params(), second.get_flat_params())


# ---------------------------------------------------------------------------
# The clock
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ALL)
def test_without_a_time_each_call_is_one_step(model: Model) -> None:
    cpg = model(
        create_fully_connected_adjacency(NUM_NODES), angle_tracking=True
    )
    for _ in range(TRACKED_STEPS):
        cpg.forward()
    assert len(cpg.angle_history) == TRACKED_STEPS
    assert len(cpg.angle_history[0]) == NUM_NODES


@pytest.mark.parametrize("model", ALL)
def test_the_first_timed_call_resets_and_does_not_move(model: Model) -> None:
    adjacency = create_fully_connected_adjacency(NUM_NODES)
    start = model(adjacency, seed=0).forward(2.0)
    cpg = model(adjacency, seed=0)
    for _ in range(5):
        cpg.forward()
    assert torch.equal(cpg.forward(2.0), start)
    assert cpg.last_time == pytest.approx(2.0)


@pytest.mark.parametrize("model", ALL)
def test_an_earlier_time_starts_a_new_episode(model: Model) -> None:
    cpg = model(create_fully_connected_adjacency(NUM_NODES), seed=0)
    start = cpg.forward(0.0)
    cpg.forward(1.0)
    assert torch.equal(cpg.forward(0.0), start)


@pytest.mark.parametrize("model", ALL)
def test_one_long_call_equals_its_sub_steps(model: Model) -> None:
    adjacency = create_fully_connected_adjacency(NUM_NODES)
    coarse = model(adjacency, seed=2)
    fine = model(adjacency, seed=2)
    coarse.forward(0.0)
    fine.forward(0.0)
    for call in range(1, 9):
        expected = coarse.forward(call * CONTROL_PERIOD)
        start = (call - 1) * CONTROL_PERIOD
        results = [
            fine.forward(start + sub * CONTROL_PERIOD / SUB_STEPS)
            for sub in range(1, SUB_STEPS + 1)
        ]
        assert torch.allclose(results[-1], expected, atol=1e-5)


@pytest.mark.parametrize("model", ALL)
def test_the_frequency_follows_the_simulation_time(model: Model) -> None:
    cpg = _unit(model)
    for call in range(81):
        cpg.forward(call * CONTROL_PERIOD)
    history = np.array(cpg.angle_history)[:, 0]
    crossings = ((history[:-1] < 0) & (history[1:] >= 0)).sum()
    assert crossings == pytest.approx(10, abs=1)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ALL)
def test_one_oscillator_has_the_set_amplitude_and_period(model: Model) -> None:
    history = _run(_unit(model), 4.0)[100:, 0]
    assert history.max() == pytest.approx(1.0, abs=AMPLITUDE_TOLERANCE)
    assert history.min() == pytest.approx(-1.0, abs=AMPLITUDE_TOLERANCE)
    crossings = ((history[:-1] < 0) & (history[1:] >= 0)).sum()
    assert crossings == pytest.approx(3, abs=1)


@pytest.mark.parametrize("model", ANY_PHASE)
def test_oscillators_lock_to_the_phase_differences(model: Model) -> None:
    targets = [0.0, 1.0, 2.5]
    cpg = _unit(model, NUM_NODES, phases=[0.0] * NUM_NODES)
    # The state starts in phase; only the coupling can pull it apart
    cpg.set_params_by_group("phase", targets)
    for _ in range(1000):
        cpg.forward()
    history = np.array(cpg.angle_history)[-200:]
    phases = [_phase(history[:, i]) for i in range(NUM_NODES)]
    for i in range(1, NUM_NODES):
        difference = (phases[i] - phases[0]) % (2 * math.pi)
        assert difference == pytest.approx(targets[i], abs=0.1)


@pytest.mark.parametrize("model", ALL)
def test_clipping_limits_the_angles_and_records_the_overshoot(
    model: Model,
) -> None:
    clipped = model({0: []}, angle_limits=(-LIMIT, LIMIT), seed=0)
    clipped.set_params_by_group("amplitudes", [5.0])
    free = model({0: []}, angle_limits=(-LIMIT, LIMIT), seed=0, clipping=False)
    free.set_params_by_group("amplitudes", [5.0])
    for _ in range(29):
        clipped.forward()
        free.forward()
    angle = clipped.forward()
    raw = free.forward()
    assert abs(angle.item()) <= LIMIT
    assert clipped.clamping_error == pytest.approx(
        max(abs(raw.item()) - LIMIT, 0.0)
    )
    assert free.clamping_error == pytest.approx(clipped.clamping_error)


@pytest.mark.parametrize("model", ALL)
def test_no_limits_means_no_clamping_error(model: Model) -> None:
    cpg = model({0: []}, angle_limits=None)
    cpg.set_params_by_group("amplitudes", [5.0])
    for _ in range(50):
        cpg.forward()
    assert cpg.clamping_error == pytest.approx(0.0)


@pytest.mark.parametrize("model", ALL)
def test_nan_in_the_parameters_is_an_error(model: Model) -> None:
    cpg = model({0: []})
    cpg.set_params_by_group("amplitudes", [float("nan")])
    with pytest.raises(ValueError, match="NaN values detected"):
        cpg.forward()


@pytest.mark.parametrize("model", ALL)
def test_reset_forgets_the_history(model: Model) -> None:
    cpg = model({0: []}, angle_tracking=True)
    cpg.forward(0.0)
    cpg.forward(0.5)
    cpg.reset_state()
    assert cpg.angle_history == []
    assert cpg.last_time is None
