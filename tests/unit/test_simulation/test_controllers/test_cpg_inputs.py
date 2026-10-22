"""Test: the input vector every CPG takes through ``set_input``."""

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from ariel.simulation import (
    CPG,
    AdaptiveHopfCPG,
    HopfCPG,
    InputRoute,
    LearnableTegotaeCPG,
    NaCPG,
    PhaseCPG,
    SwingStanceCPG,
    TegotaeCPG,
    create_fully_connected_adjacency,
)

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
WITHOUT_FAST_HANDLES = [HopfCPG, PhaseCPG, SwingStanceCPG, NaCPG]
OMEGA = 2 * math.pi
SETTLE_SECONDS = 3.0  # the filter is within 1e-9 of its target after 3 s
FILTER_SECONDS = 0.5  # 96 % of a change is done after 5 / k
SHIFT = 0.2
ONE_CYCLE = 100  # steps of 0.01 s at 1 Hz
SPEED_UP = 1.5
LOAD = 5.0
FIRST_STEP_LIMIT = 0.02  # a jump of 1 moves the handle 0.01 in 0.01 s

type Model = type[CPG]


def _unit(model: Model, routes: Sequence[InputRoute], n: int = 2) -> CPG:
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
        inputs=routes,
    )
    values: dict[str, ParamInput] = {
        "phase": [0.5 * i for i in range(n)],
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


def _run(cpg: CPG, seconds: float, u: list[float]) -> NDArray[np.float64]:
    """Feed a constant input from time 0 and record the angles.

    Returns
    -------
    NDArray[np.float64]
        One row of angles per step of ``dt``.
    """
    cpg.forward(0.0)
    cpg.set_input(u)
    rows = [
        cpg.forward(step * cpg.dt).tolist()
        for step in range(1, round(seconds / cpg.dt) + 1)
    ]
    return np.array(rows)


def _offset_group(cpg: CPG) -> str:
    return "b" if isinstance(cpg, NaCPG) else "offset"


@pytest.mark.parametrize("model", WITHOUT_FAST_HANDLES)
def test_by_default_a_model_without_fast_handles_takes_no_input(
    model: Model,
) -> None:
    cpg = model({0: [], 1: []})
    assert cpg.routes == ()
    assert "input_weights" not in cpg.parameter_groups
    cpg.set_input([])
    assert cpg.handle("offset").tolist() == [0.0, 0.0]


@pytest.mark.parametrize(
    ("model", "handle"),
    [
        (AdaptiveHopfCPG, "teaching"),
        (TegotaeCPG, "load"),
        (LearnableTegotaeCPG, "load"),
    ],
)
def test_by_default_entry_i_goes_to_the_fast_handle_of_joint_i(
    model: Model, handle: str
) -> None:
    cpg = model({0: [], 1: []})
    assert cpg.routes == (
        InputRoute(handle=handle, joint=0),
        InputRoute(handle=handle, joint=1),
    )
    cpg.set_input([2.0, 3.0])
    assert cpg.handle(handle).tolist() == [2.0, 3.0]


@pytest.mark.parametrize("model", ALL)
def test_the_amplitude_input_scales_the_amplitude(model: Model) -> None:
    steered = _unit(model, [InputRoute(handle="amplitude")])
    steered.set_params_by_group("input_weights", [-0.5, -0.5])
    reference = _unit(model, [])
    reference.set_params_by_group("amplitudes", [0.5, 0.5])
    expected = _run(reference, SETTLE_SECONDS, [])[-1]
    assert _run(steered, SETTLE_SECONDS, [1.0])[-1] == pytest.approx(
        expected, abs=1e-3
    )


@pytest.mark.parametrize("model", ALL)
def test_the_offset_input_shifts_the_offset(model: Model) -> None:
    steered = _unit(model, [InputRoute(handle="offset")])
    steered.set_params_by_group("input_weights", [SHIFT, SHIFT])
    reference = _unit(model, [])
    reference.set_params_by_group(_offset_group(reference), [SHIFT, SHIFT])
    reference.reset_state()
    # NaCPG moves the centre of its cycle, which also moves the phase a
    # little, so compare the last cycle (1 s) by its mean and its range
    expected = _run(reference, SETTLE_SECONDS, [])[-ONE_CYCLE:]
    angles = _run(steered, SETTLE_SECONDS, [1.0])[-ONE_CYCLE:]
    assert angles.mean(axis=0) == pytest.approx(expected.mean(axis=0), abs=1e-3)
    assert np.ptp(angles, axis=0) == pytest.approx(
        np.ptp(expected, axis=0), abs=1e-3
    )


@pytest.mark.parametrize("model", ALL)
def test_the_frequency_input_scales_the_frequency(model: Model) -> None:
    cpg = _unit(model, [InputRoute(handle="frequency")], n=1)
    cpg.set_params_by_group("input_weights", [SPEED_UP - 1])
    angles = _run(cpg, 6.0, [1.0])[:, 0]
    rising = np.flatnonzero((angles[:-1] < 0) & (angles[1:] >= 0)) * cpg.dt
    late = rising[rising > SETTLE_SECONDS]
    assert np.diff(late).mean() == pytest.approx(1 / SPEED_UP, abs=0.02)


def test_a_jump_of_the_input_moves_the_slow_handles_smoothly() -> None:
    cpg = _unit(HopfCPG, [InputRoute(handle="offset")])
    cpg.set_params_by_group("input_weights", [1.0, 1.0])
    cpg.set_input([1.0])
    cpg.forward()
    assert cpg.handle("offset")[0].item() < FIRST_STEP_LIMIT
    seen = [cpg.handle("offset")[0].item()]
    for _ in range(round(FILTER_SECONDS / cpg.dt) - 1):
        cpg.forward()
        seen.append(cpg.handle("offset")[0].item())
    assert seen[-1] == pytest.approx(0.96, abs=0.02)
    assert np.all(np.diff(seen) > 0)


def test_a_route_with_a_joint_reaches_that_joint_alone() -> None:
    cpg = _unit(PhaseCPG, [InputRoute(handle="offset", joint=1)])
    assert "input_weights" not in cpg.parameter_groups
    _run(cpg, SETTLE_SECONDS, [SHIFT])
    assert cpg.handle("offset").tolist() == pytest.approx([0.0, SHIFT])


def test_every_joint_has_its_own_weight() -> None:
    routes = [InputRoute(handle="offset"), InputRoute(handle="amplitude")]
    cpg = _unit(HopfCPG, routes)
    assert cpg.group_sizes["input_weights"] == 2 * 2
    assert list(cpg.parameter_groups)[-1] == "input_weights"
    lower, upper = cpg.parameter_bounds()
    assert lower[-4:].tolist() == [-1.0] * 4
    assert upper[-4:].tolist() == [1.0] * 4
    # rows follow the routes: offset weights first, then amplitude weights
    cpg.set_params_by_group("input_weights", [SHIFT, -SHIFT, 0.5, 0.0])
    _run(cpg, SETTLE_SECONDS, [1.0, -1.0])
    assert cpg.handle("offset").tolist() == pytest.approx([SHIFT, -SHIFT])
    assert cpg.amplitude_factor().tolist() == pytest.approx([0.5, 1.0])


def test_the_factors_never_go_below_zero() -> None:
    routes = [InputRoute(handle="amplitude"), InputRoute(handle="frequency")]
    cpg = _unit(PhaseCPG, routes)
    cpg.set_params_by_group("input_weights", [1.0] * 4)
    _run(cpg, SETTLE_SECONDS, [-3.0, -3.0])
    assert cpg.amplitude_factor().tolist() == [0.0, 0.0]
    assert cpg.frequency_factor().tolist() == [0.0, 0.0]


def test_a_weighted_route_can_feed_a_fast_handle() -> None:
    cpg = TegotaeCPG({0: [], 1: []}, inputs=[InputRoute(handle="load")])
    cpg.set_param_with_dict({
        "sigma": [1.0, 1.0],
        "input_weights": [0.5, 2.0],
    })
    cpg.theta = torch.tensor([0.0, 0.0])
    cpg.set_input([LOAD])
    assert cpg.feedback().tolist() == pytest.approx([-0.5 * LOAD, -2 * LOAD])


def test_new_weights_apply_from_the_next_forward() -> None:
    cpg = TegotaeCPG({0: []}, inputs=[InputRoute(handle="load")])
    cpg.set_params_by_group("input_weights", [1.0])
    cpg.set_input([LOAD])
    cpg.set_params_by_group("input_weights", [2.0])
    cpg.forward()
    assert cpg.handle("load").tolist() == [2 * LOAD]


def test_a_reset_clears_the_input() -> None:
    cpg = _unit(HopfCPG, [InputRoute(handle="offset")])
    cpg.set_params_by_group("input_weights", [1.0, 1.0])
    _run(cpg, 1.0, [1.0])
    cpg.reset_state()
    assert cpg.inputs.tolist() == [0.0]
    assert cpg.handle("offset").tolist() == [0.0, 0.0]


def test_an_unknown_handle_is_refused() -> None:
    with pytest.raises(ValueError, match="Unknown handle 'load'"):
        HopfCPG({0: []}, inputs=[InputRoute(handle="load")])


def test_an_unknown_joint_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown joint 2"):
        HopfCPG({0: [], 1: []}, inputs=[InputRoute(handle="offset", joint=2)])


def test_an_input_of_the_wrong_size_is_refused() -> None:
    cpg = TegotaeCPG({0: [], 1: []})
    with pytest.raises(ValueError, match="Expected 2 inputs, got 3"):
        cpg.set_input([1.0, 2.0, 3.0])
