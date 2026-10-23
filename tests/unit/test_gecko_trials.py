"""Test: the gecko trials the slow CPG notebooks send to their pool."""

import mujoco
import nevergrad as ng
import numpy as np
import pytest

from ariel.simulation import (
    AdaptiveHopfCPG,
    PhaseCPG,
    TegotaeCPG,
    process_pool,
)
from tests.functional.simulation import gecko_trials as g

# The PhaseCPG gait cpg_locomotion found and cpg_steering keeps; walking
# it must give the distance cpg_locomotion printed for it, to the bit
PHASE_GAIT = [
    2.751735261353428,
    6.215340343220078,
    5.556191924553975,
    1.8492441716077392,
    4.686047639390352,
    4.612207428077248,
    1.8464395860823404,
    2.5108100054803746,
    10.930232458056388,
    9.375174251009215,
    8.77832507931334,
    6.364745150089903,
    7.881603200481355,
    1.7940179703670567,
    4.978076593632627,
    1.061928070510141,
    0.07121294236914212,
    0.5183840467634308,
    1.0809107887235883,
    1.3017132840744965,
    0.8486451431459087,
    1.5385493503471535,
    0.6137673756885849,
    1.070874846863231,
    0.004863193570142421,
    -0.31032766430132586,
    -0.18225434886292702,
    0.375861175502577,
    0.00948019945742129,
    0.2640788059919923,
    0.13268655283122766,
    0.21906145125800958,
]
PHASE_DISTANCE = 2.439762899408016
SHORT = 0.5  # seconds: enough steps for every branch of a trial
PAIR = (0.0, np.pi / 2)
SETTLE = 2.0  # seconds for the gecko to come to rest
STEERING_WEIGHTS = len(g.STEERING) * 8  # five inputs to each of 8 joints


def test_each_process_builds_one_gecko_of_each_kind() -> None:
    """walker() and steerer() are built once; only steerer has a target."""
    assert g.walker() is g.walker()
    assert g.steerer() is g.steerer()
    assert g.walker().target_mocap == -1
    assert g.steerer().target_mocap >= 0
    assert g.walker().model.nu == len(g.STEERING) + 3


def test_walking_the_found_gait_gives_its_distance_in_a_worker() -> None:
    """The Walk job reproduces cpg_locomotion's number, in a pool too."""
    job = g.Walk(cpg="PhaseCPG", params=PHASE_GAIT, seed=0)
    assert g.walk(job) == PHASE_DISTANCE
    with process_pool(workers=1) as pool:
        assert list(pool.map(g.walk, [job])) == [PHASE_DISTANCE]


@pytest.mark.parametrize("cpg_class", [AdaptiveHopfCPG, TegotaeCPG])
def test_a_sensing_cpg_walks_and_after_step_sees_every_step(
    cpg_class: type[AdaptiveHopfCPG] | type[TegotaeCPG],
) -> None:
    """Joint error or limb loads feed in; the hook runs once per step."""
    gecko = g.walker()
    steps: list[int] = []

    def count(data: mujoco.MjData, step: int) -> None:
        assert data.time > 0
        steps.append(step)

    distance = gecko.walk(cpg_class(gecko.adjacency), SHORT, count)
    assert distance >= 0
    assert steps == list(range(1, len(steps) + 1))
    assert steps[-1] >= round(SHORT / gecko.model.opt.timestep)


def test_the_robot_on_the_floor_carries_its_weight() -> None:
    """At rest the floor forces add up to the gecko's weight."""
    gecko = g.walker()
    model, data = gecko.model, gecko.data
    mujoco.mj_resetData(model, data)
    while data.time < SETTLE:
        mujoco.mj_step(model, data)
    weight = model.body_mass[gecko.robot].sum() * -model.opt.gravity[2]
    assert gecko.floor_forces(data).sum() == pytest.approx(weight, rel=0.02)
    assert gecko.limb_loads(data).shape == (model.nu,)


def test_floor_forces_skip_contacts_inside_the_robot() -> None:
    """A contact between two robot bodies adds no floor force."""
    gecko = g.Gecko()
    model, data = gecko.model, gecko.data
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    robot_geoms = np.flatnonzero(gecko.robot[model.geom_bodyid])
    data.ncon = 1
    data.contact[0].geom1 = robot_geoms[0]
    data.contact[0].geom2 = robot_geoms[1]
    assert not gecko.floor_forces(data).any()


@pytest.mark.parametrize(
    ("cpg", "steered", "seconds"),
    [
        ("AdaptiveHopfCPG", True, g.SWITCH + SHORT),
        ("TegotaeCPG", False, SHORT),
        ("PhaseCPG", True, SHORT),
    ],
)
def test_a_steering_trial_records_rows_towards_the_current_target(
    cpg: str, *, steered: bool, seconds: float
) -> None:
    """One row per control call and one at the end, distances finite."""
    gecko = g.steerer()
    cpg_class = g.CLASSES[cpg]
    size = cpg_class(gecko.adjacency).num_of_parameters
    params = list(cpg_class(gecko.adjacency, seed=1).get_flat_params().numpy())
    if steered:
        params += [0.1] * STEERING_WEIGHTS
    assert len(params) == size + (STEERING_WEIGHTS if steered else 0)
    job = g.Steer(
        cpg=cpg,
        params=[float(value) for value in params],
        steered=steered,
        pair=PAIR,
        seconds=seconds,
    )
    rows = g.steer(job)
    assert rows.shape[1] == len(["time", "x", "y", "distance"])
    assert np.isfinite(rows).all()
    assert rows[-1, 0] >= seconds
    start = rows[0, 1:3]
    targets = g.targets_of(PAIR, start)
    after = rows[:, 0] >= g.SWITCH
    expected = np.where(
        after,
        np.linalg.norm(rows[:, 1:3] - targets[1], axis=1),
        np.linalg.norm(rows[:, 1:3] - targets[0], axis=1),
    )
    assert np.allclose(rows[:, 3], expected)


def test_steer_passes_every_step_to_after_step() -> None:
    """A film hook sees each step of a steering trial."""
    gecko = g.steerer()
    cpg = g.steering_cpg(
        "PhaseCPG", PHASE_GAIT, steered=False, nu=gecko.model.nu
    )
    steps: list[int] = []
    gecko.steer(cpg, PAIR, SHORT, lambda _data, step: steps.append(step))
    assert steps == list(range(1, len(steps) + 1))
    assert steps[-1] >= round(SHORT / gecko.model.opt.timestep)


def test_features_measure_the_target_from_the_core() -> None:
    """Distance capped at 1 m, bearing straight ahead gives sin 0, cos 1."""
    gecko = g.walker()
    mujoco.mj_resetData(gecko.model, gecko.data)
    mujoco.mj_forward(gecko.model, gecko.data)
    core = gecko.data.xpos[gecko.core, :2]
    forward = gecko.data.xmat[gecko.core].reshape(3, 3)[:2, 0]
    distance, sine, cosine = gecko.features(gecko.data, core + 5 * forward)
    assert distance == pytest.approx(1.0)
    assert sine == pytest.approx(0.0, abs=1e-9)
    assert cosine == pytest.approx(1.0)


def test_the_unsteered_cpg_has_only_its_sensor_routes() -> None:
    """steering_cpg adds the five steering routes only when steered."""
    plain = g.steering_cpg("PhaseCPG", PHASE_GAIT, steered=False, nu=8)
    assert isinstance(plain, PhaseCPG)
    assert plain.routes == ()


DIMENSION = 26
BUDGET = 17  # not a whole number of generations
TARGET = 0.3


def _optimizer(kind: str, seed: int) -> g.Optimizer:
    parametrization = ng.p.Array(shape=(DIMENSION,), lower=0, upper=1)
    parametrization.value = np.full(DIMENSION, 0.5)
    parametrization.random_state = np.random.RandomState(seed)
    if kind == "CMA":
        return ng.optimizers.CMA(parametrization=parametrization, budget=BUDGET)
    return ng.optimizers.ParametrizedCMA(scale=0.4)(
        parametrization=parametrization, budget=BUDGET
    )


def _loss(candidate: g.Candidate) -> float:
    return float(np.sum((np.asarray(candidate.value) - TARGET) ** 2))


@pytest.mark.parametrize("kind", ["CMA", "ParametrizedCMA"])
def test_lockstep_asks_what_one_at_a_time_asks(kind: str) -> None:
    """Whole generations in rounds give the one-at-a-time candidates."""
    alone = _optimizer(kind, 1)
    expected = []
    for _ in range(BUDGET):
        candidate = alone.ask()
        alone.tell(candidate, _loss(candidate))
        expected.append(np.asarray(candidate.value).copy())

    rounds: list[list[tuple[str, int]]] = []

    def evaluate(
        asked: list[tuple[str, int, g.Candidate]],
    ) -> list[float]:
        rounds.append([(key, index) for key, index, _ in asked])
        return [_loss(candidate) for _, _, candidate in asked]

    other = "CMA" if kind == "ParametrizedCMA" else "ParametrizedCMA"
    # Generations of 7 and 13: one search finishes a round before the other
    told = g.lockstep(
        {"a": _optimizer(kind, 1), "b": _optimizer(other, 2)}, BUDGET, evaluate
    )
    got = [np.asarray(candidate.value) for candidate, _ in told["a"]]
    assert np.array_equal(np.array(got), np.array(expected))
    assert [value for _, value in told["b"]] == [
        _loss(candidate) for candidate, _ in told["b"]
    ]
    indices = [
        index for round_ in rounds for key, index in round_ if key == "a"
    ]
    assert indices == list(range(BUDGET))
    assert len(rounds) > 1


def test_generation_size_reads_both_kinds_of_cma() -> None:
    """The meta CMA keeps its CMA-ES inside; the configured one is it."""
    meta, configured = _optimizer("CMA", 0), _optimizer("ParametrizedCMA", 0)
    meta.ask()
    configured.ask()
    assert g.generation_size(meta) == 7  # ruff: ignore[magic-value-comparison]  # measured 2026-10-23
    assert g.generation_size(configured) == 4 + int(3 * np.log(DIMENSION))


def test_walk_timed_returns_the_distance_and_seconds() -> None:
    """The timed job gives the same distance as walk."""
    job = g.Walk(cpg="PhaseCPG", params=PHASE_GAIT, seed=0)
    scored = g.walk_timed(job)
    assert scored.distance == PHASE_DISTANCE
    assert scored.seconds > 0
