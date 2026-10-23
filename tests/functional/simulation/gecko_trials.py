"""Gecko trials shared by the slow CPG notebooks and their pool workers.

The notebooks ``cpg_locomotion``, ``cpg_initialisation``,
``cpg_initialisation_search`` and ``cpg_steering`` score thousands of CPG
parameter vectors on the gecko. They send the vectors to the worker
processes of ``ariel.simulation.process_pool``, and a spawned worker can
only run a function it imports from a ``.py`` file, so the trials live
here and the notebooks import this module (the notebook runner puts this
folder on the kernel's path, and the workers inherit it).

Each process builds its gecko once, on first use (``walker()`` and
``steerer()``), and every trial resets it. A trial sets MuJoCo's control
callback, which is one global per process, so trials never run in threads.

``lockstep`` runs many CMA-ES searches together: each round, every search
asks one whole generation, all the trials of the round are scored in one
call (one ``pool.map``), and each search is told its values in the order it
asked. CMA-ES draws a generation at once and updates only when the whole
generation is told, so this gives exactly the candidates of asking and
telling one at a time.
"""

# Standard library
import time
from collections.abc import Callable, Mapping, Sequence
from functools import cache
from typing import Protocol

# Third-party libraries
import mujoco
import nevergrad as ng
import numpy as np
import numpy.typing as npt

# Local libraries
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.parameters import FrozenModel
from ariel.simulation import (
    CPG,
    AdaptiveHopfCPG,
    Controller,
    HopfCPG,
    InputRoute,
    LearnableTegotaeCPG,
    NaCPG,
    PhaseCPG,
    SimpleFlatWorld,
    SwingStanceCPG,
    TegotaeCPG,
    Tracker,
    create_fully_connected_adjacency,
)

type Floats = npt.NDArray[np.float64]
type AfterStep = Callable[[mujoco.MjData, int], None]
type Candidate = ng.p.Parameter


class Optimizer(Protocol):
    """What ``lockstep`` uses of a nevergrad optimizer."""

    def ask(self) -> Candidate:  # pragma: no cover  # typing stub only
        """Return the next candidate."""
        ...

    def tell(  # pragma: no cover  # typing stub only
        self, candidate: Candidate, loss: float
    ) -> None:
        """Report a candidate's loss."""
        ...


MODELS: list[type[CPG]] = [
    HopfCPG,
    AdaptiveHopfCPG,
    PhaseCPG,
    SwingStanceCPG,
    NaCPG,
    TegotaeCPG,
    LearnableTegotaeCPG,
]
CLASSES = {cpg_class.__name__: cpg_class for cpg_class in MODELS}
SECONDS = 10.0  # one walking trial
STEPS_PER_CONTROL = 4
SPAWN = (0.0, 0.0, 0.1)
RADIUS = 1.0  # metres from the start to targets A and B
SWITCH = 15.0  # seconds: the target jumps from A to B
STEER_SECONDS = 30.0  # one steering trial
STEERING = [
    InputRoute(handle="frequency"),  # u[0], distance to the target
    InputRoute(handle="offset"),  # u[1], sin of the bearing
    InputRoute(handle="amplitude"),  # u[2], cos of the bearing
    InputRoute(handle="amplitude"),  # u[3], sin of the bearing again
    InputRoute(handle="offset"),  # u[4], cos of the bearing again
]


def targets_of(pair: tuple[float, float], start: Floats) -> Floats:
    """Place targets A and B on the circle around the start.

    Returns
    -------
    Floats
        Row 0 is A, row 1 is B, in metres in the floor plane.
    """
    angles = np.array(pair)
    return start + RADIUS * np.stack([np.cos(angles), np.sin(angles)], axis=1)


class Gecko:
    """The gecko on a flat floor, compiled, with what the trials read.

    Parameters
    ----------
    target
        Add the red target marker of ``cpg_steering`` (a mocap body that
        touches nothing).
    """

    def __init__(self, *, target: bool = False) -> None:
        world = SimpleFlatWorld()
        world.spawn(gecko().spec, position=SPAWN)
        if target:
            marker = world.spec.worldbody.add_body(name="target", mocap=True)
            marker.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[0.03, 0.002, 0],
                rgba=[0.9, 0.1, 0.1, 1],
                contype=0,
                conaffinity=0,
            )
        self.world = world
        self.model = world.compile()
        self.data = mujoco.MjData(self.model)
        model = self.model
        self.core = next(
            i for i in range(model.nbody) if model.body(i).name.endswith("core")
        )
        self.target_mocap = (
            int(model.body("target").mocapid[0]) if target else -1
        )
        # qpos address of the joint each actuator drives
        self.joint_qpos = model.jnt_qposadr[model.actuator_trnid[:, 0]]
        self.adjacency = create_fully_connected_adjacency(model.nu)
        # limb[a, b] is 1 when actuator a's joint moves body b
        self.robot = model.body_rootid == model.body_rootid[self.core]
        joint_body = model.jnt_bodyid[model.actuator_trnid[:, 0]]
        self.limb = np.zeros((model.nu, model.nbody))
        for body in range(model.nbody):
            ancestor = body
            while ancestor != 0:
                self.limb[joint_body == ancestor, body] = 1.0
                ancestor = model.body_parentid[ancestor]
        self._contact_force = np.zeros(6)

    def floor_forces(self, d: mujoco.MjData) -> Floats:
        """Return the vertical floor force on each body of the robot.

        Returns
        -------
        Floats
            One force per body, in newtons.
        """
        model = self.model
        per_body = np.zeros(model.nbody)
        for index in range(d.ncon):
            contact = d.contact[index]
            first = model.geom_bodyid[contact.geom1]
            second = model.geom_bodyid[contact.geom2]
            if self.robot[first] == self.robot[second]:
                continue  # the robot touching itself, or two world pieces
            mujoco.mj_contactForce(model, d, index, self._contact_force)
            on = first if self.robot[first] else second
            per_body[on] += self._contact_force[0] * abs(contact.frame[2])
        return per_body

    def limb_loads(self, d: mujoco.MjData) -> Floats:
        """Sum the vertical floor force on the bodies each joint moves.

        Returns
        -------
        Floats
            One load per actuator, in newtons.
        """
        loads: Floats = self.limb @ self.floor_forces(d)
        return loads

    def _controller(
        self, callback: Callable[[mujoco.MjData], object]
    ) -> Controller:
        def control(
            _model: mujoco.MjModel,
            d: mujoco.MjData,
            /,
            *_args: object,
            **_kwargs: object,
        ) -> object:
            return callback(d)

        controller = Controller(
            controller_callback_function=control,
            time_steps_per_ctrl_step=STEPS_PER_CONTROL,
            time_steps_per_save=10**9,
            tracker=Tracker(mujoco.mjtObj.mjOBJ_BODY, "core", ["xpos"]),
        )
        controller.tracker.setup(self.world.spec, self.data)
        return controller

    def _run(
        self,
        controller: Controller,
        seconds: float,
        after_step: Callable[[int], None],
    ) -> None:
        data = self.data
        mujoco.set_mjcb_control(controller.set_control)
        step = 0
        try:
            while data.time < seconds:
                mujoco.mj_step(self.model, data)
                step += 1
                after_step(step)
        finally:
            mujoco.set_mjcb_control(None)

    def walk(
        self,
        cpg: CPG,
        seconds: float = SECONDS,
        after_step: AfterStep | None = None,
    ) -> float:
        """Run one walking trial, the trial of ``cpg_locomotion``.

        ``after_step(data, step)``, if given, runs after every physics
        step (a notebook films with it).

        Returns
        -------
        float
            The distance the core travelled in the floor plane, in metres.
        """
        data = self.data
        mujoco.mj_resetData(self.model, data)
        mujoco.mj_forward(self.model, data)
        start = data.xpos[self.core, :2].copy()

        def callback(d: mujoco.MjData) -> object:
            if isinstance(cpg, AdaptiveHopfCPG):
                cpg.set_input(d.qpos[self.joint_qpos] - d.ctrl)
            return cpg.forward(d.time).numpy()

        def step_done(step: int) -> None:
            # Read the loads after the step: inside the control callback
            # the contacts are new but the contact forces are still the
            # last step's, so mj_contactForce would read stale values there
            if isinstance(cpg, TegotaeCPG):  # LearnableTegotaeCPG too
                cpg.set_input(self.limb_loads(data))
            if after_step is not None:
                after_step(data, step)

        self._run(self._controller(callback), seconds, step_done)
        return float(np.linalg.norm(data.xpos[self.core, :2] - start))

    def features(self, d: mujoco.MjData, target: Floats) -> list[float]:
        """Return u's steering entries: distance, sin and cos of the bearing.

        Returns
        -------
        list[float]
            Distance capped at 1 m, then the sine and cosine of the angle
            from the core's local x axis to the target.
        """
        offset = target - d.xpos[self.core, :2]
        forward = d.xmat[self.core].reshape(3, 3)[:2, 0]
        bearing = np.arctan2(
            forward[0] * offset[1] - forward[1] * offset[0], forward @ offset
        )
        return [
            min(float(np.linalg.norm(offset)), 1.0),
            float(np.sin(bearing)),
            float(np.cos(bearing)),
        ]

    def steer(
        self,
        cpg: CPG,
        pair: tuple[float, float],
        seconds: float = STEER_SECONDS,
        after_step: AfterStep | None = None,
    ) -> Floats:
        """Run one steering trial: towards A, then B from SWITCH on.

        Needs the gecko built with ``target=True``.

        Returns
        -------
        Floats
            One row per control call, then one for the end: time, the
            core's x and y, and its distance to the current target.
        """
        data = self.data
        mujoco.mj_resetData(self.model, data)
        mujoco.mj_forward(self.model, data)
        start = data.xpos[self.core, :2].copy()
        targets = targets_of(pair, start)
        steered = len(cpg.routes) > len(cpg.FAST_HANDLES) * self.model.nu
        sensor = np.zeros(self.model.nu)
        rows: list[list[float]] = []

        def current(now: float) -> Floats:
            chosen: Floats = targets[0] if now < SWITCH else targets[1]
            return chosen

        def row(d: mujoco.MjData) -> list[float]:
            target = current(d.time)
            distance = np.linalg.norm(d.xpos[self.core, :2] - target)
            return [d.time, *d.xpos[self.core, :2], float(distance)]

        def callback(d: mujoco.MjData) -> object:
            target = current(d.time)
            d.mocap_pos[self.target_mocap] = [*target, 0.002]
            if isinstance(cpg, AdaptiveHopfCPG):
                sensor[:] = d.qpos[self.joint_qpos] - d.ctrl
            u = [float(value) for value in sensor] if cpg.FAST_HANDLES else []
            if steered:
                distance, sine, cosine = self.features(d, target)
                u += [distance, sine, cosine, sine, cosine]
            cpg.set_input(u)
            rows.append(row(d))
            return cpg.forward(d.time).numpy()

        def step_done(step: int) -> None:
            # Read the loads after the step (as in walk)
            if isinstance(cpg, TegotaeCPG):  # LearnableTegotaeCPG too
                sensor[:] = self.limb_loads(data)
            if after_step is not None:
                after_step(data, step)

        self._run(self._controller(callback), seconds, step_done)
        rows.append(row(data))
        return np.array(rows)


@cache
def walker() -> Gecko:
    """Return this process's gecko for walking trials, built once.

    Returns
    -------
    Gecko
        The gecko without a target marker.
    """
    return Gecko()


@cache
def steerer() -> Gecko:
    """Return this process's gecko for steering trials, built once.

    Returns
    -------
    Gecko
        The gecko with the target marker.
    """
    return Gecko(target=True)


def steering_cpg(
    name: str, params: list[float], *, steered: bool, nu: int
) -> CPG:
    """Build a CPG with its sensory inputs, and the steering ones if asked.

    The CPG keeps its sensory inputs (joint error for AdaptiveHopfCPG, limb
    loads for the Tegotae models) as the first entries of u. ``params``
    holds a gait and then, when steered, the steering weights, in the
    order of ``get_flat_params``.

    Returns
    -------
    CPG
        The network with ``params`` set.
    """
    cpg_class = CLASSES[name]
    sensors = [
        InputRoute(handle=handle, joint=joint)
        for handle in cpg_class.FAST_HANDLES
        for joint in range(nu)
    ]
    routes = [*sensors, *STEERING] if steered else sensors
    cpg = cpg_class(create_fully_connected_adjacency(nu), inputs=routes)
    cpg.set_flat_params(np.array(params, dtype=np.float64))
    return cpg


class Walk(FrozenModel):
    """One walking trial to run in a worker: a CPG and its parameters."""

    cpg: str
    params: list[float]
    seed: int | None = None


class Steer(FrozenModel):
    """One steering trial to run in a worker."""

    cpg: str
    params: list[float]
    steered: bool
    pair: tuple[float, float]
    seconds: float = STEER_SECONDS


def walk(job: Walk) -> float:
    """Score one walking trial; the function a notebook sends to the pool.

    Returns
    -------
    float
        The distance walked, in metres.
    """
    gecko_ = walker()
    cpg = CLASSES[job.cpg](gecko_.adjacency, seed=job.seed)
    cpg.set_flat_params(np.array(job.params, dtype=np.float64))
    return gecko_.walk(cpg)


def steer(job: Steer) -> Floats:
    """Run one steering trial; the function a notebook sends to the pool.

    Returns
    -------
    Floats
        The trial's rows, as ``Gecko.steer`` returns them.
    """
    gecko_ = steerer()
    cpg = steering_cpg(
        job.cpg, job.params, steered=job.steered, nu=gecko_.model.nu
    )
    return gecko_.steer(cpg, job.pair, seconds=job.seconds)


class Scored(FrozenModel):
    """A trial's distance and the seconds it took in its worker."""

    distance: float
    seconds: float


def walk_timed(job: Walk) -> Scored:
    """Score one walking trial and time it.

    Returns
    -------
    Scored
        The distance walked, in metres, and the trial's wall time.
    """
    started = time.perf_counter()
    distance = walk(job)
    return Scored(distance=distance, seconds=time.perf_counter() - started)


def steer_timed(job: Steer) -> tuple[Floats, float]:
    """Run one steering trial and time it.

    Returns
    -------
    tuple[Floats, float]
        The trial's rows, as ``steer`` returns them, and its wall time.
    """
    started = time.perf_counter()
    rows = steer(job)
    return rows, time.perf_counter() - started


def generation_size(optimizer: Optimizer) -> int:
    """Return how many candidates a CMA-ES optimizer draws per generation.

    Call it after the optimizer's first ``ask``, which builds the inner
    CMA-ES. ``ng.optimizers.CMA`` keeps it in ``_optim``; a
    ``ParametrizedCMA(...)`` optimizer is the inner CMA-ES itself. Neither
    has a public reader for the size.

    Returns
    -------
    int
        The population size.
    """
    inner = vars(optimizer).get("_optim") or optimizer
    return int(vars(inner)["_popsize"])


def lockstep[K](
    optimizers: Mapping[K, Optimizer],
    budget: int,
    evaluate: Callable[[list[tuple[K, int, Candidate]]], Sequence[float]],
) -> dict[K, list[tuple[Candidate, float]]]:
    """Run CMA-ES searches together, one generation of each per round.

    ``evaluate`` gets every candidate of a round as ``(key, index,
    candidate)``, with ``index`` the candidate's place in its own search
    (0 to ``budget - 1``), and returns one value per candidate, in order.
    Each optimizer is told those values in the order it asked; a search
    stops after ``budget`` candidates, its last generation cut short.

    Returns
    -------
    dict[K, list[tuple[Candidate, float]]]
        Every search's candidates and told values, in asking order.
    """
    told: dict[K, list[tuple[Candidate, float]]] = {
        key: [] for key in optimizers
    }
    while any(len(done) < budget for done in told.values()):
        asked: list[tuple[K, int, Candidate]] = []
        for key, optimizer in optimizers.items():
            start = len(told[key])
            if start == budget:
                continue
            batch = [optimizer.ask()]
            size = min(generation_size(optimizer), budget - start)
            batch += [optimizer.ask() for _ in range(size - 1)]
            asked += [
                (key, start + offset, candidate)
                for offset, candidate in enumerate(batch)
            ]
        values = evaluate(asked)
        for (key, _, candidate), value in zip(asked, values, strict=True):
            optimizers[key].tell(candidate, value)
            told[key].append((candidate, value))
    return told
