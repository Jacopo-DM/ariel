"""
Shared machinery of the central pattern generators in this package.

A central pattern generator (CPG) here is a network of ``n`` coupled
oscillators, one per joint. Each call of ``forward`` moves every oscillator
forward in time and returns one joint angle per oscillator.

The clock follows the simulation. ``forward(time)`` integrates over the time
that has passed since the previous timed call, in equal sub-steps of at most
``dt``. A controller that is called every 0.125 s of simulation therefore moves
a 1 Hz CPG through 0.125 of a cycle per call. The first timed call, and any
call with a time earlier than the previous one (a new episode), resets the
state first. ``forward()`` without a time advances by one step of ``dt``.

Every CPG takes one input vector ``u`` through ``set_input``, held until the
next call. ``inputs`` says where each entry goes. An ``InputRoute`` names a
handle and either one joint, which gets the value as it is, or no joint: then
every joint gets the value times a weight of its own, from the parameter group
``input_weights``, which is searched with the other parameters. With the
routes ``[InputRoute(handle="offset")]`` on two joints and the weights
``(0.2, -0.2)``, ``u = (1,)`` bends the first joint 0.2 rad one way and the
second 0.2 rad the other way.

The slow handles change the gait. With ``d`` the value routed to a joint,
``amplitude`` and ``frequency`` multiply that joint's amplitude and frequency
by ``1 + d`` (never below 0) and ``offset`` adds ``d`` radians to its offset.
They follow ``d`` through a critically damped filter,
``s'' = k^2 (d - s) - 2 k s'`` with ``k = SMOOTHING``, so a jump of ``u`` never
makes a servo jump: 96 % of a change is done after ``5 / k`` = 0.5 s. A model
may add fast handles (``FAST_HANDLES``), used at once inside its equations: the
load of the Tegotae models, the teaching signal of ``AdaptiveHopfCPG``. Its
default routes send entry ``i`` of ``u`` to the fast handle of joint ``i``.

Subclasses name their parameter groups and their sampling ranges in
``PARAMETER_RANGES`` and implement ``_reset``, ``_step`` and ``_angles``. A
group holds one value per oscillator unless ``shared_group_sizes`` gives it
another size (for example the weights of a network every joint uses).
"""

# Standard library
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

# Third-party libraries
import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

# Local libraries
from ariel.parameters import FrozenModel

__all__ = [
    "CPG",
    "BetaParams",
    "InputRoute",
    "ParamInput",
    "create_fully_connected_adjacency",
    "scaled_beta",
]

# Global constants
E = 1e-9
SLOW_HANDLES = ("amplitude", "offset", "frequency")
SMOOTHING = 10.0  # 1/s, the constant k of the filter on the slow handles

type ParamInput = list[float] | NDArray[np.float64] | torch.Tensor


class BetaParams(FrozenModel):
    """Shape parameters (a, b) and bounds (c, d) of a scaled beta."""

    a: float
    b: float
    c: float
    d: float


class InputRoute(FrozenModel):
    """Where one entry of the input vector ``u`` goes.

    ``handle`` names a slow handle (``amplitude``, ``offset``, ``frequency``)
    or one of the model's ``FAST_HANDLES``. ``joint`` sends the value to that
    joint alone; None sends it to every joint, times a learned weight each.
    """

    handle: str
    joint: int | None = None


def scaled_beta(
    params: BetaParams,
    size: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Sample a beta distribution scaled to the range ``[c, d]``.

    Parameters
    ----------
    params : BetaParams
        The shape parameters and the bounds.
    size : int
        The number of samples.
    rng : np.random.Generator
        The source of randomness.

    Returns
    -------
    torch.Tensor
        The samples, as float32.
    """
    lower = min(params.c, params.d)
    upper = max(params.c, params.d)
    sample = lower + (upper - lower) * rng.beta(params.a, params.b, size)
    return torch.tensor(sample, dtype=torch.float32)


def uniform(lower: float, upper: float) -> BetaParams:
    """Return the beta parameters of a uniform range.

    Returns
    -------
    BetaParams
        ``a = b = 1`` between ``lower`` and ``upper``.
    """
    return BetaParams(a=1, b=1, c=lower, d=upper)


def create_fully_connected_adjacency(num_nodes: int) -> dict[int, list[int]]:
    """Connect every node to every other node.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the CPG network.

    Returns
    -------
    dict[int, list[int]]
        Node index to the indices of all the other nodes.
    """
    return {
        i: [j for j in range(num_nodes) if j != i] for i in range(num_nodes)
    }


class CPG(nn.Module, ABC):
    """A network of coupled oscillators that outputs one angle per joint."""

    PARAMETER_RANGES: ClassVar[dict[str, BetaParams]]
    FAST_HANDLES: ClassVar[tuple[str, ...]] = ()
    angles: torch.Tensor
    inputs: torch.Tensor
    smoothed: torch.Tensor
    smoothed_rate: torch.Tensor
    targets: dict[str, torch.Tensor]

    def __init__(
        self,
        adjacency_dict: dict[int, list[int]],
        dt: float = 0.01,
        angle_limits: tuple[float, float] | None = (
            -math.pi / 2,
            math.pi / 2,
        ),
        seed: int | None = None,
        *,
        coupling: float = 1.0,
        clipping: bool = True,
        angle_tracking: bool = False,
        inputs: Sequence[InputRoute] | None = None,
    ) -> None:
        """Build the network and sample its parameters.

        Parameters
        ----------
        adjacency_dict : dict[int, list[int]]
            Node index to the indices of the nodes that drive it. The keys
            are ``0 .. n - 1``.
        dt : float, optional
            Largest integration step, in seconds.
        angle_limits : tuple[float, float] | None, optional
            Lower and upper bound of the output angles, in radians.
        seed : int | None, optional
            Seed of the random initial parameters.
        coupling : float, optional
            Strength of the coupling between connected oscillators.
        clipping : bool, optional
            If True, the output angles are clipped to ``angle_limits``. The
            mean distance to the limits is recorded in ``clamping_error``
            either way.
        angle_tracking : bool, optional
            If True, every output is appended to ``angle_history``.
        inputs : Sequence[InputRoute] | None, optional
            Where each entry of the vector given to ``set_input`` goes. None
            routes entry ``i`` to each fast handle of joint ``i``, and gives
            no input to a model without fast handles.

        Raises
        ------
        ValueError
            If the keys of ``adjacency_dict`` are not ``0 .. n - 1``, a
            neighbour is not a node, or a route names an unknown handle or
            joint.
        """
        super().__init__()

        self.adjacency_dict = adjacency_dict
        self.n = len(adjacency_dict)
        if set(adjacency_dict) != set(range(self.n)):
            msg = "The keys of the adjacency dict must be 0 .. n - 1."
            raise ValueError(msg)
        self.adjacency = torch.zeros(self.n, self.n)
        for node, neighbours in adjacency_dict.items():
            for neighbour in neighbours:
                if not 0 <= neighbour < self.n:
                    msg = f"Node {node} has the unknown neighbour {neighbour}."
                    raise ValueError(msg)
                self.adjacency[node, neighbour] = 1.0

        self._build_routes(inputs)

        self.dt = dt
        self.coupling = coupling
        self.angle_limits = angle_limits
        self.clipping = clipping
        self.clamping_error = 0.0
        self.angle_tracking = angle_tracking
        self.angle_history: list[list[float]] = []
        self.last_time: float | None = None

        rng = np.random.default_rng(seed)
        self.parameter_groups: dict[str, nn.Parameter] = {}
        self.parameter_ranges = dict(self.PARAMETER_RANGES)
        self.group_sizes = (
            dict.fromkeys(self.PARAMETER_RANGES, self.n)
            | self.shared_group_sizes()
        )
        if self.weighted:
            self.parameter_ranges["input_weights"] = uniform(-1, 1)
            self.group_sizes["input_weights"] = self.n * len(self.weighted)
        for name, params in self.parameter_ranges.items():
            group = nn.Parameter(
                scaled_beta(params, self.group_sizes[name], rng),
                requires_grad=False,
            )
            self.register_parameter(name, group)
            self.parameter_groups[name] = group
        self.num_of_parameters = sum(self.group_sizes.values())
        self.num_of_parameter_groups = len(self.parameter_groups)
        self.angles = torch.zeros(self.n)
        self._reset_inputs()

    def _build_routes(self, inputs: Sequence[InputRoute] | None) -> None:
        """Check the routes and turn them into matrices and weight rows.

        Raises
        ------
        ValueError
            If a route names an unknown handle or joint.
        """
        self.handles = SLOW_HANDLES + self.FAST_HANDLES
        self.routes = tuple(
            [
                InputRoute(handle=handle, joint=joint)
                for handle in self.FAST_HANDLES
                for joint in range(self.n)
            ]
            if inputs is None
            else inputs
        )
        # Routes with a joint: a fixed matrix per handle, from u to joints
        self.direct = {
            handle: torch.zeros(self.n, len(self.routes))
            for handle in self.handles
        }
        # Routes without a joint: (entry of u, handle), one weight per joint
        self.weighted: list[tuple[int, str]] = []
        for entry, route in enumerate(self.routes):
            if route.handle not in self.handles:
                msg = (
                    f"Unknown handle '{route.handle}'; "
                    f"this CPG has {self.handles}."
                )
                raise ValueError(msg)
            if route.joint is None:
                self.weighted.append((entry, route.handle))
            elif 0 <= route.joint < self.n:
                self.direct[route.handle][route.joint, entry] = 1.0
            else:
                msg = f"A route goes to the unknown joint {route.joint}."
                raise ValueError(msg)

    def shared_group_sizes(self) -> dict[str, int]:  # ruff: ignore[no-self-use]  # subclasses with shared groups override it
        """Return the size of every group that is not one value per joint.

        Returns
        -------
        dict[str, int]
            Group name to size; empty when every group has ``n`` values.
        """
        return {}

    def group(self, name: str) -> torch.Tensor:
        """Return the current values of one parameter group.

        Returns
        -------
        torch.Tensor
            One value per oscillator.
        """
        return self.parameter_groups[name].data

    # --- parameters ---------------------------------------------------------

    def param_type_converter(  # ruff: ignore[no-self-use]  # part of the public interface of every CPG
        self, params: ParamInput
    ) -> torch.Tensor:
        """Convert input parameters to a float32 tensor.

        Parameters
        ----------
        params : ParamInput
            A list, a NumPy array or a tensor.

        Returns
        -------
        torch.Tensor
            The parameters as a tensor.
        """
        if isinstance(params, list):
            return torch.tensor(params, dtype=torch.float32)
        if isinstance(params, np.ndarray):
            return torch.from_numpy(params).float()
        return params

    def set_flat_params(self, params: ParamInput) -> None:
        """Set all adaptable parameters from one flat vector.

        Parameters
        ----------
        params : ParamInput
            The groups in the order of ``parameter_groups``, one after
            another.

        Raises
        ------
        ValueError
            If the parameter vector has the wrong size.
        """
        safe_params = self.param_type_converter(params)
        if safe_params.numel() != self.num_of_parameters:
            msg = "Parameter vector has incorrect size. "
            msg += (
                f"Expected {self.num_of_parameters}, got {safe_params.numel()}."
            )
            raise ValueError(msg)
        start = 0
        for name, param in self.parameter_groups.items():
            size = self.group_sizes[name]
            param.data = safe_params[start : start + size].clone()
            start += size

    def set_param_with_dict(self, params: dict[str, ParamInput]) -> None:
        """Set parameter groups from a dict of group names to vectors.

        Parameters
        ----------
        params : dict[str, ParamInput]
            Group names as keys, one value per oscillator.
        """
        for key, value in params.items():
            self.set_params_by_group(key, value)

    def set_params_by_group(self, group_name: str, params: ParamInput) -> None:
        """Set one parameter group.

        Parameters
        ----------
        group_name : str
            A key of ``parameter_groups``.
        params : ParamInput
            ``group_sizes[group_name]`` values, one per oscillator for most
            groups.

        Raises
        ------
        ValueError
            If the group does not exist or the vector has the wrong size.
        """
        safe_params = self.param_type_converter(params)
        if group_name not in self.parameter_groups:
            msg = f"Parameter group '{group_name}' does not exist."
            raise ValueError(msg)
        param = self.parameter_groups[group_name]
        size = self.group_sizes[group_name]
        if safe_params.numel() != size:
            msg = (
                f"Parameter vector has incorrect size for group '{group_name}'."
            )
            raise ValueError(msg)
        param.data = safe_params.reshape(size).clone()

    def get_flat_params(self) -> torch.Tensor:
        """Get all adaptable parameters as one flat tensor.

        Returns
        -------
        torch.Tensor
            The groups in the order of ``parameter_groups``.
        """
        return torch.cat([p.flatten() for p in self.parameter_groups.values()])

    def parameter_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the sampling range of every flat parameter.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The lower and the upper bounds, in the order of
            ``get_flat_params``.
        """
        sizes = torch.tensor(list(self.group_sizes.values()))
        ranges = self.parameter_ranges.values()
        lower = torch.tensor([min(params.c, params.d) for params in ranges])
        upper = torch.tensor([max(params.c, params.d) for params in ranges])
        return (
            lower.repeat_interleave(sizes),
            upper.repeat_interleave(sizes),
        )

    # --- inputs -------------------------------------------------------------

    def set_input(self, values: ParamInput) -> None:
        """Set the input vector ``u`` used by the following steps.

        Parameters
        ----------
        values : ParamInput
            One value per route in ``routes``.

        Raises
        ------
        ValueError
            If the vector does not have one value per route.
        """
        safe_values = self.param_type_converter(values).flatten()
        if safe_values.numel() != len(self.routes):
            msg = f"Expected {len(self.routes)} inputs, got "
            msg += f"{safe_values.numel()}."
            raise ValueError(msg)
        self.inputs = safe_values.to(torch.float32).clone()
        self._update_targets()

    def handle(self, name: str) -> torch.Tensor:
        """Return the value a handle has now at every joint.

        Returns
        -------
        torch.Tensor
            The filtered value for a slow handle, the routed input itself
            for a fast one; one value per joint.
        """
        if name in SLOW_HANDLES:
            return self.smoothed[SLOW_HANDLES.index(name)]
        return self.targets[name]

    def amplitude_factor(self) -> torch.Tensor:
        """Return the factor the inputs put on every amplitude.

        Returns
        -------
        torch.Tensor
            ``1 + amplitude`` handle, never below 0.
        """
        return torch.clamp(1 + self.handle("amplitude"), min=0.0)

    def frequency_factor(self) -> torch.Tensor:
        """Return the factor the inputs put on every frequency.

        Returns
        -------
        torch.Tensor
            ``1 + frequency`` handle, never below 0.
        """
        return torch.clamp(1 + self.handle("frequency"), min=0.0)

    def _update_targets(self) -> None:
        """Route ``u`` to the handles with the current weights."""
        targets = {
            handle: matrix @ self.inputs
            for handle, matrix in self.direct.items()
        }
        if self.weighted:
            weights = self.group("input_weights").reshape(
                len(self.weighted), self.n
            )
            for row, (entry, handle) in zip(
                weights, self.weighted, strict=True
            ):
                targets[handle] += row * self.inputs[entry]
        self.targets = targets

    def _smooth(self, step: float) -> None:
        """Move the slow handles towards their targets over ``step``."""
        target = torch.stack([self.targets[name] for name in SLOW_HANDLES])
        acceleration = (
            SMOOTHING**2 * (target - self.smoothed)
            - 2 * SMOOTHING * self.smoothed_rate
        )
        self.smoothed_rate += acceleration * step
        self.smoothed += self.smoothed_rate * step

    def _reset_inputs(self) -> None:
        """Set ``u`` to zero and the slow handles to rest at zero."""
        self.inputs = torch.zeros(len(self.routes))
        self.smoothed = torch.zeros(len(SLOW_HANDLES), self.n)
        self.smoothed_rate = torch.zeros(len(SLOW_HANDLES), self.n)
        self._update_targets()

    # --- state and time -----------------------------------------------------

    def reset_state(self) -> None:
        """Put every oscillator at its start state and forget the history.

        The input goes back to zero, so ``set_input`` after the reset.
        """
        self._reset_inputs()
        self._reset()
        self.angles = torch.zeros(self.n)
        self.angle_history = []
        self.last_time = None

    def forward(  # ty: ignore[invalid-mutable-override]  # torch types forward as a Callable attribute; models override it with a method
        self, time: float | None = None
    ) -> torch.Tensor:
        """Advance every oscillator and return the angles.

        Parameters
        ----------
        time : float | None, optional
            The simulation time. Without it the network moves by ``dt``.

        Returns
        -------
        torch.Tensor
            One angle per oscillator, clipped to ``angle_limits`` if
            ``clipping`` is set.

        Raises
        ------
        ValueError
            If the angles contain NaN.
        """
        with torch.inference_mode():
            if time is None:
                self._update_targets()
                self._smooth(self.dt)
                self._step(self.dt)
            else:
                if self.last_time is None or time < self.last_time:
                    self.reset_state()
                    self.last_time = time
                self._update_targets()
                elapsed = time - self.last_time
                steps = math.ceil(elapsed / self.dt - E)
                for _ in range(steps):
                    self._smooth(elapsed / steps)
                    self._step(elapsed / steps)
                self.last_time = time
            angles = self._angles()

            if self.angle_limits is not None:
                clamped = torch.clamp(
                    angles, min=self.angle_limits[0], max=self.angle_limits[1]
                )
                # Distance to the limits, usable as a loss
                self.clamping_error = (angles - clamped).abs().mean().item()
                if self.clipping:
                    angles = clamped

            if self.angle_tracking:
                self.angle_history.append(angles.tolist())

        if torch.isnan(angles).any():
            msg = "NaN values detected in the angle signal.\n"
            msg += "".join(
                f"{name}={group.data}\n"
                for name, group in self.parameter_groups.items()
            )
            raise ValueError(msg)

        self.angles = angles
        return self.angles.clone()

    @abstractmethod
    def _reset(self) -> None:
        """Set the start state of every oscillator."""

    @abstractmethod
    def _step(self, step: float) -> None:
        """Integrate the oscillators over ``step`` seconds."""

    @abstractmethod
    def _angles(self) -> torch.Tensor:
        """Return the angles of the current state."""

    def phase_offsets(self) -> torch.Tensor:
        """Return the target phase of each oscillator minus that of the others.

        Returns
        -------
        torch.Tensor
            ``phase_i - phase_j`` at ``[i, j]``.
        """
        phase = self.group("phase")
        return phase[:, None] - phase[None, :]

    # --- save and load ------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the adaptable parameters to a file.

        Parameters
        ----------
        path : str | Path
            File path to save the parameters.
        """
        torch.save(
            {
                name: group.detach().cpu()
                for name, group in self.parameter_groups.items()
            },
            Path(path),
        )

    def load(self, path: str | Path) -> None:
        """Load the adaptable parameters from a file.

        Parameters
        ----------
        path : str | Path
            File path to load the parameters from.
        """
        loaded = torch.load(Path(path), map_location="cpu", weights_only=True)
        for name, group in self.parameter_groups.items():
            group.data = loaded[name]
