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

Subclasses name their parameter groups and their sampling ranges in
``PARAMETER_RANGES`` and implement ``_reset``, ``_step`` and ``_angles``. A
group holds one value per oscillator unless ``shared_group_sizes`` gives it
another size (for example the weights of a network every joint uses).
"""

# Standard library
import math
from abc import ABC, abstractmethod
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
    "ParamInput",
    "create_fully_connected_adjacency",
    "scaled_beta",
]

# Global constants
E = 1e-9

type ParamInput = list[float] | NDArray[np.float64] | torch.Tensor


class BetaParams(FrozenModel):
    """Shape parameters (a, b) and bounds (c, d) of a scaled beta."""

    a: float
    b: float
    c: float
    d: float


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
    angles: torch.Tensor

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

        Raises
        ------
        ValueError
            If the keys of ``adjacency_dict`` are not ``0 .. n - 1`` or a
            neighbour is not a node.
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
        self.group_sizes = (
            dict.fromkeys(self.PARAMETER_RANGES, self.n)
            | self.shared_group_sizes()
        )
        for name, params in self.PARAMETER_RANGES.items():
            group = nn.Parameter(
                scaled_beta(params, self.group_sizes[name], rng),
                requires_grad=False,
            )
            self.register_parameter(name, group)
            self.parameter_groups[name] = group
        self.num_of_parameters = sum(self.group_sizes.values())
        self.num_of_parameter_groups = len(self.parameter_groups)
        self.angles = torch.zeros(self.n)

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
        ranges = self.PARAMETER_RANGES.values()
        lower = torch.tensor([min(params.c, params.d) for params in ranges])
        upper = torch.tensor([max(params.c, params.d) for params in ranges])
        return (
            lower.repeat_interleave(sizes),
            upper.repeat_interleave(sizes),
        )

    # --- state and time -----------------------------------------------------

    def reset_state(self) -> None:
        """Put every oscillator at its start state and forget the history."""
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
                self._step(self.dt)
            else:
                if self.last_time is None or time < self.last_time:
                    self.reset_state()
                    self.last_time = time
                elapsed = time - self.last_time
                steps = math.ceil(elapsed / self.dt - E)
                for _ in range(steps):
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
