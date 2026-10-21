"""
Hopf oscillators with phase coupling, plain and with an adaptive frequency.

Each oscillator ``i`` is a Hopf oscillator (Righetti and Ijspeert, ICRA 2006,
equations 1 and 2) whose rotation speed also carries a Kuramoto term that pulls
it to its target phase difference with its neighbours::

    dx_i/dt = gamma (1 - r_i^2) x_i - v_i y_i
    dy_i/dt = gamma (1 - r_i^2) y_i + v_i x_i
    v_i     = w_i + h * sum_j sin((theta_j - phi_j) - (theta_i - phi_i))
    theta_i = atan2(y_i, x_i)
    angle_i = offset_i + A_i * x_i

The limit cycle is the unit circle, so ``A_i`` alone sets the amplitude and
``gamma`` how fast a perturbed oscillator returns to the cycle. When the
oscillators are locked, ``theta_i - theta_j = phi_i - phi_j``.

``AdaptiveHopfCPG`` turns the frequency into a state that learns from an input
signal ``F_i`` (Righetti and Ijspeert, ICRA 2006, equations 1 to 3)::

    dx_i/dt   = ... + eps F_i
    dw_i/dt   = -eps F_i y_i / r_i

With a periodic input the frequency converges to one of the input's
frequencies. With no input it behaves like ``HopfCPG``. The paper's networks
that learn a whole teaching signal (equations 4 to 11) are not built here: an
evolved controller has no teaching signal. The input is set with ``set_input``
before each ``forward``; feeding back, for example, the difference between
the measured and the commanded joint angle lets the frequency follow the body.

References
----------
    [1] L. Righetti and A. J. Ijspeert, "Programmable central pattern
        generators: an application to biped locomotion control", ICRA 2006.
"""

# Standard library
import math
from typing import ClassVar

# Third-party libraries
import torch
from torch import nn

# Local libraries
from ariel.simulation.controllers.cpg._base import (
    CPG,
    BetaParams,
    E,
    ParamInput,
    uniform,
)

__all__ = ["AdaptiveHopfCPG", "HopfCPG"]


class HopfCPG(CPG):
    """Hopf oscillators with Kuramoto phase coupling."""

    PARAMETER_RANGES: ClassVar[dict[str, BetaParams]] = {
        "phase": uniform(0, 2 * math.pi),
        "w": uniform(0, 4 * math.pi),  # up to 2 Hz
        "amplitudes": uniform(0, math.pi / 2),
        "offset": uniform(-0.5, 0.5),
    }

    phase: nn.Parameter
    w: nn.Parameter
    amplitudes: nn.Parameter
    offset: nn.Parameter
    xy: torch.Tensor

    def __init__(
        self,
        adjacency_dict: dict[int, list[int]],
        gamma: float = 10.0,
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
        """Initialize the network.

        Parameters
        ----------
        adjacency_dict : dict[int, list[int]]
            Node index to the indices of the nodes that drive it.
        gamma : float, optional
            Speed of convergence to the limit cycle, in 1/s.
        dt : float, optional
            Largest Euler step, in seconds.
        angle_limits : tuple[float, float] | None, optional
            Lower and upper bound of the output angles, in radians.
        seed : int | None, optional
            Seed of the random initial parameters.
        coupling : float, optional
            Strength ``h`` of the phase coupling, in rad/s per neighbour.
        clipping : bool, optional
            If True, the output angles are clipped to ``angle_limits``.
        angle_tracking : bool, optional
            If True, every output is appended to ``angle_history``.
        """
        super().__init__(
            adjacency_dict,
            dt,
            angle_limits,
            seed,
            coupling=coupling,
            clipping=clipping,
            angle_tracking=angle_tracking,
        )
        self.gamma = gamma
        self.reset_state()

    def _reset(self) -> None:
        """Put every oscillator on the unit circle at its target phase."""
        phase = self.group("phase")
        self.xy = torch.stack([torch.cos(phase), torch.sin(phase)], dim=1)

    def _rates(self, frequency: torch.Tensor) -> torch.Tensor:
        """Return the time derivative of the state.

        Parameters
        ----------
        frequency : torch.Tensor
            The intrinsic frequency of each oscillator, in rad/s.

        Returns
        -------
        torch.Tensor
            ``(dx/dt, dy/dt)`` per oscillator.
        """
        x = self.xy[:, 0]
        y = self.xy[:, 1]
        relative = torch.atan2(y, x) - self.group("phase")
        pull = (
            torch.sin(relative[None, :] - relative[:, None]) * self.adjacency
        ).sum(dim=1)
        rotation = frequency + self.coupling * pull
        gain = self.gamma * (1 - (x**2 + y**2))
        return torch.stack(
            [gain * x - rotation * y, gain * y + rotation * x], dim=1
        )

    def _step(self, step: float) -> None:
        """Take one Euler step of ``step`` seconds."""
        self.xy += self._rates(self.group("w")) * step

    def _angles(self) -> torch.Tensor:
        """Return ``offset_i + A_i * x_i``.

        Returns
        -------
        torch.Tensor
            One angle per oscillator.
        """
        return self.group("offset") + self.group("amplitudes") * self.xy[:, 0]


class AdaptiveHopfCPG(HopfCPG):
    """Hopf oscillators whose frequencies adapt to an input signal."""

    frequency: torch.Tensor
    inputs: torch.Tensor

    def __init__(  # ruff: ignore[too-many-arguments]  # one argument per constant of the paper's equations
        self,
        adjacency_dict: dict[int, list[int]],
        epsilon: float = 1.0,
        gamma: float = 10.0,
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
        """Initialize the network.

        Parameters
        ----------
        adjacency_dict : dict[int, list[int]]
            Node index to the indices of the nodes that drive it.
        epsilon : float, optional
            Strength ``eps`` of the input, which also sets how fast the
            frequency adapts.
        gamma : float, optional
            Speed of convergence to the limit cycle, in 1/s.
        dt : float, optional
            Largest Euler step, in seconds.
        angle_limits : tuple[float, float] | None, optional
            Lower and upper bound of the output angles, in radians.
        seed : int | None, optional
            Seed of the random initial parameters.
        coupling : float, optional
            Strength ``h`` of the phase coupling, in rad/s per neighbour.
        clipping : bool, optional
            If True, the output angles are clipped to ``angle_limits``.
        angle_tracking : bool, optional
            If True, every output is appended to ``angle_history``.
        """
        self.epsilon = epsilon
        super().__init__(
            adjacency_dict,
            gamma,
            dt,
            angle_limits,
            seed,
            coupling=coupling,
            clipping=clipping,
            angle_tracking=angle_tracking,
        )

    def set_input(self, values: ParamInput) -> None:
        """Set the input ``F_i`` used by the following steps.

        Parameters
        ----------
        values : ParamInput
            One value per oscillator.
        """
        self.inputs = self.param_type_converter(values).reshape(self.n).clone()

    def _reset(self) -> None:
        """Start on the unit circle, at the set frequencies, with no input."""
        super()._reset()
        self.frequency = self.group("w").clone()
        self.inputs = torch.zeros(self.n)

    def _step(self, step: float) -> None:
        """Take one Euler step of ``step`` seconds."""
        rates = self._rates(self.frequency)
        rates[:, 0] += self.epsilon * self.inputs
        radius = torch.sqrt((self.xy**2).sum(dim=1)) + E
        frequency_rate = -self.epsilon * self.inputs * self.xy[:, 1] / radius
        self.xy += rates * step
        self.frequency += frequency_rate * step
