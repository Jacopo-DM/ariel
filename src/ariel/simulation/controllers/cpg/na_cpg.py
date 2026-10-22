"""
Normalized asymmetric central pattern generator (NA-CPG).

The model follows Tong et al., "NA-CPG: A robust and stable rhythm generator
for robot motion control", Biomimetic Intelligence and Robotics 2 (2022)
100075 (equations 2 to 12). Each oscillator ``i`` has a state ``(x_i, y_i)``
and obeys::

    d(x_i, y_i)/dt = K_i (x_i, y_i - b_i)
                        + h * sum_j R(phi_i - phi_j) (x_j, y_j - b_j)

    K_i = [[alpha (1 - r_i^2), -w_i / zeta_i],
            [w_i / zeta_i,       alpha (1 - r_i^2)]]
    r_i^2 = x_i^2 + (y_i - b_i)^2
    zeta_i = 1 - ha_i * sign(dx_i/dt_old)
    theta_i = A_i * y_i

* Normalized: the limit cycle has radius 1 whatever ``A_i`` is, so the
    amplitude and the learning rate ``alpha`` are decoupled. The amplitude only
    scales the output ``theta_i``.
* Asymmetric: ``zeta_i`` slows the rotation in one half of the cycle and
    speeds it up in the other (``ha_i`` in (-1, 1), 0 is symmetric), without
    changing the shape of the cycle.
* Constrained: the new derivative may differ from the previous one by at
    most ``CF(w_i)`` per step (equations 4 and 12). The paper fits ``CF`` to the
    largest second derivative seen at each angular frequency and does not
    publish the fit. By default ``CF(w) = CONSTRAINT_MARGIN * w^2 * step``, the
    largest change of the derivative per step on the unit circle, with a
    margin for the asymmetry. A cubic fitted elsewhere can be given as
    ``coefficients``.

The ``offset`` input moves the centre ``b_i`` of the cycle, so the output
glides to the new offset; ``frequency`` scales ``w_i``, in the constraint too.

References
----------
    [1] https://www.sciencedirect.com/science/article/pii/S2667379722000353
"""

# Standard library
import math
from collections.abc import Sequence
from typing import ClassVar

# Third-party libraries
import numpy as np
import torch
from torch import nn

# Local libraries
from ariel.simulation.controllers.cpg._base import (
    CPG,
    BetaParams,
    E,
    InputRoute,
    ParamInput,
    uniform,
)

__all__ = ["NaCPG"]

# Global constants
CONSTRAINT_MARGIN = 3.0  # covers ha up to about 0.4: 1 / (1 - ha)^2 = 2.8


class NaCPG(CPG):
    """Normalized asymmetric CPG network (Tong et al., 2022)."""

    PARAMETER_RANGES: ClassVar[dict[str, BetaParams]] = {
        "phase": uniform(0, 2 * math.pi),
        "w": uniform(0, 4 * math.pi),  # up to 2 Hz
        "amplitudes": uniform(0, math.pi / 2),
        "ha": uniform(0, 0.5),
        "b": uniform(-0.5, 0.5),
    }

    phase: nn.Parameter
    w: nn.Parameter
    amplitudes: nn.Parameter
    ha: nn.Parameter
    b: nn.Parameter
    xy: torch.Tensor
    xy_dot_old: torch.Tensor

    def __init__(  # ruff: ignore[too-many-arguments]  # the shared CPG arguments plus the model's constants
        self,
        adjacency_dict: dict[int, list[int]],
        alpha: float = 30.0,
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
        """Initialize the NA-CPG network.

        Parameters
        ----------
        adjacency_dict : dict[int, list[int]]
            Node index to the indices of the nodes that drive it.
        alpha : float, optional
            Learning rate, the speed of convergence to the unit limit cycle.
            The paper uses 30 with ``dt = 0.01``.
        dt : float, optional
            Largest Euler step, in seconds.
        angle_limits : tuple[float, float] | None, optional
            Lower and upper bound of the output angles, in radians.
        seed : int | None, optional
            Seed of the random initial parameters.
        coupling : float, optional
            Coupling coefficient ``h`` between connected oscillators.
        clipping : bool, optional
            If True, the output angles are clipped to ``angle_limits``.
        angle_tracking : bool, optional
            If True, every output is appended to ``angle_history``.
        inputs : Sequence[InputRoute] | None, optional
            Where each entry of the vector given to ``set_input`` goes.
            None gives no input.
        """
        super().__init__(
            adjacency_dict,
            dt,
            angle_limits,
            seed,
            coupling=coupling,
            clipping=clipping,
            angle_tracking=angle_tracking,
            inputs=inputs,
        )
        self.alpha = alpha
        # Cubic (highest power first) for CF(w); None uses the default
        self.coefficients: tuple[float, float, float, float] | None = None
        self.reset_state()

    def zeta(self) -> torch.Tensor:
        """Return the asymmetry factor of every oscillator (equation 3).

        Returns
        -------
        torch.Tensor
            ``1 - ha * sign(x_dot_old)``, never below ``E``.
        """
        sign = torch.sign(self.xy_dot_old[:, 0])
        return torch.clamp(1 - self.group("ha") * sign, min=E)

    def constraint_function(self, step: float) -> torch.Tensor:
        """Return the largest allowed change of the derivative in one step.

        Parameters
        ----------
        step : float
            The length of the step, in seconds.

        Returns
        -------
        torch.Tensor
            ``CF(w)`` for every oscillator, never below zero.
        """
        w = self.group("w") * self.frequency_factor()
        if self.coefficients is None:
            return CONSTRAINT_MARGIN * w**2 * step
        value = np.polyval(self.coefficients, w.numpy())
        return torch.clamp(torch.tensor(value, dtype=torch.float32), min=0.0)

    def set_state(self, xy: ParamInput, xy_dot_old: ParamInput) -> None:
        """Set the oscillator states and the previous derivatives.

        Parameters
        ----------
        xy : ParamInput
            The states ``(x_i, y_i)``, ``n * 2`` values.
        xy_dot_old : ParamInput
            The previous derivatives, ``n * 2`` values.
        """
        self.xy = self.param_type_converter(xy).reshape(self.n, 2).clone()
        self.xy_dot_old = (
            self.param_type_converter(xy_dot_old).reshape(self.n, 2).clone()
        )

    def _reset(self) -> None:
        """Put every oscillator on its limit cycle at its phase, at rest.

        The state is ``(cos phi_i, sin phi_i + b_i)`` and the previous
        derivative is zero. Oscillators that start there are already phase
        locked, so the first steps only show the transient of the coupling.
        """
        phase = self.group("phase")
        self.xy = torch.stack(
            [torch.cos(phase), torch.sin(phase) + self.group("b")], dim=1
        )
        self.xy_dot_old = torch.zeros(self.n, 2)

    def _step(self, step: float) -> None:
        """Take one constrained Euler step of ``step`` seconds."""
        x = self.xy[:, 0]
        centre = self.group("b") + self.handle("offset")
        y = self.xy[:, 1] - centre  # centred on the offset
        gain = self.alpha * (1 - (x**2 + y**2))
        rotation = self.group("w") * self.frequency_factor() / self.zeta()

        # Coupling: R(phi_i - phi_j) (x_j, y_j - b_j), over the neighbours
        delta = self.phase_offsets()
        cos_d = torch.cos(delta) * self.adjacency
        sin_d = torch.sin(delta) * self.adjacency
        coupled_x = (cos_d * x[None, :] - sin_d * y[None, :]).sum(dim=1)
        coupled_y = (sin_d * x[None, :] + cos_d * y[None, :]).sum(dim=1)

        xy_dot = torch.stack(
            [
                gain * x - rotation * y + self.coupling * coupled_x,
                rotation * x + gain * y + self.coupling * coupled_y,
            ],
            dim=1,
        )

        # Constraint function (equations 4 and 12)
        limit = self.constraint_function(step)[:, None]
        xy_dot = torch.clamp(
            xy_dot, self.xy_dot_old - limit, self.xy_dot_old + limit
        )

        self.xy += xy_dot * step
        self.xy_dot_old = xy_dot

    def _angles(self) -> torch.Tensor:
        """Return ``A_i * y_i``, with the amplitude input.

        Returns
        -------
        torch.Tensor
            One angle per oscillator.
        """
        return (
            self.group("amplitudes") * self.amplitude_factor() * self.xy[:, 1]
        )
