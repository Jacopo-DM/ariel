"""
Amplitude-controlled phase oscillators.

This is the CPG of Ijspeert et al.'s salamander robot (Science, 2007), reviewed
in Ijspeert (Neural Networks, 2008) and used by Bellegarda and Ijspeert's
CPG-RL (2022) and by Raj and Kos (2026, equations 1 and 2). Each oscillator
``i`` has a phase ``theta_i`` and an amplitude ``r_i``::

    dtheta_i/dt = w_i + h * sum_j r_j sin(theta_j - theta_i - psi_ij)
    d2r_i/dt2   = a (a / 4 (R_i - r_i) - dr_i/dt)
    angle_i     = offset_i + r_i cos(theta_i)

with ``psi_ij = phi_j - phi_i``, so locked oscillators keep
``theta_i - theta_j = phi_i - phi_j``. The amplitude follows a critically
damped second-order equation, ``r = R (1 - exp(-a t / 2) (1 + a t / 2))``: it
starts at zero, reaches 96 % of ``R_i`` after ``10 / a`` seconds (0.5 s at the
default ``a = 20``) and follows any later change of ``R_i`` without a jump. The
phase has no amplitude of its own, so it cannot be knocked off a limit cycle;
only the coupling moves it.

References
----------
    [1] A. J. Ijspeert, A. Crespi, D. Ryczko and J.-M. Cabelguen, "From
        swimming to walking with a salamander robot driven by a spinal cord
        model", Science 315 (2007) 1416-1420.
    [2] A. J. Ijspeert, "Central pattern generators for locomotion control in
        animals and robots: a review", Neural Networks 21 (2008) 642-653.
    [3] G. Bellegarda and A. J. Ijspeert, "CPG-RL: Learning central pattern
        generators for quadruped locomotion", IEEE RA-L 7 (2022) 12547-12554.
"""

# Standard library
import math
from collections.abc import Sequence
from typing import ClassVar

# Third-party libraries
import torch
from torch import nn

# Local libraries
from ariel.simulation.controllers.cpg._base import (
    CPG,
    BetaParams,
    InputRoute,
    uniform,
)

__all__ = ["PhaseCPG"]


class PhaseCPG(CPG):
    """Phase oscillators with critically damped amplitudes."""

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
    theta: torch.Tensor
    r: torch.Tensor
    r_dot: torch.Tensor

    def __init__(  # ruff: ignore[too-many-arguments]  # the shared CPG arguments plus the model's constants
        self,
        adjacency_dict: dict[int, list[int]],
        convergence: float = 20.0,
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
        """Initialize the network.

        Parameters
        ----------
        adjacency_dict : dict[int, list[int]]
            Node index to the indices of the nodes that drive it.
        convergence : float, optional
            The constant ``a`` of the amplitude equation, in 1/s.
        dt : float, optional
            Largest integration step, in seconds.
        angle_limits : tuple[float, float] | None, optional
            Lower and upper bound of the output angles, in radians.
        seed : int | None, optional
            Seed of the random initial parameters.
        coupling : float, optional
            Weight ``h`` of every connection, in rad/s per radian of the
            neighbour's amplitude.
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
        self.convergence = convergence
        self.reset_state()

    def _reset(self) -> None:
        """Start every phase at its target, with zero amplitude at rest."""
        self.theta = self.group("phase").clone()
        self.r = torch.zeros(self.n)
        self.r_dot = torch.zeros(self.n)

    def _step(self, step: float) -> None:
        """Take one semi-implicit Euler step of ``step`` seconds."""
        # error[i, j] = theta_j - theta_i - (phi_j - phi_i)
        error = (
            self.theta[None, :] - self.theta[:, None] - self.phase_offsets().T
        )
        pull = (self.r[None, :] * torch.sin(error) * self.adjacency).sum(dim=1)
        w = self.group("w") * self.frequency_factor()
        theta_dot = w + self.coupling * pull
        a = self.convergence
        amplitude = self.group("amplitudes") * self.amplitude_factor()
        r_ddot = a * (a / 4 * (amplitude - self.r) - self.r_dot)
        self.theta += theta_dot * step
        self.r_dot += r_ddot * step
        self.r += self.r_dot * step

    def _angles(self) -> torch.Tensor:
        """Return ``offset_i + r_i cos(theta_i)``, with the offset input.

        Returns
        -------
        torch.Tensor
            One angle per oscillator.
        """
        offset = self.group("offset") + self.handle("offset")
        return offset + self.r * torch.cos(self.theta)
