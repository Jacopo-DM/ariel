"""
Hopf oscillators with separate swing and stance frequencies.

The oscillator of Righetti and Ijspeert, "Pattern generators with sensory
feedback for the control of quadruped locomotion" (ICRA 2008, equations 1 to
5)::

    dx_i/dt = alpha (1 - r_i^2) x_i - v_i y_i
    dy_i/dt = beta (1 - r_i^2) y_i + v_i x_i + sum_j k_ij y_j
    v_i     = w_stance_i / (exp(-b y_i) + 1) + w_swing_i / (exp(b y_i) + 1)
    angle_i = offset_i + A_i * x_i

The oscillator turns at ``w_stance`` while ``y > 0`` (stance) and at
``w_swing`` while ``y < 0`` (swing), so one cycle lasts
``pi / w_stance + pi / w_swing`` and a leg can, for example, lift fast and push
slowly. The shape of the cycle does not depend on the two frequencies.

The paper picks each coupling ``k_ij`` as +1 or -1 to build one quadruped
gait. Here the network has any number of joints, so the coupling comes from
the target phases: ``k_ij = h cos(phi_i - phi_j)``. Two joints with the same
target phase pull each other into phase (+h), two half a cycle apart push
each other into anti-phase (-h), and that reproduces the paper's matrices for
walk, trot, pace and bound. A difference of a quarter cycle gives no coupling,
so intermediate phase differences are only loosely held. The paper's touch
sensor feedback is left out: ARIEL's modules have no foot sensors.

References
----------
    [1] L. Righetti and A. J. Ijspeert, "Pattern generators with sensory
        feedback for the control of quadruped locomotion", ICRA 2008.
"""

# Standard library
import math
from typing import ClassVar

# Third-party libraries
import torch
from torch import nn

# Local libraries
from ariel.simulation.controllers.cpg._base import CPG, BetaParams, uniform

__all__ = ["SwingStanceCPG"]


class SwingStanceCPG(CPG):
    """Hopf oscillators with one frequency per half cycle."""

    PARAMETER_RANGES: ClassVar[dict[str, BetaParams]] = {
        "phase": uniform(0, 2 * math.pi),
        "w_swing": uniform(0, 4 * math.pi),
        "w_stance": uniform(0, 4 * math.pi),
        "amplitudes": uniform(0, math.pi / 2),
        "offset": uniform(-0.5, 0.5),
    }

    phase: nn.Parameter
    w_swing: nn.Parameter
    w_stance: nn.Parameter
    amplitudes: nn.Parameter
    offset: nn.Parameter
    xy: torch.Tensor

    def __init__(  # ruff: ignore[too-many-arguments]  # one argument per constant of the paper's equations
        self,
        adjacency_dict: dict[int, list[int]],
        alpha: float = 5.0,
        beta: float = 50.0,
        dt: float = 0.01,
        angle_limits: tuple[float, float] | None = (
            -math.pi / 2,
            math.pi / 2,
        ),
        seed: int | None = None,
        *,
        steepness: float = 100.0,
        coupling: float = 1.0,
        clipping: bool = True,
        angle_tracking: bool = False,
    ) -> None:
        """Initialize the network.

        Parameters
        ----------
        adjacency_dict : dict[int, list[int]]
            Node index to the indices of the nodes that drive it.
        alpha : float, optional
            Speed of convergence of ``x`` to the limit cycle, in 1/s. The
            paper uses 5.
        beta : float, optional
            Speed of convergence of ``y``, in 1/s. The paper uses 50.
        dt : float, optional
            Largest Euler step, in seconds.
        angle_limits : tuple[float, float] | None, optional
            Lower and upper bound of the output angles, in radians.
        seed : int | None, optional
            Seed of the random initial parameters.
        steepness : float, optional
            The constant ``b`` of the switch between the two frequencies.
        coupling : float, optional
            Size ``h`` of every coupling ``k_ij``, in 1/s.
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
        self.alpha = alpha
        self.beta = beta
        self.steepness = steepness
        self.reset_state()

    def frequency(self) -> torch.Tensor:
        """Return the rotation speed of each oscillator in its current half.

        Returns
        -------
        torch.Tensor
            ``w_stance`` for ``y`` well above 0, ``w_swing`` well below.
        """
        y = self.xy[:, 1]
        stance = torch.sigmoid(self.steepness * y)  # 1 / (exp(-b y) + 1)
        swing = torch.sigmoid(-self.steepness * y)  # 1 / (exp(b y) + 1)
        return self.group("w_stance") * stance + self.group("w_swing") * swing

    def _reset(self) -> None:
        """Put every oscillator on the unit circle at its target phase."""
        phase = self.group("phase")
        self.xy = torch.stack([torch.cos(phase), torch.sin(phase)], dim=1)

    def _step(self, step: float) -> None:
        """Take one Euler step of ``step`` seconds."""
        x = self.xy[:, 0]
        y = self.xy[:, 1]
        radial = 1 - (x**2 + y**2)
        rotation = self.frequency()
        weights = self.coupling * torch.cos(self.phase_offsets())
        coupled = (weights * self.adjacency * y[None, :]).sum(dim=1)
        rates = torch.stack(
            [
                self.alpha * radial * x - rotation * y,
                self.beta * radial * y + rotation * x + coupled,
            ],
            dim=1,
        )
        self.xy += rates * step

    def _angles(self) -> torch.Tensor:
        """Return ``offset_i + A_i * x_i``.

        Returns
        -------
        torch.Tensor
            One angle per oscillator.
        """
        return self.group("offset") + self.group("amplitudes") * self.xy[:, 0]
