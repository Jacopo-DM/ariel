"""
Phase oscillators steered by touch: Tegotae control, fixed and learnable.

Tegotae (Owaki et al.) is a Japanese word for the sense that an action got the
response it expected. Each joint ``i`` has a phase ``phi_i`` and expects its
limb to bear load in one half of the cycle, where ``C(phi) = -sin(phi) > 0``.
The Tegotae function compares that expectation with the load ``N_i`` the limb
feels, and the phase moves to raise it (Herneth, Hayashibe and Owaki, ICRA
2023, equations 1, 2 and 5)::

    T_i         = C(phi_i) S(N_i)
    dphi_i/dt   = w_i + sigma_i dT_i/dphi_i = w_i - sigma_i N_i cos(phi_i)
    angle_i     = offset_i + A_i cos(phi_i)

A limb that already bears load slows its phase in the stance half and pushes
it out of the swing half, so the limbs settle into a rhythm that fits the
body, with no wiring between the oscillators (``coupling`` is 0 by default).
``A_i`` may be negative, which flips which half of the swing is the stance.

``LearnableTegotaeCPG`` follows the paper's generalisation (equations 3 to 5):
the derivative of the action term, the reaction term and the gain become
small networks, shared by every joint and tuned with the other parameters::

    dphi_i/dt = w_i + G(cos phi_i, sin phi_i, N_i) A'(cos phi_i, sin phi_i)
                      (R(N_i) - R(0))

Each network has one hidden layer of ``hidden`` tanh units. ``R(0)`` is
subtracted so that a limb that touches nothing gets no feedback, as in the
fixed rule; the paper does not say how it handles that case.

The load comes from outside: call ``set_input`` with one value per joint
(for example the vertical ground force on the bodies the joint moves, in
newtons) before ``forward``. It is held until the next call.

References
----------
    [1] C. Herneth, M. Hayashibe and D. Owaki, "Learnable Tegotae-based
        feedback in CPGs with sparse observation produces efficient and
        adaptive locomotion", ICRA 2023, 1155-1161.
    [2] D. Owaki and A. Ishiguro, "A quadruped robot exhibiting spontaneous
        gait transitions from walking to trotting to galloping", Scientific
        Reports 7 (2017) 277.
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
    ParamInput,
    uniform,
)

__all__ = ["LearnableTegotaeCPG", "TegotaeCPG"]


class TegotaeCPG(CPG):
    """Phase oscillators with the fixed Tegotae feedback of Owaki et al."""

    PARAMETER_RANGES: ClassVar[dict[str, BetaParams]] = {
        "phase": uniform(0, 2 * math.pi),
        "w": uniform(0, 4 * math.pi),  # up to 2 Hz
        "amplitudes": uniform(-math.pi / 2, math.pi / 2),
        "offset": uniform(-0.5, 0.5),
        "sigma": uniform(0, 2),  # rad/s per newton
    }

    phase: nn.Parameter
    w: nn.Parameter
    amplitudes: nn.Parameter
    offset: nn.Parameter
    theta: torch.Tensor
    loads: torch.Tensor

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
        coupling: float = 0.0,
        clipping: bool = True,
        angle_tracking: bool = False,
    ) -> None:
        """Initialize the network.

        Parameters
        ----------
        adjacency_dict : dict[int, list[int]]
            Node index to the indices of the nodes that drive it.
        dt : float, optional
            Largest Euler step, in seconds.
        angle_limits : tuple[float, float] | None, optional
            Lower and upper bound of the output angles, in radians.
        seed : int | None, optional
            Seed of the random initial parameters.
        coupling : float, optional
            Strength of a Kuramoto pull to the target phase differences, in
            rad/s per neighbour; 0, as in the papers, leaves the body alone
            to coordinate the joints.
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
        self.reset_state()

    def set_input(self, values: ParamInput) -> None:
        """Set the load ``N_i`` each joint feels, used by the next steps.

        Parameters
        ----------
        values : ParamInput
            One value per oscillator, 0 for a limb that touches nothing.
        """
        self.loads = self.param_type_converter(values).reshape(self.n).clone()

    def feedback(self) -> torch.Tensor:
        """Return the Tegotae feedback ``sigma_i dT_i/dphi_i``.

        Returns
        -------
        torch.Tensor
            ``-sigma_i N_i cos(phi_i)``, in rad/s.
        """
        return -self.group("sigma") * self.loads * torch.cos(self.theta)

    def _reset(self) -> None:
        """Start every phase at its target, with no load."""
        self.theta = self.group("phase").clone()
        self.loads = torch.zeros(self.n)

    def _step(self, step: float) -> None:
        """Take one Euler step of ``step`` seconds."""
        # error[i, j] = theta_j - theta_i - (phi_j - phi_i)
        error = (
            self.theta[None, :] - self.theta[:, None] - self.phase_offsets().T
        )
        pull = (torch.sin(error) * self.adjacency).sum(dim=1)
        rate = self.group("w") + self.coupling * pull + self.feedback()
        self.theta += rate * step

    def _angles(self) -> torch.Tensor:
        """Return ``offset_i + A_i cos(phi_i)``.

        Returns
        -------
        torch.Tensor
            One angle per oscillator.
        """
        return self.group("offset") + self.group("amplitudes") * torch.cos(
            self.theta
        )


def mlp(
    weights: torch.Tensor, inputs: torch.Tensor, hidden: int
) -> torch.Tensor:
    """Evaluate a network with one hidden tanh layer and one output.

    Parameters
    ----------
    weights : torch.Tensor
        The input weights, the hidden biases, the output weights and the
        output bias, flat, ``(k + 2) * hidden + 1`` values.
    inputs : torch.Tensor
        One row of ``k`` inputs per oscillator.
    hidden : int
        The number of hidden units.

    Returns
    -------
    torch.Tensor
        One output per row.
    """
    k = inputs.shape[1]
    first = weights[: k * hidden].reshape(k, hidden)
    bias = weights[k * hidden : (k + 1) * hidden]
    second = weights[(k + 1) * hidden : (k + 2) * hidden]
    return torch.tanh(inputs @ first + bias) @ second + weights[-1]


class LearnableTegotaeCPG(TegotaeCPG):
    """Tegotae control with learnable action, reaction and gain networks."""

    PARAMETER_RANGES: ClassVar[dict[str, BetaParams]] = {
        "phase": uniform(0, 2 * math.pi),
        "w": uniform(0, 4 * math.pi),  # up to 2 Hz
        "amplitudes": uniform(-math.pi / 2, math.pi / 2),
        "offset": uniform(-0.5, 0.5),
        "action": uniform(-1, 1),  # weights of A'(cos phi, sin phi)
        "reaction": uniform(-1, 1),  # weights of R(N)
        "gain": uniform(-1, 1),  # weights of G(cos phi, sin phi, N)
    }
    INPUTS: ClassVar[dict[str, int]] = {"action": 2, "reaction": 1, "gain": 3}

    def __init__(
        self,
        adjacency_dict: dict[int, list[int]],
        hidden: int = 4,
        dt: float = 0.01,
        angle_limits: tuple[float, float] | None = (
            -math.pi / 2,
            math.pi / 2,
        ),
        seed: int | None = None,
        *,
        coupling: float = 0.0,
        clipping: bool = True,
        angle_tracking: bool = False,
    ) -> None:
        """Initialize the network.

        Parameters
        ----------
        adjacency_dict : dict[int, list[int]]
            Node index to the indices of the nodes that drive it.
        hidden : int, optional
            Hidden units of each of the three networks. With 4, the three
            networks hold 51 weights.
        dt : float, optional
            Largest Euler step, in seconds.
        angle_limits : tuple[float, float] | None, optional
            Lower and upper bound of the output angles, in radians.
        seed : int | None, optional
            Seed of the random initial parameters.
        coupling : float, optional
            Strength of a Kuramoto pull to the target phase differences.
        clipping : bool, optional
            If True, the output angles are clipped to ``angle_limits``.
        angle_tracking : bool, optional
            If True, every output is appended to ``angle_history``.
        """
        self.hidden = hidden
        super().__init__(
            adjacency_dict,
            dt,
            angle_limits,
            seed,
            coupling=coupling,
            clipping=clipping,
            angle_tracking=angle_tracking,
        )

    def shared_group_sizes(self) -> dict[str, int]:
        """Return the weight count of each network.

        Returns
        -------
        dict[str, int]
            ``(inputs + 2) * hidden + 1`` per network.
        """
        return {
            name: (inputs + 2) * self.hidden + 1
            for name, inputs in self.INPUTS.items()
        }

    def feedback(self) -> torch.Tensor:
        """Return ``G A' (R(N) - R(0))`` for every joint.

        Returns
        -------
        torch.Tensor
            The feedback, in rad/s.
        """
        phase = torch.stack([torch.cos(self.theta), torch.sin(self.theta)], 1)
        loads = self.loads[:, None]
        action = mlp(self.group("action"), phase, self.hidden)
        reaction = mlp(self.group("reaction"), loads, self.hidden) - mlp(
            self.group("reaction"), torch.zeros_like(loads), self.hidden
        )
        gain = mlp(
            self.group("gain"), torch.cat([phase, loads], dim=1), self.hidden
        )
        return gain * action * reaction
