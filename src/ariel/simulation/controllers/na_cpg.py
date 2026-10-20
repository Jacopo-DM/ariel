"""
Normalized asymmetric central pattern generator (NA-CPG) as a PyTorch module.

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
    most ``CF(w_i)`` (equations 4 and 12). The paper fits ``CF`` to the largest
    second derivative seen at each angular frequency and does not publish the
    fit. By default ``CF(w) = CONSTRAINT_MARGIN * w^2 * dt``, the largest
    change of the derivative per step on the unit circle, with a margin for the
    asymmetry. A cubic fitted elsewhere can be given as ``coefficients``.

References
----------
    [1] https://www.sciencedirect.com/science/article/pii/S2667379722000353
"""

# Standard library
from pathlib import Path

# Third-party libraries
import numpy as np
import torch
from numpy.typing import NDArray
from rich.console import Console
from torch import nn

# Local libraries
from ariel.parameters import FrozenModel
from ariel.simulation.controllers.simple_cpg import (
    create_fully_connected_adjacency,
)

__all__ = ["BetaParams", "NaCPG", "create_fully_connected_adjacency"]

# Global constants
E = 1e-9
CONSTRAINT_MARGIN = 3.0  # covers ha up to about 0.4: 1 / (1 - ha)^2 = 2.8

# --- DATA SETUP ---
DATA = Path.cwd() / "__data__"

console = Console()


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


# Ranges of the adaptable parameters at initialisation (uniform betas)
PHASE_PARAMS = BetaParams(a=1, b=1, c=0, d=2 * np.pi)
W_PARAMS = BetaParams(a=1, b=1, c=0, d=4 * np.pi)  # up to 2 Hz
AMPLITUDE_PARAMS = BetaParams(a=1, b=1, c=0, d=np.pi / 2)
HA_PARAMS = BetaParams(a=1, b=1, c=0, d=0.5)
B_PARAMS = BetaParams(a=1, b=1, c=-0.5, d=0.5)


class NaCPG(nn.Module):
    """Normalized asymmetric CPG network (Tong et al., 2022)."""

    xy: torch.Tensor
    xy_dot_old: torch.Tensor
    angles: torch.Tensor

    def __init__(
        self,
        adjacency_dict: dict[int, list[int]],
        alpha: float = 30.0,
        dt: float = 0.01,
        angle_limits: tuple[float, float] | None = (
            -torch.pi / 2,
            torch.pi / 2,
        ),
        seed: int | None = None,
        *,
        coupling: float = 1.0,
        clipping: bool = True,
        angle_tracking: bool = False,
    ) -> None:
        """Initialize the NA-CPG network. Inherits from **torch.nn.Module**.

        Parameters
        ----------
        adjacency_dict : dict[int, list[int]]
            Node index to the indices of the nodes that drive it. The keys
            are ``0 .. n - 1``.
        alpha : float, optional
            Learning rate, the speed of convergence to the unit limit cycle.
            The paper uses 30 with ``dt = 0.01``.
        dt : float, optional
            Time step of the Euler update, in seconds.
        angle_limits : tuple[float, float] | None, optional
            Lower and upper bound of the output angles, in radians.
        seed : int | None, optional
            Seed of the random initial parameters.
        coupling : float, optional
            Coupling coefficient ``h`` between connected oscillators.
        clipping : bool, optional
            If True, the output angles are clipped to ``angle_limits``. The
            distance to the limits is recorded in ``clamping_error`` either
            way.
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

        self.angle_tracking = angle_tracking
        self.angle_history: list[list[float]] = []
        self.angle_limits = angle_limits
        self.clipping = clipping
        self.clamping_error = 0.0

        # Inherent parameters, fixed during learning
        self.alpha = alpha
        self.dt = dt
        self.coupling = coupling
        # Cubic (highest power first) for CF(w); None uses the default
        self.coefficients: tuple[float, float, float, float] | None = None

        # Adaptable parameters
        rng = np.random.default_rng(seed)
        self.phase = self._parameter(PHASE_PARAMS, rng)
        self.w = self._parameter(W_PARAMS, rng)
        self.amplitudes = self._parameter(AMPLITUDE_PARAMS, rng)
        self.ha = self._parameter(HA_PARAMS, rng)
        self.b = self._parameter(B_PARAMS, rng)
        self.parameter_groups = {
            "phase": self.phase,
            "w": self.w,
            "amplitudes": self.amplitudes,
            "ha": self.ha,
            "b": self.b,
        }
        self.num_of_parameters = sum(p.numel() for p in self.parameters())
        self.num_of_parameter_groups = len(self.parameter_groups)

        # Internal states (buffers, not learnable)
        self.register_buffer("xy", torch.zeros(self.n, 2))
        self.register_buffer("xy_dot_old", torch.zeros(self.n, 2))
        self.register_buffer("angles", torch.zeros(self.n))
        self.reset_state()

    def _parameter(
        self, params: BetaParams, rng: np.random.Generator
    ) -> nn.Parameter:
        """Sample one parameter group.

        Returns
        -------
        nn.Parameter
            The group, with no gradient.
        """
        return nn.Parameter(
            scaled_beta(params, self.n, rng), requires_grad=False
        )

    def param_type_converter(  # ruff: ignore[no-self-use]  # keeps the interface shared with the sibling classes
        self,
        params: list[float] | NDArray[np.float64] | torch.Tensor,
    ) -> torch.Tensor:
        """Convert input parameters to torch.Tensor if needed.

        Parameters
        ----------
        params : list[float] | NDArray[np.float64] | torch.Tensor
            Input parameters to convert.

        Returns
        -------
        torch.Tensor
            Converted parameters as a float32 torch.Tensor.
        """
        if isinstance(params, list):
            return torch.tensor(params, dtype=torch.float32)
        if isinstance(params, np.ndarray):
            return torch.from_numpy(params).float()
        return params

    def set_flat_params(
        self, params: list[float] | NDArray[np.float64] | torch.Tensor
    ) -> None:
        """Set all adaptable parameters from one flat vector.

        Parameters
        ----------
        params : list[float] | NDArray[np.float64] | torch.Tensor
            The groups in the order of ``parameter_groups``, one after another.

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
        pointer = 0
        for param in self.parameter_groups.values():
            num_param = param.numel()
            param.data = safe_params[pointer : pointer + num_param].view_as(
                param
            )
            pointer += num_param

    def set_param_with_dict(
        self,
        params: dict[str, list[float] | NDArray[np.float64] | torch.Tensor],
    ) -> None:
        """Set parameter groups from a dict of group names to vectors.

        Parameters
        ----------
        params : dict[str, list[float] | NDArray[np.float64] | torch.Tensor]
            Group names as keys, one value per oscillator.
        """
        for key, value in params.items():
            self.set_params_by_group(key, value)

    def set_params_by_group(
        self,
        group_name: str,
        params: list[float] | NDArray[np.float64] | torch.Tensor,
    ) -> None:
        """Set one parameter group.

        Parameters
        ----------
        group_name : str
            ``phase``, ``w``, ``amplitudes``, ``ha`` or ``b``.
        params : list[float] | NDArray[np.float64] | torch.Tensor
            One value per oscillator.

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
        if safe_params.numel() != param.numel():
            msg = (
                f"Parameter vector has incorrect size for group '{group_name}'."
            )
            raise ValueError(msg)
        param.data = safe_params.view_as(param)

    def get_flat_params(self) -> torch.Tensor:
        """Get all adaptable parameters as one flat tensor.

        Returns
        -------
        torch.Tensor
            The groups in the order of ``parameter_groups``.
        """
        return torch.cat([p.flatten() for p in self.parameter_groups.values()])

    def zeta(self) -> torch.Tensor:
        """Return the asymmetry factor of every oscillator (equation 3).

        Returns
        -------
        torch.Tensor
            ``1 - ha * sign(x_dot_old)``, never below ``E``.
        """
        sign = torch.sign(self.xy_dot_old[:, 0])
        return torch.clamp(1 - self.ha * sign, min=E)

    def constraint_function(self) -> torch.Tensor:
        """Return the largest allowed change of the derivative per step.

        Returns
        -------
        torch.Tensor
            ``CF(w)`` for every oscillator, never below zero.
        """
        if self.coefficients is None:
            return CONSTRAINT_MARGIN * self.w**2 * self.dt
        value = np.polyval(self.coefficients, self.w.numpy())
        return torch.clamp(torch.tensor(value, dtype=torch.float32), min=0.0)

    def set_state(
        self,
        xy: list[float] | NDArray[np.float64] | torch.Tensor,
        xy_dot_old: list[float] | NDArray[np.float64] | torch.Tensor,
    ) -> None:
        """Set the oscillator states and the previous derivatives.

        Parameters
        ----------
        xy : list[float] | NDArray[np.float64] | torch.Tensor
            The states ``(x_i, y_i)``, ``n * 2`` values.
        xy_dot_old : list[float] | NDArray[np.float64] | torch.Tensor
            The previous derivatives, ``n * 2`` values.
        """
        self.xy = self.param_type_converter(xy).reshape(self.n, 2).clone()
        self.xy_dot_old = (
            self.param_type_converter(xy_dot_old).reshape(self.n, 2).clone()
        )

    def reset_state(self) -> None:
        """Put every oscillator on its limit cycle at its phase, at rest.

        The state is ``(cos phi_i, sin phi_i + b_i)`` and the previous
        derivative is zero. Oscillators that start there are already phase
        locked, so the first steps only show the transient of the coupling.
        """
        self.xy = torch.stack(
            [torch.cos(self.phase), torch.sin(self.phase) + self.b], dim=1
        ).detach()
        self.xy_dot_old = torch.zeros(self.n, 2)
        self.angles = torch.zeros(self.n)
        self.angle_history = []

    def forward(self, time: float | None = None) -> torch.Tensor:
        """Advance every oscillator by one step and return the angles.

        Parameters
        ----------
        time : float | None, optional
            The simulation time. At 0 the state is reset first.

        Returns
        -------
        torch.Tensor
            The output angles ``A_i * y_i``, clipped to ``angle_limits`` if
            ``clipping`` is set.

        Raises
        ------
        ValueError
            If the angles contain NaN.
        """
        if time is not None and np.isclose(time, 0.0):
            self.reset_state()

        with torch.inference_mode():
            x = self.xy[:, 0]
            y = self.xy[:, 1] - self.b  # centred on the offset
            gain = self.alpha * (1 - (x**2 + y**2))
            rotation = self.w / self.zeta()

            # Coupling: R(phi_i - phi_j) (x_j, y_j - b_j), over the neighbours
            delta = self.phase[:, None] - self.phase[None, :]
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
            limit = self.constraint_function()[:, None]
            xy_dot = torch.clamp(
                xy_dot, self.xy_dot_old - limit, self.xy_dot_old + limit
            )

            self.xy += xy_dot * self.dt
            self.xy_dot_old = xy_dot
            angles = self.amplitudes * self.xy[:, 1]

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
            msg += f"{self.xy=}\n{self.xy_dot_old=}\n{self.ha=}\n{self.w=}\n"
            msg += f"{self.amplitudes=}\n{self.phase=}\n{self.b=}\n"
            raise ValueError(msg)

        self.angles = angles
        return self.angles.clone()

    def save(self, path: str | Path) -> None:
        """Save the adaptable parameters to a file.

        Parameters
        ----------
        path : str | Path
            File path to save the parameters.
        """
        path = Path(path)
        torch.save(
            {
                name: group.detach().cpu()
                for name, group in self.parameter_groups.items()
            },
            path,
        )
        console.log(f"[green]Saved parameters to {path}[/green]")

    def load(self, path: str | Path) -> None:
        """Load the adaptable parameters from a file.

        Parameters
        ----------
        path : str | Path
            File path to load the parameters from.
        """
        path = Path(path)
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        for name, group in self.parameter_groups.items():
            group.data = loaded[name]
        console.log(f"[green]Loaded parameters from {path}[/green]")


# Example usage
def main() -> None:
    """Show example usage of the NaCPG class."""
    DATA.mkdir(exist_ok=True)
    na_cpg = NaCPG(create_fully_connected_adjacency(3), angle_tracking=True)

    for _ in range(1000):
        na_cpg.forward()

    import matplotlib.pyplot as plt  # ruff: ignore[import-outside-top-level]

    hist = torch.tensor(na_cpg.angle_history)
    times = torch.arange(hist.shape[0]) * na_cpg.dt

    plt.figure(figsize=(8, 4))
    for j in range(hist.shape[1]):
        plt.plot(times, hist[:, j], label=f"joint {j}")
    plt.xlabel("time (s)")
    plt.ylabel("angle (rad)")
    plt.title("NA-CPG angle histories")
    plt.legend()
    plt.grid(visible=True)
    plt.tight_layout()
    plt.savefig(DATA / "na_cpg_angles.png")
    plt.show()

    na_cpg.save(DATA / "na_cpg_params.pt")


if __name__ == "__main__":
    main()
