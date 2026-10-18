"""Controller that applies a callback to a MuJoCo simulation."""

# Standard library
from dataclasses import dataclass, field  # ruff: ignore[banned-api]  # behaviour first: wraps a callback and a tracker; the fields are its settings
from typing import Protocol

# Third-party libraries
import mujoco as mj
import numpy as np

# Local libraries
from ariel.utils.tracker import Tracker


class ControlCallback(Protocol):
    """A function that returns the control values for one control step."""

    def __call__(
        self,
        model: mj.MjModel,
        data: mj.MjData,
        /,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Return the control values for the current simulation state."""
        ...


@dataclass
class Controller:
    # Function that executes the actual control step
    """Apply a control callback to a MuJoCo simulation, optionally tracked."""

    controller_callback_function: ControlCallback

    # How often to call the controller (for every simulation step)
    time_steps_per_ctrl_step: int = 50  # control frequency

    # How often to save the data
    time_steps_per_save: int = 500  # data-sampling frequency

    # How big a step to take towards the output fot the callback function
    alpha: float = 0.5

    # Optional tracker to save data during simulation
    tracker: Tracker = field(default_factory=Tracker)

    def set_control(
        self,
        model: mj.MjModel,
        data: mj.MjData,
        *args: object,
        **kwargs: object,
    ) -> None:
        """Set the controller callback function and gives additional arguments.

        Parameters
        ----------
        model : mj.MjModel
            The Mujoco model of the simulation. This will be used to access
            model parameters, such as the number of actuators.
        data : mj.MjData
            The Mujoco data of the simulation. This will be used to access
            simulation variables, such as the current time and control values.
        *args : object
            Additional arguments to pass to the controller callback function.
        **kwargs : object
            Additional keyword arguments to pass to the controller callback
            function.

        Raises
        ------
        ValueError
            If the control signal contains NaN.
        """
        # Calculate current time step
        time = data.time
        time_step = model.opt.timestep
        deduced_time_step = np.ceil(time / time_step)

        # Execute saving only at specific time-steps
        if (deduced_time_step % self.time_steps_per_save) == 0:
            self.tracker.update(data)

        # Execute control strategy only at specific time-steps
        if (deduced_time_step % self.time_steps_per_ctrl_step) == 0:
            # Save the old control values
            old_ctrl = data.ctrl.copy()

            # Execute the custom control function of the user
            output = np.array(
                self.controller_callback_function(
                    model,
                    data,
                    *args,
                    **kwargs,
                ),
            )

            # Calculate the new control values
            new_ctrl = (old_ctrl * (1 - self.alpha)) + (output * self.alpha)

            # Ensure that the new control values are within the servo bounds
            data.ctrl = np.clip(new_ctrl, -np.pi / 2, np.pi / 2)

            # Check if there are any NaN values in the control signal
            if np.any(np.isnan(data.ctrl)):
                msg = "NaN values detected in the control signal.\n"
                msg += f"{data.ctrl}"
                raise ValueError(msg)
