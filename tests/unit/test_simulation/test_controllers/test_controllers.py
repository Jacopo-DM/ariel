"""Test: Controller class initialization and set_control logic."""

import mujoco
import numpy as np
import pytest

from ariel.body_phenotypes.robogen_lite.prebuilt_robots.spider import spider
from ariel.simulation.controllers.controller import Controller
from ariel.utils.tracker import Tracker

DEFAULT_CTRL_STEP = 50
DEFAULT_SAVE_STEP = 500
CUSTOM_CTRL_STEP = 10
CUSTOM_SAVE_STEP = 100


def _zero_callback(
    model: mujoco.MjModel,
    _data: mujoco.MjData,
    /,
    *_args: object,
    **_kwargs: object,
) -> list[float]:
    """Control callback that always outputs zeros.

    Returns
    -------
    list[float]
        A list of zeros.
    """
    return [0.0] * model.nu


def _build_spider_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    core = spider()
    model = core.spec.compile()
    data = mujoco.MjData(model)
    return model, data


def test_controller_default_initialization() -> None:
    """Controller initializes with expected defaults."""
    ctrl = Controller(controller_callback_function=_zero_callback)
    assert ctrl.time_steps_per_ctrl_step == DEFAULT_CTRL_STEP
    assert ctrl.time_steps_per_save == DEFAULT_SAVE_STEP
    assert ctrl.alpha == pytest.approx(0.5)
    assert isinstance(ctrl.tracker, Tracker)


def test_controller_custom_params() -> None:
    """Controller stores custom parameters correctly."""
    ctrl = Controller(
        controller_callback_function=_zero_callback,
        time_steps_per_ctrl_step=CUSTOM_CTRL_STEP,
        time_steps_per_save=CUSTOM_SAVE_STEP,
        alpha=0.8,
    )
    assert ctrl.time_steps_per_ctrl_step == CUSTOM_CTRL_STEP
    assert ctrl.time_steps_per_save == CUSTOM_SAVE_STEP
    assert ctrl.alpha == pytest.approx(0.8)


def test_controller_set_control_runs() -> None:
    """set_control does not raise when called on a valid model/data."""
    model, data = _build_spider_model()
    ctrl = Controller(
        controller_callback_function=_zero_callback,
        time_steps_per_ctrl_step=1,
    )
    mujoco.mj_step(model, data)
    ctrl.set_control(model, data)
    del model, data


def test_controller_set_control_clips_output() -> None:
    """set_control clips control values to [-pi/2, pi/2]."""

    def large_output(
        model: mujoco.MjModel,
        _data: mujoco.MjData,
        /,
        *_args: object,
        **_kwargs: object,
    ) -> list[float]:
        return [999.0] * model.nu

    model, data = _build_spider_model()
    ctrl = Controller(
        controller_callback_function=large_output,
        time_steps_per_ctrl_step=1,
    )
    mujoco.mj_step(model, data)
    ctrl.set_control(model, data)
    assert np.all(data.ctrl <= np.pi / 2 + 1e-9)
    assert np.all(data.ctrl >= -np.pi / 2 - 1e-9)
    del model, data


def test_controller_nan_output_raises() -> None:
    """set_control raises ValueError when the callback returns NaN."""

    def nan_callback(
        model: mujoco.MjModel,
        _data: mujoco.MjData,
        /,
        *_args: object,
        **_kwargs: object,
    ) -> list[float]:
        return [float("nan")] * model.nu

    model, data = _build_spider_model()
    ctrl = Controller(
        controller_callback_function=nan_callback,
        time_steps_per_ctrl_step=1,
        alpha=1.0,
    )
    mujoco.mj_step(model, data)
    with pytest.raises(ValueError, match="NaN"):
        ctrl.set_control(model, data)
    del model, data
