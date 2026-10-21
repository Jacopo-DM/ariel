"""Test: the MuJoCo worker base class and the example locomotion worker."""

from __future__ import annotations

import importlib
import os
import runpy
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from ariel.simulation.mujoco_worker import EvalConfig, MuJoCoWorkerBase

if TYPE_CHECKING:
    from types import ModuleType

    import mujoco

# The example lives in examples/; it is imported by its module name (not
# loaded from a path) so that the spawned pool can unpickle its worker.
EXAMPLES = Path(__file__).resolve().parents[3] / "examples"

ROBOT_XML = """
<mujoco>
    <worldbody>
    <body name="core" pos="0 0 0.1">
        <joint type="free"/>
        <geom type="box" size="0.1 0.1 0.05" mass="1"/>
        <body name="leg" pos="0.1 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.1"/>
        </body>
    </body>
    </worldbody>
    <actuator><motor joint="hinge" gear="10"/></actuator>
</mujoco>
"""
SPAWN = (0.0, 0.0, 0.1)
TARGET = (2.0, 0.0, 0.1)
SEED = 3
FITNESS_FLOOR = 0.0


class _CountingWorker(MuJoCoWorkerBase[EvalConfig]):
    """Reports the number of actuators and the seed of its config."""

    offset = 0.0

    def evaluate(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: EvalConfig,
    ) -> float:
        del data
        return float(model.nu * 100 + (config.seed or 0)) + self.offset


def test_worker_loads_the_model_and_calls_evaluate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """__call__ builds the model from XML and hands it to evaluate()."""
    # The worker sets these in this process; monkeypatch restores them.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    config = EvalConfig(spawn_position=SPAWN, target_position=TARGET, seed=SEED)
    assert _CountingWorker()((ROBOT_XML, config)) == pytest.approx(103.0)
    assert os.environ["OMP_NUM_THREADS"] == "1"


def test_eval_config_is_frozen_and_strict() -> None:
    """EvalConfig rejects unknown fields and cannot be changed."""
    config = EvalConfig(spawn_position=SPAWN, target_position=TARGET)
    with pytest.raises(ValueError, match="frozen"):
        config.seed = 1  # type: ignore[misc]  # frozen on purpose
    with pytest.raises(ValueError, match="extra"):
        EvalConfig(spawn_position=SPAWN, target_position=TARGET, speed=1)  # type: ignore[call-arg]  # deliberately wrong


@pytest.fixture
def example(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import ``examples/parallel_robot_eval.py`` by its module name.

    Returns
    -------
    ModuleType
        The example module.
    """
    monkeypatch.syspath_prepend(str(EXAMPLES))
    return importlib.import_module("parallel_robot_eval")


def _tiny_config(example: ModuleType) -> EvalConfig:
    config: EvalConfig = example.LocomotionConfig(
        spawn_position=SPAWN,
        target_position=TARGET,
        seed=SEED,
        cma_generations=1,
        cma_pop_size=2,
        hidden_size=2,
    )
    return config


def test_locomotion_worker_returns_a_distance(example: ModuleType) -> None:
    """The example worker optimises briefly and returns a fitness."""
    worker = example.TargetedLocomotionWorker()
    fitness = worker((ROBOT_XML, _tiny_config(example)))
    assert fitness >= FITNESS_FLOOR


def test_evaluate_population_uses_a_process_pool(example: ModuleType) -> None:
    """Two evaluations in a spawned pool give two fitness values."""
    args: list[tuple[str, EvalConfig]] = [
        (ROBOT_XML, _tiny_config(example)),
        (ROBOT_XML, _tiny_config(example)),
    ]
    results = example.evaluate_population(
        args, example.TargetedLocomotionWorker(), 1
    )
    assert len(results) == len(args)


def test_example_script_runs(capsys: pytest.CaptureFixture[str]) -> None:
    """The example's own demo prints the fitness values."""
    runpy.run_path(
        str(EXAMPLES / "parallel_robot_eval.py"), run_name="__main__"
    )
    assert "Fitness values" in capsys.readouterr().out
