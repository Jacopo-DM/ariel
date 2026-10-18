"""Test: the Lynx evolution, replay and log-plotting scripts."""

from __future__ import annotations

import json
import runpy
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Self

import matplotlib as mpl
import mujoco
import mujoco.viewer
import numpy as np
import pytest

from ariel.body_phenotypes.lynx_mjspec.unified_pipeline import (
    common,
    evolve,
    plot_from_logs,
    replay,
)
from ariel.body_phenotypes.lynx_mjspec.unified_pipeline.common import (
    NUM_TUBES,
    FastNumpyNetwork,
    PolicySpec,
)

mpl.use("Agg")

if TYPE_CHECKING:
    from numpy.typing import NDArray

HIDDEN = 2
SIM_STEPS = 12
CTRL_FREQ = 4
HUGE = 100.0
NEVER = 0.0
HOLD_STEPS = 2
POPULATION = 2
GENERATIONS = 1
RUN_NAME = "20260101T000000Z"
LOG_LINES = (
    "gen=001 best=1.50000 mean=2.00000 std=0.1 worst=3.0 valid=2/2\n"
    "gen=002 best=1.25000 mean=1.80000 std=0.1 worst=2.5 valid=2/2\n"
)


def _genome(
    policy: PolicySpec, tubes: float = 0.5, weight: float = 0.0
) -> NDArray[np.float64]:
    return np.concatenate([
        np.full(NUM_TUBES, tubes),
        np.full(common.num_network_params(policy), weight),
    ])


def _evaluate(
    genome: NDArray[np.float64],
    policy: PolicySpec,
    *,
    touch_threshold: float = NEVER,
    hold_threshold: float = NEVER,
) -> float:
    return evolve.evaluate_candidate(
        genome,
        policy,
        common.DEFAULT_TARGET,
        SIM_STEPS,
        CTRL_FREQ,
        touch_threshold,
        hold_threshold,
        HOLD_STEPS,
        evolve.DEFAULT_TIME_BONUS_WEIGHT,
    )


# --- evolve ----------------------------------------------------------------- #


def test_make_run_name_is_a_utc_stamp() -> None:
    """The run name looks like 20260101T000000Z."""
    assert len(evolve.make_run_name()) == len(RUN_NAME)
    assert evolve.make_run_name().endswith("Z")


def test_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no arguments the documented defaults are returned."""
    monkeypatch.setattr(sys, "argv", ["evolve"])
    args = evolve.parse_args()
    assert args.optimizer == "CMA"
    assert args.seed == 0


def test_evaluate_untouched_candidate_is_finite() -> None:
    """A zero policy stays put: finite fitness, no touch bonus."""
    policy = PolicySpec(hidden_size=HIDDEN)
    fitness = _evaluate(_genome(policy), policy)
    assert np.isfinite(fitness)
    assert fitness < evolve.INVALID_FITNESS


def test_evaluate_touching_candidate_stops_early() -> None:
    """A huge threshold counts as touched and holds, so the run ends early."""
    policy = PolicySpec(hidden_size=HIDDEN)
    far = _evaluate(_genome(policy), policy)
    near = _evaluate(
        _genome(policy), policy, touch_threshold=HUGE, hold_threshold=HUGE
    )
    assert near < far


def test_evaluate_penalises_tubes_out_of_bounds() -> None:
    """Tube lengths outside the bounds add a penalty."""
    policy = PolicySpec(hidden_size=HIDDEN)
    inside = _evaluate(_genome(policy, tubes=common.TUBE_MAX), policy)
    outside = _evaluate(_genome(policy, tubes=common.TUBE_MAX + 1.0), policy)
    assert outside > inside


def test_evaluate_wrong_genome_size_is_invalid() -> None:
    """A genome of the wrong size gets the invalid-fitness value."""
    policy = PolicySpec(hidden_size=HIDDEN)
    assert _evaluate(np.zeros(1), policy) == pytest.approx(
        evolve.INVALID_FITNESS
    )


def test_evaluate_diverging_simulation_is_ranked_after_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A simulation that produces NaN returns the minimum distance plus 1."""

    def blow_up(_model: mujoco.MjModel, data: mujoco.MjData) -> None:
        data.qpos[0] = np.nan

    monkeypatch.setattr(mujoco, "mj_step", blow_up)
    policy = PolicySpec(hidden_size=HIDDEN)
    fitness = _evaluate(_genome(policy), policy)
    assert fitness > 1.0
    assert fitness < evolve.INVALID_FITNESS


def _run_evolve(
    monkeypatch: pytest.MonkeyPatch, out_dir: Path, optimizer: str
) -> Path:
    # Threads run in this process, so coverage sees the work.
    monkeypatch.setattr(evolve, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evolve",
            "--generations", str(GENERATIONS),
            "--population", str(POPULATION),
            "--workers", "1",
            "--hidden-size", str(HIDDEN),
            "--sim-steps", str(SIM_STEPS),
            "--ctrl-freq", str(CTRL_FREQ),
            "--optimizer", optimizer,
            "--out-dir", str(out_dir),
        ],
    )  # fmt: skip
    evolve.main()
    (run,) = [p for p in out_dir.iterdir() if p.is_dir()]
    return run


@pytest.mark.filterwarnings(
    "ignore::nevergrad.common.errors.NevergradRuntimeWarning"
)
@pytest.mark.parametrize("optimizer", ["CMA", "TBPSA"])
def test_evolve_main_saves_the_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    optimizer: str,
) -> None:
    """A tiny run writes the genome, tubes, weights and metadata."""
    run = _run_evolve(monkeypatch, tmp_path / "out", optimizer)
    names = {p.name for p in run.iterdir()}
    assert names == {
        "best_genome.npy",
        "best_tube_lengths.npy",
        "best_brain_weights.npy",
        "metadata.json",
    }
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["policy"]["hidden_size"] == HIDDEN
    assert "gen=001" in capsys.readouterr().out


# --- replay ----------------------------------------------------------------- #


def _artifacts(folder: Path) -> Path:
    # Write a run folder the way evolve does and return it.
    policy = PolicySpec(hidden_size=HIDDEN)
    run = folder / RUN_NAME
    run.mkdir(parents=True)
    np.save(run / "best_tube_lengths.npy", np.full(NUM_TUBES, 0.5))
    np.save(
        run / "best_brain_weights.npy",
        np.zeros(common.num_network_params(policy)),
    )
    metadata = {
        "policy": policy.model_dump(),
        "target": common.DEFAULT_TARGET.tolist(),
        "touch_threshold": HUGE,
    }
    (run / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return run


def test_resolve_artifact_dir_finds_the_run(tmp_path: Path) -> None:
    """A parent folder resolves to the newest run inside it."""
    run = _artifacts(tmp_path)
    assert replay.resolve_artifact_dir(tmp_path) == run
    assert replay.resolve_artifact_dir(run) == run


def test_resolve_artifact_dir_reports_an_empty_folder(tmp_path: Path) -> None:
    """A folder without any run is an error."""
    with pytest.raises(FileNotFoundError, match=r"metadata\.json"):
        replay.resolve_artifact_dir(tmp_path)


def test_load_artifacts_returns_policy_and_arrays(tmp_path: Path) -> None:
    """Weights, tubes, the policy spec and the metadata come back."""
    run = _artifacts(tmp_path)
    tubes, brain, policy, metadata = replay.load_artifacts(run)
    assert tubes.shape == (NUM_TUBES,)
    assert brain.shape == (common.num_network_params(policy),)
    assert policy.hidden_size == HIDDEN
    assert "target" in metadata


def test_load_artifacts_needs_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A run folder whose metadata.json vanished is an error."""
    monkeypatch.setattr(replay, "resolve_artifact_dir", lambda path: path)
    with pytest.raises(FileNotFoundError, match="Missing metadata"):
        replay.load_artifacts(tmp_path)


def test_run_rollout_locks_controls_after_a_touch() -> None:
    """With a huge threshold the first step counts as the touch."""
    policy = PolicySpec(hidden_size=HIDDEN)
    model, data, tcp, tgt, joints = common.build_model(np.full(NUM_TUBES, 0.5))
    net = FastNumpyNetwork(
        policy.input_size,
        policy.hidden_size,
        policy.output_size,
        np.zeros(common.num_network_params(policy)),
    )
    initial, low, final, touched = replay.run_rollout(
        model, data, net, policy, tcp, tgt, joints,
        common.DEFAULT_TARGET, SIM_STEPS, CTRL_FREQ, HUGE,
    )  # fmt: skip
    assert low <= initial
    assert final >= 0.0
    assert touched is not None
    del model, data


def _replay_argv(
    monkeypatch: pytest.MonkeyPatch, folder: Path, *extra: str
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "replay",
            "--artifact-dir", str(folder),
            "--sim-steps", str(SIM_STEPS),
            "--ctrl-freq", str(CTRL_FREQ),
            *extra,
        ],
    )  # fmt: skip


def test_replay_eval_only_prints_distances(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--eval-only prints the four numbers and returns."""
    _artifacts(tmp_path)
    _replay_argv(monkeypatch, tmp_path, "--eval-only")
    replay.main()
    out = capsys.readouterr().out
    assert "initial_distance=" in out
    assert "touch_time=" in out


def test_replay_eval_only_with_a_custom_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A target other than the default replaces the stored one."""
    _artifacts(tmp_path)
    _replay_argv(monkeypatch, tmp_path, "--eval-only", "--target-x", "0.3")
    replay.main()
    assert "final_distance=" in capsys.readouterr().out


class _FakeViewer:
    """Stands in for the passive viewer: runs a few frames."""

    def __init__(self, frames: int) -> None:
        self.frames = frames
        self.syncs = 0
        self.opt = _FakeOptions()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def is_running(self) -> bool:
        self.frames -= 1
        return self.frames >= 0

    def sync(self) -> None:
        self.syncs += 1


class _FakeOptions:
    def __init__(self) -> None:
        self.flags: dict[object, bool] = {}
        self.sitegroup = np.zeros(6, dtype=np.int32)


class _NoSiteFlag:
    """An mjtVisFlag without mjVIS_SITE."""


_SiteFlag = type("_SiteFlag", (), {"mjVIS_SITE": 1})


@pytest.mark.parametrize("has_site_flag", [True, False])
def test_replay_viewer_loop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    has_site_flag: bool,
) -> None:
    """The viewer loop steps the model and syncs the viewer."""
    _artifacts(tmp_path)
    _replay_argv(monkeypatch, tmp_path)
    viewer = _FakeViewer(frames=SIM_STEPS)
    monkeypatch.setattr(
        mujoco.viewer, "launch_passive", lambda _model, _data: viewer
    )
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        mujoco, "mjtVisFlag", _SiteFlag if has_site_flag else _NoSiteFlag
    )
    replay.main()
    assert viewer.syncs == SIM_STEPS


# --- plot_from_logs --------------------------------------------------------- #


def test_plot_training_logs_writes_a_figure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Valid logs give an aggregated plot; bad ones are reported."""
    monkeypatch.chdir(tmp_path)
    good = tmp_path / "good.log"
    good.write_text(LOG_LINES, encoding="utf-8")
    empty = tmp_path / "empty.log"
    empty.write_text("nothing here\n", encoding="utf-8")
    plot_from_logs.plot_training_logs([
        str(good),
        str(empty),
        str(tmp_path / "missing.log"),
    ])
    out = capsys.readouterr().out
    assert "No valid data" in out
    assert "not found" in out
    assert (tmp_path / "lynx_arm_experiment_10x.png").exists()


def test_plot_training_logs_accepts_a_single_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A plain string is treated as a single log file."""
    monkeypatch.chdir(tmp_path)
    good = tmp_path / "good.log"
    good.write_text(LOG_LINES, encoding="utf-8")
    plot_from_logs.plot_training_logs(str(good))
    assert (tmp_path / "lynx_arm_experiment_10x.png").exists()


def test_plot_training_logs_without_data(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No usable log means nothing to plot."""
    plot_from_logs.plot_training_logs([])
    assert "No data to plot" in capsys.readouterr().out


def test_plot_main_block_without_logs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The script reports when the batch folder holds no train.log."""
    (tmp_path / "__data__/lynx_mjspec/unified_batch_10x").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(Path(plot_from_logs.__file__)), run_name="__main__")
    assert "No train.log files found" in capsys.readouterr().out


def test_plot_main_block_plots_the_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The script plots every run folder that has a train.log."""
    run = tmp_path / "__data__/lynx_mjspec/unified_batch_10x/run1"
    run.mkdir(parents=True)
    (run / "train.log").write_text(LOG_LINES, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(Path(plot_from_logs.__file__)), run_name="__main__")
    assert "Plotting 1 runs" in capsys.readouterr().out
    assert (tmp_path / "lynx_arm_experiment_10x.png").exists()
