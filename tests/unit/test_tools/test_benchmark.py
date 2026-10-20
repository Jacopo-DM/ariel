"""Tests for the benchmark tool in ``tools/`` (tiny workloads, real code)."""

from __future__ import annotations

# Standard library
import importlib.util
import subprocess  # ruff: ignore[suspicious-subprocess-import]  # only builds fake results
from pathlib import Path
from typing import TYPE_CHECKING

# Third-party libraries
import pytest

# Local libraries
from ariel.ec import Individual, Population

if TYPE_CHECKING:
    from types import ModuleType

SCRIPT = Path(__file__).resolve().parents[3] / "tools" / "benchmark.py"
RUNS = 3
SAMPLE = '{"wall": 2.0, "cpu": 1.5, "rss": 100.0}'
TINY_POP = 4


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("benchmark", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _shrink(monkeypatch: pytest.MonkeyPatch, module: ModuleType) -> None:
    for name, value in {
        "EVO_POP": TINY_POP,
        "EVO_STEPS": 1,
        "EVO_MODULES": 6,
        "EVO_SIM_STEPS": 5,
        "SIM_STEPS": 5,
        "BODIES": 2,
        "BODY_MODULES": 6,
    }.items():
        monkeypatch.setattr(module, name, value)


def test_summarise_gives_median_and_spread() -> None:
    module = _load()
    samples = [
        module.Sample(wall=wall, cpu=1.0, rss=50.0) for wall in (3.0, 1.0, 2.0)
    ]
    cells = module.summarise("evolution", samples).split()
    expected = ["evolution", "3", "2.000", "1.000", "3.000", "1.000", "50.0"]
    assert cells == expected


def test_run_once_reports_time_and_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load()
    monkeypatch.setitem(module.WORKLOADS, "tiny", lambda: sum(range(10**5)))
    sample = module.run_once("tiny")
    assert sample.wall > 0
    assert sample.cpu > 0
    assert sample.rss > 0


def test_the_tests_workload_counts_the_child_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load()
    calls: list[list[str]] = []

    def fake(command: list[str], **_kwargs: object) -> object:
        calls.append(command)
        return subprocess.CompletedProcess(args=command, returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake)
    sample = module.run_once("tests")
    assert calls[0][:3] == ["uv", "run", "--no-sync"]
    assert "pytest" in calls[0]
    assert sample.wall >= 0


def test_the_cheap_workloads_run(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load()
    _shrink(monkeypatch, module)
    module.simulation()
    module.bodies()


def test_the_evolution_workload_runs_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load()
    _shrink(monkeypatch, module)
    module.evolution()


def test_selection_keeps_the_fittest_alive_and_marks_the_better_half() -> None:
    module = _load()
    individuals = []
    for fitness in (1.0, 4.0, 2.0, 3.0):
        individual = Individual()
        individual.fitness = fitness
        individuals.append(individual)
    ranked = module.Evolution.parent_selection(Population(individuals))
    assert [ind.tags["ps"] for ind in ranked] == [True, True, False, False]
    survivors = module.Evolution.survivor_selection(Population(individuals))
    assert [ind.alive for ind in survivors] == [True] * 4


def test_reproduction_adds_children_that_are_valid_genomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load()
    _shrink(monkeypatch, module)
    evolution = module.Evolution()
    parents = [evolution.create() for _ in range(TINY_POP)]
    for parent in parents:
        parent.tags["ps"] = True
    grown = evolution.reproduction(Population(parents))
    assert len(grown) == 2 * TINY_POP
    for individual in grown:
        assert module.load(individual).nodes


def test_measure_runs_each_workload_in_a_fresh_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load()
    calls: list[list[str]] = []

    def fake(command: list[str], **_kwargs: object) -> object:
        calls.append(command)
        return subprocess.CompletedProcess(
            args=command, returncode=0, stdout=f"noise\n{SAMPLE}\n"
        )

    monkeypatch.setattr(module.subprocess, "run", fake)
    samples = module.measure("bodies", RUNS)
    assert len(samples) == RUNS
    assert [s.wall for s in samples] == pytest.approx([2.0] * RUNS)
    assert calls[0][-2:] == ["--once", "bodies"]


def test_main_prints_the_table_and_rejects_an_unknown_workload(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load()
    monkeypatch.setattr(
        module,
        "measure",
        lambda _name, runs: [module.Sample(wall=1.0, cpu=1.0, rss=1.0)] * runs,
    )
    assert module.main(["bodies", "--runs", "2"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("workload")
    assert lines[1].split()[:2] == ["bodies", "2"]
    assert module.main(["--runs", "1"]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert len(printed) == 1 + len(module.WORKLOADS)
    with pytest.raises(SystemExit):
        module.main(["nonsense"])


def test_once_prints_one_json_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load()
    monkeypatch.setitem(module.WORKLOADS, "tiny", lambda: None)
    assert module.main(["--once", "tiny"]) == 0
    sample = module.Sample.model_validate_json(capsys.readouterr().out)
    assert sample.wall >= 0
