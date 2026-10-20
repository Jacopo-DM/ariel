"""Time the four workloads of the speed-up goal.

Run from the repo root: ``uv run --no-sync python tools/benchmark.py
[WORKLOAD ...] [--runs N]``. With no workload name it runs all four. Every run
happens in a fresh process, so peak memory is not carried over from an earlier
run. It prints, per workload, the median and the spread (smallest and largest)
of the wall-clock time, the median CPU time and the median peak memory.

Workloads (all seeded, so the work done is the same on every run):

* ``evolution``: a tree-genome evolution of ``EVO_POP`` robots for
    ``EVO_STEPS`` generations; every new robot is built and simulated for
    ``EVO_SIM_STEPS`` steps with a sine control.
* ``simulation``: one prebuilt robot in a flat world for ``SIM_STEPS`` steps.
* ``bodies``: ``BODIES`` random 20-module graphs built into bodies, plain and
    with ``repair_overlaps=True``.
* ``tests``: the whole test suite, ``uv run --no-sync pytest -q`` (it includes
    the coverage run that the gate uses).

Notes
-----
    * Peak memory is the largest resident set of the process (of its child
        processes for ``tests``), in MiB, from ``resource.getrusage``.
    * A gain smaller than the spread of a workload does not count.
"""

import argparse
import random
import resource
import statistics
import subprocess  # ruff: ignore[suspicious-subprocess-import]  # runs this script and pytest
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, cast

import mujoco
import numpy as np

# Local libraries
from ariel.body_phenotypes import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.prebuilt_robots import john_set
from ariel.ec import EA, EAOperation, Individual, Population
from ariel.ec.genotypes.tree import (
    TreeGenome,
    crossover_subtree,
    mutate_replace_node,
    mutate_subtree_replacement,
    prune_invalid_edges,
    random_tree,
)
from ariel.parameters import FrozenModel
from ariel.simulation import SimpleFlatWorld

if TYPE_CHECKING:
    from ariel.ec.individual import JSONIterable

SEED = 42
HALF = 0.5
EVO_POP = 20
EVO_STEPS = 10
EVO_MODULES = 15
EVO_SIM_STEPS = 1000
SIM_STEPS = 100_000
BODIES = 60
BODY_MODULES = 20
SPAWN_HEIGHT = 0.1
MIB = 1024 * 1024
KIB_PER_MIB = 1024  # ru_maxrss is in KiB on Linux
RUN_FLAG = "--once"
ROOT = Path(__file__).resolve().parents[1]


class Sample(FrozenModel):
    """What one run of a workload cost."""

    wall: float  # seconds
    cpu: float  # seconds
    rss: float  # MiB


def simulate(spec: mujoco.MjSpec, steps: int) -> float:
    """Drop a robot in a flat world and step it with a sine control.

    Parameters
    ----------
    spec : mujoco.MjSpec
        The robot.
    steps : int
        How many physics steps to take.

    Returns
    -------
    float
        How far the first body moved along x.
    """
    world = SimpleFlatWorld(load_precompiled=False)
    world.spawn(spec, position=(0.0, 0.0, SPAWN_HEIGHT))
    model = world.spec.compile()
    data = mujoco.MjData(model)
    start = float(data.qpos[0])
    for _ in range(steps):
        data.ctrl[:] = np.sin(data.time * 2 * np.pi)
        mujoco.mj_step(model, data)
    return float(data.qpos[0]) - start


def load(individual: Individual) -> TreeGenome:
    """Read the tree genome of an individual.

    Parameters
    ----------
    individual : Individual
        An individual whose genotype was stored by ``store``.

    Returns
    -------
    TreeGenome
        The genome.
    """
    return TreeGenome.from_dict(
        cast("Mapping[str, object]", individual.genotype)
    )


def store(individual: Individual, genome: TreeGenome) -> None:
    """Store a tree genome as the genotype of an individual.

    Parameters
    ----------
    individual : Individual
        The individual.
    genome : TreeGenome
        The genome.
    """
    individual.genotype = cast("JSONIterable", genome.to_dict())


class Evolution:
    """The seeded tree-genome evolution of the ``evolution`` workload."""

    def __init__(self) -> None:
        """Seed both random generators."""
        random.seed(SEED)
        self.rng = np.random.default_rng(SEED)

    @staticmethod
    def create() -> Individual:
        """Make one random individual.

        Returns
        -------
        Individual
            A new individual with a random tree genome.
        """
        individual = Individual()
        store(individual, random_tree(EVO_MODULES))
        return individual

    @staticmethod
    def evaluate(population: Population) -> Population:
        """Build and simulate every new individual.

        Parameters
        ----------
        population : Population
            The population.

        Returns
        -------
        Population
            The same population, with the new fitness values.
        """
        for individual in population:
            if individual.alive and individual.requires_eval:
                genome = load(individual)
                core = construct_mjspec_from_graph(genome.to_networkx())
                individual.fitness = simulate(core.spec, EVO_SIM_STEPS)
                individual.requires_eval = False
        return population

    @staticmethod
    def parent_selection(population: Population) -> Population:
        """Mark the better half as parents.

        Parameters
        ----------
        population : Population
            The population.

        Returns
        -------
        Population
            The population, best first.
        """
        ranked = population.sort(sort="max")
        for rank, individual in enumerate(ranked):
            individual.tags["ps"] = rank < len(ranked) // 2
        return ranked

    def offspring(self, parents: list[Individual]) -> Individual:
        """Make one child by crossover or copy, then mutation.

        Parameters
        ----------
        parents : list[Individual]
            The parents to draw two from.

        Returns
        -------
        Individual
            The child.
        """
        first, second = random.sample(parents, 2)
        genome = load(first)
        if self.rng.random() < HALF:
            other = load(second)
            genome = crossover_subtree(genome, other)[0]
        if self.rng.random() < HALF:
            mutate_replace_node(genome)
        else:
            mutate_subtree_replacement(genome, max_modules=EVO_MODULES)
        prune_invalid_edges(genome)
        child = Individual()
        store(child, genome)
        return child

    def reproduction(self, population: Population) -> Population:
        """Add ``EVO_POP`` children to the population.

        Parameters
        ----------
        population : Population
            The population, with parents marked.

        Returns
        -------
        Population
            The population with the children added.
        """
        parents = [ind for ind in population if ind.tags.get("ps")]
        population.extend([self.offspring(parents) for _ in range(EVO_POP)])
        return population

    @staticmethod
    def survivor_selection(population: Population) -> Population:
        """Keep the ``EVO_POP`` fittest alive.

        Parameters
        ----------
        population : Population
            The population.

        Returns
        -------
        Population
            The population, best first, with the others marked dead.
        """
        ranked = population.sort(sort="max", attribute="fitness_")
        for rank, individual in enumerate(ranked):
            individual.alive = rank < EVO_POP
        return ranked

    def run(self) -> None:
        """Evolve for ``EVO_STEPS`` generations in a temporary folder."""
        first = self.evaluate(
            Population([self.create() for _ in range(EVO_POP)])
        )
        operations = [
            EAOperation(self.parent_selection),
            EAOperation(self.reproduction),
            EAOperation(self.evaluate),
            EAOperation(self.survivor_selection),
        ]
        with tempfile.TemporaryDirectory() as folder:
            EA(
                first,
                operations=operations,
                num_steps=EVO_STEPS,
                is_maximisation=True,
                quiet=True,
                db_file_path=Path(folder) / "database.db",
                db_handling="delete",
            ).run()


def evolution() -> None:
    """Run the seeded tree-genome evolution."""
    Evolution().run()


def simulation() -> None:
    """Simulate one prebuilt robot for ``SIM_STEPS`` steps."""
    simulate(john_set.gecko().spec, SIM_STEPS)


def bodies() -> None:
    """Build random bodies without and with the overlap repair."""
    random.seed(SEED)
    graphs = [random_tree(BODY_MODULES).to_networkx() for _ in range(BODIES)]
    for graph in graphs:
        construct_mjspec_from_graph(graph)
    for graph in graphs:
        construct_mjspec_from_graph(graph, repair_overlaps=True)


def tests() -> None:
    """Run the whole test suite in a child process."""
    subprocess.run(
        ["uv", "run", "--no-sync", "pytest", "-q", "-p", "no:cacheprovider"],  # ruff: ignore[start-process-with-partial-path]  # uv from PATH
        cwd=ROOT,
        check=False,
        capture_output=True,
    )


WORKLOADS: dict[str, Callable[[], None]] = {
    "evolution": evolution,
    "simulation": simulation,
    "bodies": bodies,
    "tests": tests,
}


def run_once(name: str) -> Sample:
    """Run one workload in this process and report what it cost.

    Parameters
    ----------
    name : str
        A key of ``WORKLOADS``.

    Returns
    -------
    Sample
        Wall time, CPU time and peak memory. For ``tests`` the CPU time and
        memory are those of the child processes.
    """
    who = resource.RUSAGE_CHILDREN if name == "tests" else resource.RUSAGE_SELF
    wall = time.perf_counter()
    cpu = time.process_time()
    WORKLOADS[name]()
    usage = resource.getrusage(who)
    return Sample(
        wall=time.perf_counter() - wall,
        cpu=usage.ru_utime + usage.ru_stime
        if name == "tests"
        else time.process_time() - cpu,
        rss=usage.ru_maxrss / KIB_PER_MIB,
    )


def measure(name: str, runs: int) -> list[Sample]:
    """Run a workload ``runs`` times, each in a fresh process.

    Parameters
    ----------
    name : str
        A key of ``WORKLOADS``.
    runs : int
        How many runs.

    Returns
    -------
    list[Sample]
        One sample per run.
    """
    command = [sys.executable, str(Path(__file__).resolve()), RUN_FLAG, name]
    return [
        Sample.model_validate_json(
            subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  # this script
                command,
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()[-1]
        )
        for _ in range(runs)
    ]


def summarise(name: str, samples: list[Sample]) -> str:
    """Format the median and spread of the samples of one workload.

    Parameters
    ----------
    name : str
        The workload.
    samples : list[Sample]
        Its runs.

    Returns
    -------
    str
        One line: runs, wall median and range, CPU median, memory median.
    """
    wall = [s.wall for s in samples]
    return (
        f"{name:<11} {len(samples):>4}  "
        f"{statistics.median(wall):>9.3f}  {min(wall):>9.3f}  "
        f"{max(wall):>9.3f}  "
        f"{statistics.median(s.cpu for s in samples):>9.3f}  "
        f"{statistics.median(s.rss for s in samples):>9.1f}\n"
    )


def main(argv: list[str]) -> int:
    """Run the workloads and print the table.

    Parameters
    ----------
    argv : list[str]
        The command line without the program name.

    Returns
    -------
    int
        Always 0.
    """
    if argv[:1] == [RUN_FLAG]:
        sys.stdout.write(run_once(argv[1]).model_dump_json() + "\n")
        return 0
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("workloads", nargs="*")
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args(argv)
    unknown = [name for name in args.workloads if name not in WORKLOADS]
    if unknown:
        parser.error(
            f"unknown workload {unknown}; choose from {list(WORKLOADS)}"
        )
    sys.stdout.write(
        "workload    runs  wall med s  wall min  wall max   cpu med s  "
        "rss med MiB\n"
    )
    for name in args.workloads or list(WORKLOADS):
        sys.stdout.write(summarise(name, measure(name, args.runs)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
