"""Run every script in ``examples/`` briefly and report each one.

``uv run --no-sync python tools/run_examples.py [SCRIPT ...]`` runs each
example as its own process, in a temporary folder, with the small settings
of its row in ``EXAMPLES``, and prints one line per script: its status
(PASS, FAIL or SKIP), its path and the seconds it took. A failing script
also prints the last line it wrote to stderr; a skipped one prints the
reason. The exit code is 1 when any script fails (SKIP is not a failure).

Every ``.py`` file under ``examples/`` must be a row of ``EXAMPLES`` or be
listed in ``MODULES`` (files the scripts import, which are not run on their
own); a file in neither is a FAIL, so a new example cannot skip the check.

A row may name a producer with ``after``: the script that writes the files
it reads. The producer then runs first, in the same folder, and its readers
run in that folder after it; a reader whose producer did not pass is
skipped. An argument with ``*`` in it is matched against the files in that
folder, and the last match in sorted order is passed (the producers stamp
their file names with the time).

A row with ``opengl=True`` renders, and is skipped when this machine cannot
create a MuJoCo renderer. The scripts run with ``MPLBACKEND=Agg``, so a
``plt.show()`` returns at once, and with no input, so a prompt fails.

With script arguments (paths relative to ``examples/``) only those scripts
run, together with their producers. Up to ``JOBS`` scripts run at once:
first every producer and every script without one, then the readers.
"""

import os
import subprocess  # ruff: ignore[suspicious-subprocess-import]  # runs the repo's own examples
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

EXAMPLES_FOLDER = Path(__file__).resolve().parents[1] / "examples"
TIMEOUT_SECONDS = 120
# Scripts run at once; one at a time took 3 min 37 s for all 33 (ER6)
JOBS = 4
# The same probe as tools/run_notebooks.py, which is the user's shared
# template and stays a copy of it, so the probe is repeated here.
OPENGL_PROBE = (
    "import ariel.rendering, mujoco; "
    "from ariel.simulators import compile_model; "
    "model = compile_model(mujoco.MjSpec.from_string('<mujoco/>')); "
    "mujoco.Renderer(model, 8, 8).close()"
)

type Status = Literal["PASS", "FAIL", "SKIP"]


class Example(BaseModel):
    """How the runner calls one script."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    args: tuple[str, ...] = ()
    opengl: bool = False
    after: str = ""
    timeout: int = TIMEOUT_SECONDS


class Outcome(BaseModel):
    """Result of one script."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    status: Status
    seconds: float
    detail: str = ""


C_GENOTYPES_8_DATA = "__data__/8_bootstrap_pipeline_population_evolution"
C_GENOTYPES_3 = "c_genotypes/3_body_brain_joint_evolution_tree_nn_learning.py"
C_GENOTYPES_4 = (
    "c_genotypes/"
    "4_body_brain_joint_evolution_tree_nn_learning_multiprocessing.py"
)
C_GENOTYPES_8 = "c_genotypes/8_bootstrap_pipeline_population_evolution.py"


def row(
    args: str = "",
    *,
    opengl: bool = False,
    after: str = "",
) -> Example:
    """Build a row from its arguments written as one string.

    Returns
    -------
    Example
        The row, with ``args`` split on spaces.
    """
    return Example(args=tuple(args.split()), opengl=opengl, after=after)


EXAMPLES: dict[str, Example] = {
    "a_mujoco/0_mujoco_launcher.py": row("--no-visualize"),
    "a_mujoco/1_render_single_frame.py": row(opengl=True),
    "a_mujoco/2_video_recorder.py": row(opengl=True),
    "a_mujoco/3_cameras_in_simulation.py": row("--no-visualize", opengl=True),
    "b_robots/_graph_to_robot.py": row("--no-visualize", opengl=True),
    "b_robots/_hi_prob_dec.py": row(),
    "b_robots/_robogen_build.py": row("--no-visualize", opengl=True),
    "b_robots/_terrains.py": row("--no-visualize", opengl=True),
    "c_genotypes/1_body_evolution_tree.py": row(
        "--budget 1 --pop 4 --max-modules 5 --no-visualize"
    ),
    "c_genotypes/2_body_brain_evolution_tree.py": row(
        "--budget 1 --pop 4 --dur 3 --no-visualize", opengl=True
    ),
    C_GENOTYPES_3: row(
        "--budget 1 --pop 2 --dur 2 --eval-delay 0.5 --learn-budget 1"
        " --learn-pop 2 --max-modules 5 --max-depth 3 --no-visualize"
    ),
    C_GENOTYPES_4: row(
        "--bud 1 --pop 2 --dur 2 --eval-delay 0.5 --learn-bud 1 --learn-pop 2"
        " --eval-workers 2 --max-modules 5 --max-depth 3 --no-save-video"
        " --no-visualize"
    ),
    "c_genotypes/5_body_evolution_cppn.py": row(
        "--budget 1 --pop 4 --max-modules 5 --no-visualize"
    ),
    C_GENOTYPES_8: row(
        "--bootstrap-pop 2 --bootstrap-morph-budget 1 --bootstrap-morph-pop 2"
        " --bootstrap-max-attempts 1 --evo-budget 1 --max-modules 5"
        " --max-depth 3 --learn-budget 1 --learn-pop 2 --learn-workers 1"
        " --dur 2 --eval-delay 0.5 --no-visualize"
    ),
    "c_genotypes/cppn_genome_checks.py": row(),
    "c_genotypes/runner_best_individual.py": row(
        f"--morphology {C_GENOTYPES_8_DATA}/best_morphology_*.json"
        f" --brain {C_GENOTYPES_8_DATA}/best_brain_*.npy"
        " --duration 1 --no-visualize",
        after=C_GENOTYPES_8,
    ),
    "c_genotypes/simulate_from_db.py": row(
        "--db_path __data__/2_body_brain_evolution_tree/database.db"
        " --viewer simple --dur 1",
        after="c_genotypes/2_body_brain_evolution_tree.py",
    ),
    "new_EC_engine_example.py": row(),
    "parallel_robot_eval.py": row(),
    "re_book/1_brain_evolution.py": row(
        "--budget 1 --dur 1 --population 2", opengl=True
    ),
    "re_book/1_brain_evolution_multiprocessing.py": row(
        "--budget 1 --dur 1 --population 2 --workers 2 --seed 1", opengl=True
    ),
    "re_book/1_brain_evolution_multithreaded.py": row(
        "--budget 1 --dur 1 --population 2 --workers 2", opengl=True
    ),
    "re_book/2_body_brain_evolution.py": row(
        "--budget 1 --pop 4 --dur 3 --no-visualize", opengl=True
    ),
    "re_book/3_body_brain_lr.py": row(opengl=True),
    "re_book/4_sequential_waypoints.py": row(
        "--budget 1 --population 2 --dur 1 --workers 1 --no-video", opengl=True
    ),
    "re_book/5_randomized_waypoints.py": row(
        "--budget 1 --population 2 --dur 1 --workers 1 --no-video",
        opengl=True,
    ),
    "z_ec_course/A2_template.py": row("--no-visualize"),
    "z_ec_course/A2_template_cpg.py": row("--budget 2 --dur 2 --no-visualize"),
    "z_ec_course/A3_competition_tester.py": row(
        "--graph __data__/A3_template/robot_graph.json --duration 2",
        opengl=True,
        after="z_ec_course/A3_template.py",
    ),
    "z_ec_course/A3_template.py": row(opengl=True),
    "z_ec_course/A3_template_jack.py": row("--samples 2 --dur 2", opengl=True),
    "z_ec_course/plot_fit_per_gen.py": row(
        after="c_genotypes/1_body_evolution_tree.py"
    ),
    "z_ec_course/plot_learning_deltas_per_individual.py": row(
        "--path __data__/"
        "4_body_brain_joint_evolution_tree_nn_learning_multiprocessing",
        after=C_GENOTYPES_4,
    ),
}

# Imported by the scripts, never run on their own
MODULES = frozenset({
    "re_book/_vision.py",
    "z_ec_course/A3_plot_function.py",
})


def opengl_works() -> bool:
    """Tell whether MuJoCo can create a renderer on this machine.

    Returns
    -------
    bool
        True when a tiny render context opens in a fresh process.
    """
    with tempfile.TemporaryDirectory() as folder:
        done = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  # fixed argv, no user input
            [sys.executable, "-c", OPENGL_PROBE],
            cwd=folder,
            capture_output=True,
            check=False,
        )
    return done.returncode == 0


def expand(args: tuple[str, ...], folder: Path) -> list[str]:
    """Replace each argument with ``*`` by its last match in ``folder``.

    Returns
    -------
    list[str]
        The arguments, with every pattern replaced by a relative path.

    Raises
    ------
    FileNotFoundError
        When a pattern matches no file.
    """
    expanded: list[str] = []
    for arg in args:
        if "*" not in arg:
            expanded.append(arg)
            continue
        matches = sorted(folder.glob(arg))
        if not matches:
            msg = f"no file matches {arg}"
            raise FileNotFoundError(msg)
        expanded.append(str(matches[-1].relative_to(folder)))
    return expanded


def last_line(text: str) -> str:
    """Return the last line of ``text`` that is not blank.

    Returns
    -------
    str
        The line without surrounding spaces, or an empty string.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def run_example(
    name: str,
    example: Example,
    folder: Path,
    *,
    opengl: bool,
) -> Outcome:
    """Run one script as its own process in ``folder``.

    Returns
    -------
    Outcome
        PASS, FAIL (with the last line of stderr) or SKIP (OpenGL missing).
    """
    if example.opengl and not opengl:
        return Outcome(
            name=name,
            status="SKIP",
            seconds=0.0,
            detail="needs opengl, which does not work on this machine",
        )
    try:
        args = expand(example.args, folder)
    except FileNotFoundError as error:
        return Outcome(name=name, status="FAIL", seconds=0.0, detail=str(error))
    start = time.perf_counter()
    try:
        done = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  # the repo's own script, fixed arguments
            [sys.executable, str(EXAMPLES_FOLDER / name), *args],
            cwd=folder,
            env={**os.environ, "MPLBACKEND": "Agg"},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=example.timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return Outcome(
            name=name,
            status="FAIL",
            seconds=time.perf_counter() - start,
            detail=f"timeout after {example.timeout} s",
        )
    seconds = time.perf_counter() - start
    if done.returncode != 0:
        detail = last_line(done.stderr) or last_line(done.stdout)
        return Outcome(
            name=name,
            status="FAIL",
            seconds=seconds,
            detail=f"exit {done.returncode}: {detail}",
        )
    return Outcome(name=name, status="PASS", seconds=seconds)


def unlisted() -> list[str]:
    """List the ``.py`` files under ``examples/`` that no table names.

    Returns
    -------
    list[str]
        Paths relative to ``examples/``, sorted.
    """
    found = (
        path.relative_to(EXAMPLES_FOLDER).as_posix()
        for path in EXAMPLES_FOLDER.rglob("*.py")
    )
    return sorted(
        name for name in found if name not in EXAMPLES and name not in MODULES
    )


def chosen(names: list[str]) -> list[str]:
    """Order the scripts to run, each producer before its readers.

    Returns
    -------
    list[str]
        The rows of ``EXAMPLES`` named in ``names`` (all of them when it is
        empty) and their producers, producers first.
    """
    wanted = set(names) if names else set(EXAMPLES)
    wanted |= {EXAMPLES[name].after for name in wanted if EXAMPLES[name].after}
    producers = sorted(name for name in wanted if not EXAMPLES[name].after)
    readers = sorted(name for name in wanted if EXAMPLES[name].after)
    return producers + readers


def report(outcome: Outcome) -> None:
    """Print the line of one script, and its detail below it."""
    sys.stdout.write(
        f"{outcome.status:<5} {outcome.name:<60} {outcome.seconds:6.1f} s\n"
    )
    if outcome.detail:
        sys.stdout.write(f"      {outcome.detail}\n")


def main(argv: list[str]) -> int:
    """Run the examples, print one line each and return the exit code.

    Returns
    -------
    int
        1 when any script failed or is in no table, otherwise 0.
    """
    names = [arg for arg in argv if not arg.startswith("--")]
    failed = 0
    if not names:
        for name in unlisted():
            report(
                Outcome(
                    name=name,
                    status="FAIL",
                    seconds=0.0,
                    detail="not in EXAMPLES or MODULES of this runner",
                )
            )
            failed += 1
    opengl = opengl_works()
    order = chosen(names)
    outcomes: dict[str, Outcome] = {}
    with (
        tempfile.TemporaryDirectory() as root,
        ThreadPoolExecutor(max_workers=JOBS) as pool,
    ):

        def start(name: str) -> Outcome:
            example = EXAMPLES[name]
            producer = example.after
            if producer and outcomes[producer].status != "PASS":
                return Outcome(
                    name=name,
                    status="SKIP",
                    seconds=0.0,
                    detail=f"needs {producer}, which did not pass",
                )
            folder = Path(root) / (producer or name)
            folder.mkdir(parents=True, exist_ok=True)
            return run_example(name, example, folder, opengl=opengl)

        # Producers first, then the scripts that read their files
        for wave in (
            [name for name in order if not EXAMPLES[name].after],
            [name for name in order if EXAMPLES[name].after],
        ):
            for name, outcome in zip(wave, pool.map(start, wave), strict=True):
                outcomes[name] = outcome
                report(outcome)
                failed += outcome.status == "FAIL"
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
