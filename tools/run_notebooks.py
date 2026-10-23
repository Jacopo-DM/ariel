"""Run every functional notebook in ``tests/functional`` and report each one.

``uv run --no-sync python tools/run_notebooks.py [FILE ...]`` executes each
notebook from a clean kernel in a temporary folder and prints one line per
notebook: its status (PASS, FAIL or SKIP), its name and the seconds it took.
A failing notebook also prints the number of the cell that raised and the
error; a skipped one prints the reason. The exit code is 1 when any notebook
fails (SKIP is not a failure).

With file arguments only those notebooks run. Without, every notebook in
``tests/functional`` and its subfolders runs (Jupyter's
``.ipynb_checkpoints`` copies excepted) except the ones marked
``"slow": true`` in their metadata; ``--slow`` includes them.
``--timeout SECONDS`` sets the limit per notebook (default 600).

A notebook that needs OpenGL carries ``"requires": ["opengl"]`` in its
metadata and is skipped when this machine cannot create a MuJoCo renderer.
``--save`` writes the executed notebooks, with their outputs, back to the
repo; without it the notebooks on disk are never changed.
"""

import subprocess  # ruff: ignore[suspicious-subprocess-import]  # runs one fixed probe command
import sys
import tempfile
import time
from pathlib import Path
from typing import Literal, cast

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, DeadKernelError
from pydantic import BaseModel, ConfigDict

FUNCTIONAL = Path(__file__).resolve().parents[1] / "tests" / "functional"
TIMEOUT_SECONDS = 600
# ariel.rendering picks a working backend (probing in the empty folder), as
# the notebooks do; plain ``import ariel`` only reads a saved choice.
OPENGL_PROBE = (
    "import ariel.rendering, mujoco; "
    "from ariel.simulators import compile_model; "
    "model = compile_model(mujoco.MjSpec.from_string('<mujoco/>')); "
    "mujoco.Renderer(model, 8, 8).close()"
)

type Status = Literal["PASS", "FAIL", "SKIP"]


class Outcome(BaseModel):
    """Result of one notebook."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    status: Status
    seconds: float
    detail: str = ""


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


def read_notebook(path: Path) -> nbformat.NotebookNode:
    """Read a notebook file (``nbformat`` has no type hints).

    Returns
    -------
    nbformat.NotebookNode
        The notebook, converted to format version 4.
    """
    notebook = nbformat.read(path, as_version=4)  # type: ignore[no-untyped-call]  # nbformat is untyped
    return cast("nbformat.NotebookNode", notebook)


def write_notebook(notebook: nbformat.NotebookNode, path: Path) -> None:
    """Write a notebook file (``nbformat`` has no type hints)."""
    nbformat.write(notebook, path)  # type: ignore[no-untyped-call]  # nbformat is untyped


def needs_opengl(notebook: nbformat.NotebookNode) -> bool:
    """Tell whether the notebook metadata asks for OpenGL.

    Returns
    -------
    bool
        True when ``requires`` in the metadata lists ``opengl``.
    """
    return "opengl" in notebook.metadata.get("requires", [])


def failing_cell(notebook: nbformat.NotebookNode) -> int:
    """Find the number (from 1) of the cell that stopped the run.

    Cells run in order and the run stops at the first error, so the failing
    cell is the last one that was started (it has an execution count). The
    error output itself may not be stored yet when the exception arrives.

    Returns
    -------
    int
        The cell number, or 0 when no cell was started.
    """
    started = [
        number
        for number, cell in enumerate(notebook.cells, start=1)
        if cell.cell_type == "code" and cell.get("execution_count") is not None
    ]
    return started[-1] if started else 0


def is_slow(notebook: nbformat.NotebookNode) -> bool:
    """Tell whether the notebook metadata marks it as slow.

    Returns
    -------
    bool
        True when ``slow`` in the metadata is true.
    """
    return bool(notebook.metadata.get("slow", False))


def run_notebook(
    path: Path,
    *,
    opengl: bool,
    save: bool = False,
    timeout: int = TIMEOUT_SECONDS,
) -> Outcome:
    """Execute one notebook from a clean kernel in a temporary folder.

    Returns
    -------
    Outcome
        PASS, FAIL (with the failing cell) or SKIP (OpenGL missing).
    """
    notebook = read_notebook(path)
    if needs_opengl(notebook) and not opengl:
        return Outcome(
            name=path.stem,
            status="SKIP",
            seconds=0.0,
            detail="needs opengl, which does not work on this machine",
        )
    # Forget the counts of an earlier saved run: failing_cell reads them
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.execution_count = None
    start = time.perf_counter()
    status: Status = "PASS"
    detail = ""
    with tempfile.TemporaryDirectory() as folder:
        client = NotebookClient(
            notebook,
            timeout=timeout,
            kernel_name="python3",
            resources={"metadata": {"path": folder}},
        )
        try:
            client.execute()
        except CellExecutionError as error:
            status = "FAIL"
            cell = failing_cell(notebook)
            where = f"cell {cell}: " if cell else ""
            detail = f"{where}{error.ename}: {error.evalue}"
        except DeadKernelError:
            status = "FAIL"
            detail = "the kernel died (a crash, not a Python error)"
        except TimeoutError as error:
            status = "FAIL"
            detail = f"timeout after {timeout} s: {error}"
    if save and status == "PASS":
        write_notebook(notebook, path)
    return Outcome(
        name=path.stem,
        status=status,
        seconds=time.perf_counter() - start,
        detail=detail,
    )


def main(argv: list[str]) -> int:
    """Run the notebooks, print one line each and return the exit code.

    Returns
    -------
    int
        1 when any notebook failed, otherwise 0.
    """
    args = list(argv)
    save = "--save" in args
    slow = "--slow" in args
    timeout = TIMEOUT_SECONDS
    if "--timeout" in args:
        at = args.index("--timeout")
        timeout = int(args[at + 1])
        del args[at : at + 2]
    names = [a for a in args if not a.startswith("--")]
    paths = (
        [Path(n) for n in names]
        if names
        else sorted(
            path
            for path in FUNCTIONAL.rglob("*.ipynb")
            if ".ipynb_checkpoints" not in path.parts
        )
    )
    opengl = opengl_works()
    failed = 0
    for path in paths:
        if not names and not slow and is_slow(read_notebook(path)):
            continue
        outcome = run_notebook(path, opengl=opengl, save=save, timeout=timeout)
        sys.stdout.write(
            f"{outcome.status:<5} {outcome.name:<20} {outcome.seconds:7.1f} s\n"
        )
        if outcome.detail:
            sys.stdout.write(f"      {outcome.detail}\n")
        failed += outcome.status == "FAIL"
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
