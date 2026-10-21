"""Importing ARIEL's modules must not create folders in the working folder."""

# Standard library
import subprocess  # ruff: ignore[suspicious-subprocess-import]  # runs a fixed child process
import sys
from pathlib import Path

CODE = (
    "import ariel.parameters\n"
    "import ariel.ec.genotypes.nde\n"
    "import ariel.ec.optimizers.revde\n"
)


def test_importing_modules_leaves_the_working_folder_empty(
    tmp_path: Path,
) -> None:
    """Only code that saves something creates ``__data__``."""
    subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  # fixed argv, no user input
        [sys.executable, "-I", "-c", CODE],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert list(tmp_path.iterdir()) == []
