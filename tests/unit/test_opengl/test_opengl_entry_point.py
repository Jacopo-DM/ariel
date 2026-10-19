"""Importing ``ariel`` and ``ariel.opengl`` must not load MuJoCo."""

# Standard library
import subprocess  # ruff: ignore[suspicious-subprocess-import]  # runs a fixed child process
import sys
from pathlib import Path

CODE = (
    "import sys\n"
    "import ariel\n"
    "import ariel.opengl\n"
    "print('mujoco' in sys.modules)\n"
)


def test_importing_ariel_and_opengl_does_not_load_mujoco(
    tmp_path: Path,
) -> None:
    """MuJoCo reads MUJOCO_GL when it loads, after ``ariel`` has set it."""
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  # fixed argv, no user input
        [sys.executable, "-I", "-c", CODE],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"
