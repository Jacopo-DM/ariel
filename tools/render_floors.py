"""Render one picture of every world to look at its floor.

Run from the repo root on a machine with a working OpenGL backend:
``uv run --no-sync python tools/render_floors.py``. The pictures go to
``__data__/floors/`` under the current folder, and their paths are printed.
"""

import sys
from collections.abc import Callable
from pathlib import Path

# ariel first: it picks the OpenGL backend, which MuJoCo reads when it loads
from ariel import DATA
from ariel.rendering import single_frame_renderer
from ariel.simulation import (
    AmphitheatreTerrainWorld,
    BaseWorld,
    CompoundWorld,
    CraterTerrainWorld,
    OlympicArena,
    RuggedTerrainWorld,
    RuggedTiltedWorld,
    SimpleFlatWorld,
    SimpleTiltedWorld,
)

# isort: split
import mujoco

WORLDS: dict[str, Callable[[], BaseWorld]] = {
    "simple-flat": lambda: SimpleFlatWorld(load_precompiled=False),
    "simple-tilted": lambda: SimpleTiltedWorld(load_precompiled=False),
    "simple-tilted-plain": lambda: SimpleTiltedWorld(
        checker_floor=False,
        load_precompiled=False,
    ),
    "compound": lambda: CompoundWorld(load_precompiled=False),
    "rugged": lambda: RuggedTerrainWorld(load_precompiled=False),
    "rugged-tilted": lambda: RuggedTiltedWorld(load_precompiled=False),
    "crater": lambda: CraterTerrainWorld(load_precompiled=False),
    "amphitheatre": lambda: AmphitheatreTerrainWorld(load_precompiled=False),
    "olympic-arena": lambda: OlympicArena(load_precompiled=False),  # type: ignore[abstract]  # known bug: OlympicArena is abstract (PLAN.md)
}


def render_all(folder: Path) -> list[Path]:
    """Render every world once and save the pictures.

    Parameters
    ----------
    folder : Path
        Where the PNG files go; created if missing.

    Returns
    -------
    list[Path]
        The files written, one per world.
    """
    folder.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, make in WORLDS.items():
        model = make().spec.compile()
        data = mujoco.MjData(model)
        path = folder / f"{name}.png"
        single_frame_renderer(model, data, save=True, save_path=path)
        written.append(path)
    return written


def main() -> int:
    """Render the worlds into ``__data__/floors`` and print the paths.

    Returns
    -------
    int
        Always 0.
    """
    for path in render_all(DATA / "floors"):
        sys.stdout.write(f"{path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
