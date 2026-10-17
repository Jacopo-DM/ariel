# DESIGN

## Specs

None recorded.

## Decisions

Stack and architecture as found in the repo. Reasons are not recorded unless stated.

- Python ≥ 3.12 with uv for environments and dependencies (`uv.lock`); setuptools build backend; `nox` for task automation (docs only today). Reason: the README names uv; no other reason recorded.
- Simulation on MuJoCo (`mujoco`, `mujoco-mjx[warp]`, `dm-control`); bodies are built as MuJoCo `MjSpec` objects. Reason not recorded.
- Evolution engine in `ariel.ec`: `EA` with operations over a `Population` of `Individual`s stored with SQLModel/SQLAlchemy (database-backed, with restart from the database and an `Archive`). Reason not recorded.
- Body encodings: RoboGen-lite modular robots (`Core`, `Brick`, `Hinge`) decoded from genotypes (tree, CPPN-NEAT, NDE in PyTorch, vectors). The Lynx arm is a separate body family.
- Controllers are CPG-based (`SimpleCPG`, `NaCPG`), and tasks are free-function fitness functions.
- Optimisation libraries available: DEAP, EvoTorch, Nevergrad, plus an in-repo `RevDE`.
- Dashboards use Plotly/Panel/NiceGUI.
- Typing and docs: strict ruff (`ALL`), strict mypy and pyright, NumPy docstrings, 100% coverage target. Free functions and `@dataclass` are preferred over classes (`.claude/dev/functional_vs_oo.md`).
- Global state at import: `ariel/__init__.py` sets `DATA`, a seeded `RNG` (seed 42) and rich logging. Reason not recorded.

- Image handling: switch `opencv-python` to `opencv-python-headless` (decided 2026-10-17, for goal test-env-green). Reason: `import cv2` fails with `libGL.so.1` missing on machines without that system library, and nothing in `src/` needs a cv2 window. The one live window, in `examples/a_mujoco/3_cameras_in_simulation.py`, moves to matplotlib. Performance hint: if image or camera display ever feels slow, suspect that matplotlib replaced `cv2.imshow`, which is faster for live frames. Not yet implemented. Related to but separate from goal gl-autodetect: cv2 and MuJoCo use different GL paths, so this switch does not settle which `MUJOCO_GL` backend to use.
- Entry point: no console script; `[project.scripts]` is removed rather than adding `ariel.__main__` (decided 2026-10-17). Reason: the user does not want a CLI. Not yet implemented.
- Dashboards leave `main` for a `feature-dashboard` branch, to be revisited once the core is fixed (decided 2026-10-17). Reason: untested, three near-duplicate versions, about 5k lines. Not yet implemented.

## Rejected approaches

- Keeping `opencv-python` and documenting `sudo apt install libgl1` only: fixes one machine, leaves CI and new users to hit the same error.
- Adding `ariel.__main__` with a `cli` function: not wanted.
- `ompl` dependency removed in commit 08f5b10 ("problematic"); no other reason recorded.

## Results

None recorded.

## Gotchas

See "Things that bite" in CLAUDE.md for repo-wide ones. None specific to a design.
