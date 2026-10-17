# DESIGN

## Specs

Proposed 2026-10-17 with the task list; recorded as agreed once the user approves the tasks. All three goals keep core behaviour unchanged.

**test-env-green.** Users: anyone running `pytest tests/unit` on a Linux machine without `libGL.so.1`. Acceptance: `uv run --no-sync pytest tests/unit` reports 0 collection errors and 0 failures. Existing behaviour that must not regress: the 419 tests that pass today. Integration points: `pyproject.toml` (`opencv-python`), `uv.lock`, and the 8 files importing `cv2` (`utils/video_recorder.py`, `simulation/environments/heightmap_functions.py`, 5 examples, and the camera example). Non-goals: choosing the MuJoCo GL backend (that is gl-autodetect), fixing test failures unrelated to libGL, raising coverage. Measurement: one command, deterministic count; time 6.58-6.60 s over three runs.

**dashboards-out.** Acceptance: branch `feature-dashboard` contains today's dashboards; `src/ariel/visualisation/dashboard/` is gone from `main`; no file on `main` except git history refers to it; pytest result identical before and after. Non-goals: consolidating the three dashboards, pushing the branch, touching `visualisation/analysis/`. Dependencies used only by dashboards are a separate step needing per-name approval.

**repo-hygiene.** Acceptance: no `[project.scripts]` in `pyproject.toml`; `src/ariel/ec/generators copy.py` is gone (done, commit 95b6e86); root `CONTRIBUTING.md` states `uv sync` and Python >= 3.12 (done, commits 1632fc7, 27f0dc9); no `TODO(jmdm): description of script` placeholder docstring remains in `src/` (the real TODO at `morphological_descriptor.py:664` is kept); `.claude/dev/testing_and_tooling.md` makes no claim `noxfile.py` contradicts. Non-goals: adding a CLI, fixing lint or type findings.

Architecture alternatives: none; these are removals and fixes, no new modules.

## Decisions

Stack and architecture as found in the repo. Reasons are not recorded unless stated.

- Python ≥ 3.12 with uv for environments and dependencies (`uv.lock`); setuptools build backend; `nox` for task automation (docs only today). Reason: the README names uv; no other reason recorded.
- Simulation on MuJoCo (`mujoco`, `mujoco-mjx[warp]`, `dm-control`); bodies are built as MuJoCo `MjSpec` objects. Reason not recorded.
- Evolution engine in `ariel.ec`: `EA` with operations over a `Population` of `Individual`s stored with SQLModel/SQLAlchemy (database-backed, with restart from the database and an `Archive`). Reason not recorded.
- Body encodings: RoboGen-lite modular robots (`Core`, `Brick`, `Hinge`) decoded from genotypes (tree, CPPN-NEAT, NDE in PyTorch, vectors). The Lynx arm is a separate body family.
- Controllers are CPG-based (`SimpleCPG`, `NaCPG`), and tasks are free-function fitness functions.
- Optimisation libraries available: DEAP, EvoTorch, Nevergrad, plus an in-repo `RevDE`.
- Dashboards (Plotly/Panel) live on the `feature-dashboard` branch, not on `main`; `main` has no plotting dependencies.
- Typing and docs: strict ruff (`ALL`), strict mypy and pyright, NumPy docstrings, 100% coverage target. Free functions and `@dataclass` are preferred over classes (`.claude/dev/functional_vs_oo.md`).
- Global state at import: `ariel/__init__.py` sets `DATA`, a seeded `RNG` (seed 42) and rich logging. Reason not recorded.

- Image handling: switch `opencv-python` to `opencv-python-headless` (decided 2026-10-17, for goal test-env-green). Reason: `import cv2` fails with `libGL.so.1` missing on machines without that system library, and nothing in `src/` needs a cv2 window. The one live window, in `examples/a_mujoco/3_cameras_in_simulation.py`, moves to matplotlib. Performance hint: if image or camera display ever feels slow, suspect that matplotlib replaced `cv2.imshow`, which is faster for live frames. Implemented in T3 with the version pinned to 4.11.0.86 (the locked opencv-python version; PyPI's newest is a 5.x major). Related to but separate from goal gl-autodetect: cv2 and MuJoCo use different GL paths, so this switch does not settle which `MUJOCO_GL` backend to use.
- gl-autodetect config (decided 2026-10-17): `mujoco_user_config.json` lives in the user's working directory, not `~/.config`, because ARIEL is used as a library and already keys its outputs to the working directory (`ariel.DATA = cwd/__data__`). Cost: the probe re-runs per working folder although the result depends on the machine, and the file needs a `.gitignore` note. It is regenerated on first run if missing and on request via the manager script; a missing file must never crash import (falls back to MuJoCo's default). Not yet implemented.
- Entry point: no console script; `[project.scripts]` is removed rather than adding `ariel.__main__` (decided 2026-10-17). Reason: the user does not want a CLI. Not yet implemented.
- Dashboards leave `main` for a `feature-dashboard` branch, to be revisited once the core is fixed (decided 2026-10-17; done in T4, T5). Reason: untested, three near-duplicate versions, about 5.5k lines.

## Rejected approaches

- Keeping `opencv-python` and documenting `sudo apt install libgl1` only: fixes one machine, leaves CI and new users to hit the same error.
- Config file in `~/.config/ariel/` (recommended by Claude; per-machine and shared across checkouts): the user preferred the working directory for library use.
- Adding `ariel.__main__` with a `cli` function: not wanted.
- `ompl` dependency removed in commit 08f5b10 ("problematic"); no other reason recorded.

## Results

| Date | Task | Goal | Result |
|---|---|---|---|
| 2026-10-17 | T1 remove `[project.scripts]` | repo-hygiene | `uv lock --check` passes, `uv sync --all-groups --frozen` clean, pytest 419 passed (unchanged) |
| 2026-10-17 | T2 stale dev docs | repo-hygiene | `.claude/dev/testing_and_tooling.md` and `contribution_workflow.md` no longer name nox sessions, a `tests_compiled` session or a blocking CI that do not exist; pytest 419 passed (unchanged, docs only) |
| 2026-10-17 | T3 `opencv-python-headless==4.11.0.86` | test-env-green | `pytest tests/unit`: 0 collection errors, 437 passed, 0 failed (was 419 passed + 2 collection errors); `cv2` 4.11.0. `uv.lock` diff is large (1102 lines changed) but names changed: opencv-python -> opencv-python-headless and one added macOS-only package (`pyobjc-framework-pubsub`). Camera example edited, not run (no display) |
| 2026-10-17 | T4 dashboards off main | dashboards-out | branch `feature-dashboard` (local, not pushed) holds the 5 dashboard files; `main` no longer has `src/ariel/visualisation/dashboard/` (5,539 lines removed); pytest 437 passed (unchanged); mypy errors 517 -> 262 (a side effect, the dashboards were the untyped-call hotspot) |
| 2026-10-17 | T5 drop `panel`, `plotly`, `nicegui`, `kaleido`, `pyvista` | dashboards-out | `uv.lock` 481 -> 437 packages (-1058 lines); `.venv` 9.3 GB -> 8.4 GB; `uv sync` 1.2 s; pytest 437 passed, mypy 262 errors (both unchanged by T5) |
| 2026-10-17 | T6 placeholder docstrings | repo-hygiene | 18 files (not 17: counted again after T4) had `TODO(...): description of script`; now 0. One real TODO stays (`utils/morphological_descriptor.py:664`). ruff findings 1330 -> 1330 (the Baseline count of 1858 included files since removed); pytest 437 passed |

## Gotchas

See "Things that bite" in CLAUDE.md for repo-wide ones. None specific to a design.
