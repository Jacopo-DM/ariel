# PLAN

## Goals

| Name | Goal | Kind | Success measure | Status | Added |
|---|---|---|---|---|---|
| test-env-green | All unit tests can be collected and pass | improvement, behaviour unchanged | `uv run pytest tests/unit` reports 0 collection errors and 0 failures (today 2 collection errors, 419 passed) | draft | 2026-10-17 |
| dashboards-out | Move the dashboards off `main` into a `feature-dashboard` branch | improvement, behaviour unchanged for the core | `feature-dashboard` branch holds the current `main`; on `main`, `src/ariel/visualisation/dashboard/` is gone, references in docs/README/.claude updated, dependencies used only by dashboards (panel, nicegui, plotly, kaleido, pyvista: none imported elsewhere in `.py` files) removed or listed as a decision; `pytest tests/unit` result is unchanged | draft | 2026-10-17 |
| repo-hygiene | Remove the broken entry point, the stray file and the stale docs | improvement, behaviour unchanged | the `[project.scripts]` entry is removed from `pyproject.toml` (decided 2026-10-17; `ariel.__main__` does not exist); `src/ariel/ec/generators copy.py` removed (done 2026-10-17, user approved; unreferenced); `CONTRIBUTING.md` says `uv sync` and Python >= 3.12; placeholder `TODO(jmdm)` docstrings count in `src/` goes from 19 to 0 or each is listed as kept | draft | 2026-10-17 |
| lint-types-policy | Decide which ruff/mypy/pyright/pre-commit rules are sensible, then make the code meet them | improvement | pending: measure set after the config review with the user (baseline: ruff 1858 findings, format 51 files, mypy 517 errors) | draft, blocked on config review | 2026-10-17 |
| gl-autodetect | ARIEL never crashes because the user's OpenGL backend is unsuitable. A manager script runs one subprocess worker per backend (glfw, egl, osmesa), records which work and how fast in `mujoco_user_config.json`, and ARIEL reads that file at import to set `MUJOCO_GL`. | new behaviour | not set yet; acceptance criteria to be proposed and agreed (candidates: the manager writes a valid file on this Linux machine; a crashing backend is recorded as failed without killing the manager; importing `ariel` sets `MUJOCO_GL` from the file). Order: after test-env-green (both touch the rendering/cv2 environment). | draft | 2026-10-17 |

## Current state

Phase: planning. Goals drafted (five), awaiting approval. Next step: user approves the goals in words (libGL and entry-point questions are answered).

## Baseline

Brownfield. Measured 2026-10-17 after `uv sync --all-groups` (Python 3.12.9, Linux, `uv run --no-sync`):

| Check | Result | Time |
|---|---|---|
| `uv lock --check` | pass | ~2 s |
| `ruff check --no-fix src tests` | fail: 1858 errors (461 auto-fixable); top: ANN001 331, E501 281, W505 105 | <1 s |
| `ruff format --check src tests` | fail: 51 files would be reformatted, 96 formatted | <1 s |
| `mypy src` (strict) | fail: 517 errors in 49 of 102 files | ~40 s |
| `pytest tests/unit` | 419 passed, 2 collection errors (see Known problems) | 6.6 s (3 runs: 6.58, 6.59, 6.60) |
| coverage, same run | 5033 statements, 2760 missed (about 45% line coverage; `fail_under` is 100) | with `--cov` ~10 s |

## Tasks

None yet.

## Open questions

- gl-autodetect: can we test on Windows or macOS?
- gl-autodetect: selection rule, proposed by the user, not yet decided. A GUI is a requirement of the run, not a preference: the interactive viewer needs glfw, batch and evolution runs do not. Choose among working backends by what the run needs first, then by measured render time. Order for viewing: glfw, egl, osmesa. Order for headless batch: egl, glfw, osmesa.

Evidence for gl-autodetect (checked 2026-10-17):
- Nothing in `src/`, `examples/` or `tests/` sets `MUJOCO_GL` or `PYOPENGL_PLATFORM`, and `mujoco_user_config.json` does not exist.
- `src/ariel/utils/renderers.py` constructs `Renderer(` in 3 places; 22 `mujoco.Renderer(` / `mj.Renderer(` calls in `examples/` (the request said about 15).
- `examples/re_book/1_brain_evolution_multithreaded.py` line 366 documents a crash: an access violation in `glfwCreateWindow` when `mujoco.Renderer()` is created in a worker thread. So each backend must be probed in its own subprocess.
- `MUJOCO_GL` must be set before `mujoco` is imported, so the config would be applied early; `ariel/__init__.py` already sets global state at import.

## Known problems

Commands from config and CI. Ran on 2026-10-17 (results in Baseline); not yet moved into CLAUDE.md. Not run: sphinx; pydoclint and pre-commit are not installed.
- Fast tests: `uv run pytest tests/unit` (pytest, pytest-cov, pytest-modern; coverage `fail_under = 100`).
- Lint: `uv run ruff check --no-fix .` and `uv run ruff format --check .`.
- Types: `uv run mypy src` (strict); `pyrightconfig.json` also exists.
- Docstrings: `uv run pydoclint` (pre-commit hook).
- Docs: `uv run sphinx-build -b html docs docs/_build` (CI, `.github/workflows/sphinx.yml`) and `nox -s docs` (live reload).
- `nox --session=tests` and `nox --session=pre-commit` are named in `.claude/dev/testing_and_tooling.md`, but `noxfile.py` defines only a `docs` session.

Other problems:
- `tests/unit/test_simulation/test_environments/test_environments.py` and `tests/unit/test_utils/test_tracker.py` fail at collection: `import cv2` raises `libGL.so.1: cannot open shared object file` (system library missing on this machine; the dependency is `opencv-python`, not headless). Their tests are not in the 419.
- `uv run ariel` fails: `src/ariel/__main__.py` is missing.
- `.pre-commit-config.yaml` hooks use `language: system`, but `pre-commit`, `pydoclint` and the hook tools are in no dependency group, so the hooks cannot run from the venv (checked 2026-10-17).
- No tests for: `visualisation/`, `body_phenotypes/lynx_mjspec/`, `simulation/mujoco_worker.py`, `simulation/example_usage.py`, `utils/renderers.py`, `utils/video_recorder.py`, `utils/morphological_descriptor.py`, `utils/noise_gen.py`. Behaviour of `examples/` is pinned by nothing; `tests/functional/` holds only notebooks. (From the file layout; not confirmed by coverage.)
- Not-owned files: none; the tree was clean.
