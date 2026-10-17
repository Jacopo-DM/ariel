# PLAN

## Goals

| Name | Goal | Kind | Success measure | Status | Added |
|---|---|---|---|---|---|
| test-env-green | All unit tests can be collected and pass | improvement | done 2026-10-17: commits matching `git log --grep '^T3:'`; pytest 419 passed + 2 collection errors -> 437 passed, 0 errors | done | 2026-10-17 |
| dashboards-out | Move the dashboards off `main` into a `feature-dashboard` branch | improvement | done 2026-10-17: commits matching `git log --grep '^T[45]:'`; measure: dashboard dir gone from main, branch holds it, pytest 437 passed unchanged | done | 2026-10-17 |
| repo-hygiene | Remove the broken entry point, the stray file and the stale docs | improvement | done 2026-10-17: commits matching `git log --grep '^T[126]:'` and the housekeeping merges; console script, `generators copy.py`, placeholder docstrings (18 files to 0) and stale dev-doc claims fixed. Left: nox session names in root and docs CONTRIBUTING.md (see Known problems) | done | 2026-10-17 |
| lint-types-policy | Decide which ruff/mypy/pyright/pre-commit rules are sensible, then make the code meet them | improvement | pending: measure set once the rule-group walkthrough ends (baseline today: ruff 1330 findings with current config, 1146 with lint preview off; format 47 files; mypy 262 errors) | draft, config review in progress (defaults, type checkers and enforcement decided; rule groups open) | 2026-10-17 |
| gl-autodetect | ARIEL never crashes because the user's OpenGL backend is unsuitable. A manager script runs one subprocess worker per backend (glfw, egl, osmesa), records which work and how fast in `mujoco_user_config.json`, and ARIEL reads that file at import to set `MUJOCO_GL`. | new behaviour | not set yet; acceptance criteria to be proposed and agreed (candidates: the manager writes a valid file on this Linux machine; a crashing backend is recorded as failed without killing the manager; importing `ariel` sets `MUJOCO_GL` from the file). Order: after test-env-green (both touch the rendering/cv2 environment). | draft | 2026-10-17 |

## Current state

Phase: planning. test-env-green, dashboards-out and repo-hygiene are done. lint-types-policy and gl-autodetect are still draft. Next step: the config review with the user (lint-types-policy), then gl-autodetect acceptance criteria.

## Baseline

Brownfield. Measured 2026-10-17 after `uv sync --all-groups` (Python 3.12.9, Linux, `uv run --no-sync`):

| Check | Result | Time |
|---|---|---|
| `uv lock --check` | pass | ~2 s |
| `ruff check --no-fix src tests` | fail: 1858 errors; after T6: 1330 (461 auto-fixable); top: ANN001 331, E501 281, W505 105 | <1 s |
| `ruff format --check src tests` | fail: 51 files would be reformatted, 96 formatted; after T6: 47 and 94 | <1 s |
| `mypy src` (strict) | fail: 517 errors in 49 of 102 files; after T4: 262 errors in 44 of 96 files | ~40 s |
| `pytest tests/unit` | 419 passed, 2 collection errors (before T3); after T3: 437 passed, 0 errors | 6.6 s (3 runs: 6.58, 6.59, 6.60) |
| coverage, same run | 5033 statements, 2760 missed (about 45% line coverage; `fail_under` is 100) | with `--cov` ~10 s |

## Tasks

Specs and measurements are in DESIGN.md (Specs). No task changes behaviour except removing the dashboards and the `ariel` console script, both approved. Every task: tests before and after (`pytest` line in CLAUDE.md), one task per commit, PLAN.md updated in the same commit. Pytest count and the grep counts are deterministic (no noise); pytest time was 6.58, 6.59, 6.60 s over three runs, so only a change over 0.1 s would count, and no goal claims a time.

Order: Tier 1 first, in the order listed. T1 to T6 done (see DESIGN.md Results). gl-autodetect no longer waits (T3 done).

### Tier 1: small diff, low risk

### Tier 2: clear effect, more files or medium risk

None left.

### Tier 3: left alone on purpose (not executed)

- Raise unit coverage from about 45% to the 100% `fail_under`: outside the three approved goals; large; belongs with lint-types-policy.
- Characterisation tests for the dashboards: they leave `main`; revisit on `feature-dashboard`.
- Fix the 1858 ruff findings, 51 unformatted files and 517 mypy errors: goal lint-types-policy, blocked on the config review.
- Install `pre-commit` and `pydoclint` so the hooks run: part of the config review.
- Resolve the `morphological_descriptor.py:664` normaliser TODO: needs a behaviour decision.
- Push `feature-dashboard` or `main` to origin: needs your say-so.

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
- `examples/a_mujoco/3_cameras_in_simulation.py` was changed in T3 (cv2 window replaced by matplotlib) but not run: this machine has no display, and the script fails earlier at `mujoco.Renderer(...)` (`AttributeError: ... _mjr_context`, no OpenGL context). Needs a manual run on a machine with a display. The Renderer failure is further evidence for gl-autodetect.
- Root `CONTRIBUTING.md` (lines 80, 103) and `docs/source/contributing_guide/CONTRIBUTING.md` (line 228) still name nox sessions `tests` and `pre-commit` that `noxfile.py` does not define. Not in T2's file list; add to a task if wanted.
- `.pre-commit-config.yaml` hooks use `language: system`, but `pre-commit`, `pydoclint` and the hook tools are in no dependency group, so the hooks cannot run from the venv (checked 2026-10-17).
- No tests for: `visualisation/`, `body_phenotypes/lynx_mjspec/`, `simulation/mujoco_worker.py`, `simulation/example_usage.py`, `utils/renderers.py`, `utils/video_recorder.py`, `utils/morphological_descriptor.py`, `utils/noise_gen.py`. Behaviour of `examples/` is pinned by nothing; `tests/functional/` holds only notebooks. (From the file layout; not confirmed by coverage.)
- Not-owned files: none; the tree was clean.
