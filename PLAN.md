# PLAN

## Goals

| Name | Goal | Kind | Success measure | Status | Added |
|---|---|---|---|---|---|
| test-env-green | All unit tests can be collected and pass | improvement, behaviour unchanged | `uv run pytest tests/unit` reports 0 collection errors and 0 failures (today 2 collection errors, 419 passed) | approved | 2026-10-17 |
| dashboards-out | Move the dashboards off `main` into a `feature-dashboard` branch | improvement, behaviour unchanged for the core | `feature-dashboard` branch holds the current `main`; on `main`, `src/ariel/visualisation/dashboard/` is gone, references in docs/README/.claude updated, dependencies used only by dashboards (panel, nicegui, plotly, kaleido, pyvista: none imported elsewhere in `.py` files) removed or listed as a decision; `pytest tests/unit` result is unchanged | approved | 2026-10-17 |
| repo-hygiene | Remove the broken entry point, the stray file and the stale docs | improvement, behaviour unchanged | the `[project.scripts]` entry is removed from `pyproject.toml` (decided 2026-10-17; `ariel.__main__` does not exist); `src/ariel/ec/generators copy.py` removed (done 2026-10-17, user approved; unreferenced); `CONTRIBUTING.md` says `uv sync` and Python >= 3.12; placeholder `TODO(jmdm)` docstrings count in `src/` goes from 19 to 0 or each is listed as kept | approved | 2026-10-17 |
| lint-types-policy | Decide which ruff/mypy/pyright/pre-commit rules are sensible, then make the code meet them | improvement | pending: measure set after the config review with the user (baseline: ruff 1858 findings, format 51 files, mypy 517 errors) | draft, blocked on config review | 2026-10-17 |
| gl-autodetect | ARIEL never crashes because the user's OpenGL backend is unsuitable. A manager script runs one subprocess worker per backend (glfw, egl, osmesa), records which work and how fast in `mujoco_user_config.json`, and ARIEL reads that file at import to set `MUJOCO_GL`. | new behaviour | not set yet; acceptance criteria to be proposed and agreed (candidates: the manager writes a valid file on this Linux machine; a crashing backend is recorded as failed without killing the manager; importing `ariel` sets `MUJOCO_GL` from the file). Order: after test-env-green (both touch the rendering/cv2 environment). | draft | 2026-10-17 |

## Current state

Phase: planning. Three goals approved 2026-10-17 (test-env-green, dashboards-out, repo-hygiene); lint-types-policy and gl-autodetect still draft. Task list below awaits approval. Next step: user approves the task list in words; no code changes before that.

## Baseline

Brownfield. Measured 2026-10-17 after `uv sync --all-groups` (Python 3.12.9, Linux, `uv run --no-sync`):

| Check | Result | Time |
|---|---|---|
| `uv lock --check` | pass | ~2 s |
| `ruff check --no-fix src tests` | fail: 1858 errors (461 auto-fixable); top: ANN001 331, E501 281, W505 105 | <1 s |
| `ruff format --check src tests` | fail: 51 files would be reformatted, 96 formatted | <1 s |
| `mypy src` (strict) | fail: 517 errors in 49 of 102 files; after T4: 262 errors in 44 of 96 files | ~40 s |
| `pytest tests/unit` | 419 passed, 2 collection errors (before T3); after T3: 437 passed, 0 errors | 6.6 s (3 runs: 6.58, 6.59, 6.60) |
| coverage, same run | 5033 statements, 2760 missed (about 45% line coverage; `fail_under` is 100) | with `--cov` ~10 s |

## Tasks

Specs and measurements are in DESIGN.md (Specs). No task changes behaviour except removing the dashboards and the `ariel` console script, both approved. Every task: tests before and after (`pytest` line in CLAUDE.md), one task per commit, PLAN.md updated in the same commit. Pytest count and the grep counts are deterministic (no noise); pytest time was 6.58, 6.59, 6.60 s over three runs, so only a change over 0.1 s would count, and no goal claims a time.

Order: Tier 1 first, in the order listed. T1 to T4 done (see DESIGN.md Results). T4 is done; T5 and T6 follow it. gl-autodetect no longer waits (T3 done).

### Tier 1: small diff, low risk

### Tier 2: clear effect, more files or medium risk

**T5. Drop dependencies used only by the dashboards** (dashboards-out)
- Changes: `pyproject.toml`, `uv.lock`. Candidates: `panel`, `plotly` (imported only by dashboards); `nicegui`, `kaleido`, `pyvista` (imported nowhere in code, checked in `.py`, `.ipynb`, `.md`). `pandas` and `json5` are imported by the dashboards but declared nowhere (transitive). Each one removed only after you confirm by name.
- Effect: smaller `.venv` (9.3 GB today; measured after), shorter `uv sync`. Estimate: unknown until measured; torch dominates.
- Risk medium: a notebook or wiki code sample could use one (grep found none); `matplotlib`/`pandas` users may depend on them transitively.
- Cost: -5 lines in `pyproject.toml`, lockfile churn.
- Verify: pytest unchanged, `uv sync --all-groups` clean, `uv lock --check`, mypy error count not higher than 517 (T4 may lower it).

**T6. Replace placeholder docstrings** (repo-hygiene)
- Changes: the `TODO(jmdm): description of script` module docstrings in 17 files (recount after T4; list from `grep -rn "description of script" src`). Each gets a one-line real description read from the module. The single real TODO in `src/ariel/utils/morphological_descriptor.py:664` (normaliser check) is kept and listed, because it is a correctness question, not a placeholder.
- Effect: count of placeholder TODOs in `src/`: 17 files to 0 (the real TODO at `morphological_descriptor.py:664` stays).
- Risk low (docstrings only). Cost: about +40 lines of prose.
- Verify: `grep -rn "description of script" src` empty; `ruff check --no-fix` count not higher than before; pytest unchanged.

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
