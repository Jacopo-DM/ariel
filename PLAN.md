# PLAN

## Goals

No goals yet: run /zz-project-plan.

## Current state

Phase: setup done, no plan. Next step: run /zz-project-plan.

## Baseline

Brownfield. Not measured: there is no `.venv`, and installing dependencies (`uv sync`) was out of scope for setup.
Measured on 2026-10-17: `uv lock --check` passes (480 packages, ~2 s).

## Tasks

None yet.

## Open questions

None yet.

## Known problems

Commands taken from config and CI but not run (no venv). Run them after `uv sync` and move the ones that pass into CLAUDE.md:
- Fast tests: `uv run pytest tests/unit` (pytest, pytest-cov, pytest-modern; coverage `fail_under = 100`).
- Lint: `uv run ruff check --no-fix .` and `uv run ruff format --check .`.
- Types: `uv run mypy src` (strict); `pyrightconfig.json` also exists.
- Docstrings: `uv run pydoclint` (pre-commit hook).
- Docs: `uv run sphinx-build -b html docs docs/_build` (CI, `.github/workflows/sphinx.yml`) and `nox -s docs` (live reload).
- `nox --session=tests` and `nox --session=pre-commit` are named in `.claude/dev/testing_and_tooling.md`, but `noxfile.py` defines only a `docs` session.

Other problems:
- `uv run ariel` fails: `src/ariel/__main__.py` is missing.
- `src/ariel/ec/generators copy.py` has a space in the filename and looks like a stray copy of `generators.py`.
- `CONTRIBUTING.md` is out of date (`uv install`, Python 3.9+, and the source/issue links point to a personal fork).
- `.pre-commit-config.yaml` hooks use `language: system`, so they need the tools on PATH.
- No tests for: `visualisation/`, `body_phenotypes/lynx_mjspec/`, `simulation/mujoco_worker.py`, `simulation/example_usage.py`, `utils/renderers.py`, `utils/video_recorder.py`, `utils/morphological_descriptor.py`, `utils/noise_gen.py`. Behaviour of `examples/` is pinned by nothing; `tests/functional/` holds only notebooks. (From the file layout; not confirmed by coverage.)
- Not-owned files: none; the tree was clean.
