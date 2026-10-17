# Testing & tooling

ARIEL uses **uv** for environments, **nox** for docs builds, **pytest** for
tests, and a **pre-commit** config for commit-time gates. Source of truth:
[noxfile.py](../../noxfile.py), [pyproject.toml](../../pyproject.toml),
[.pre-commit-config.yaml](../../.pre-commit-config.yaml),
[CONTRIBUTING.md](../../CONTRIBUTING.md).

## Environment

`pyproject.toml` requires **Python ≥ 3.12**. Set up with uv:

```bash
uv venv
uv sync
uv run examples/re_book/1_brain_evolution.py   # run anything via `uv run`
```

Use `uv sync --all-groups` to include the dev and docs groups.

## Tests & coverage

```bash
uv run pytest tests/unit                  # unit tests
uv run pytest tests/unit --cov=ariel      # with coverage
nox --list                                # sessions: only `docs` exists today
```

- Tests live in [tests/](../../tests/), written with `pytest`
  (`pytest`, `pytest-cov`, `pytest-modern`).
- **Coverage target is 100%** — `pyproject.toml` sets
  `[tool.coverage.report] fail_under = 100`. Measured coverage is about 45%
  today (see PLAN.md), so the threshold is not met. New code needs new tests.
  Excluded lines: `pragma: no cover`, `if TYPE_CHECKING:`.
- Asserts are allowed in test files (ruff `S101` carve-out for `tests/`,
  `test_*.py`, `noxfile.py`).

When adding a feature, add the test in the matching `tests/…` path **in the same
change** — don't defer it. No CI job runs the tests; the only workflow builds
the docs ([.github/workflows/sphinx.yml](../../.github/workflows/sphinx.yml)).

## Pre-commit

There is no nox session for pre-commit, and `pre-commit` and `pydoclint` are
in no dependency group, so the hooks cannot be installed from the venv today.

The hooks (see [.pre-commit-config.yaml](../../.pre-commit-config.yaml)) run
`ruff check` + `ruff format`, `pydoclint` (NumPy docstring linting), `prettier`,
and TOML/YAML validators. Code that isn't ruff-clean and numpydoc-compliant will
be rejected at commit time — write it to standard up front (see
[coding_standards.md](coding_standards.md)).

## Optional: mypyc compiled build

ARIEL can be compiled with `mypyc` for speed (opt-in, off by default). Triggered
by the `ARIEL_COMPILE_MYPYC=1` env var via [setup.py](../../setup.py). No nox
session builds it. You won't need this for normal development.
