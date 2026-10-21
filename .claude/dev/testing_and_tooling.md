# Testing & tooling

ARIEL uses **uv** for environments and docs builds, **pytest** for
tests, and a **pre-commit** config for commit-time gates. Source of truth:
[pyproject.toml](../../pyproject.toml),
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
```

- Tests live in [tests/](../../tests/), written with `pytest`
  (`pytest`, `pytest-cov`, `pytest-modern`).
- **Coverage target is 100%**, `pyproject.toml` sets
  `[tool.coverage.report] fail_under = 54`, the measured value, so the suite
  passes; raise it as tests are added. New code needs new tests.
  Excluded lines: `pragma: no cover`, `if TYPE_CHECKING:`.
- Asserts are allowed in test files (ruff `S101` carve-out for `tests/`,
  `test_*.py`).

When adding a feature, add the test in the matching `tests/…` path **in the same
change**, don't defer it. CI runs the tests, ruff and mypy
([.github/workflows/checks.yml](../../.github/workflows/checks.yml)) and builds
the docs ([.github/workflows/sphinx.yml](../../.github/workflows/sphinx.yml)).

## Pre-commit

`pre-commit` is in the dev group. Run `uv run --no-sync pre-commit run
--all-files`.

The hooks (see [.pre-commit-config.yaml](../../.pre-commit-config.yaml)) run
`ruff check` + `ruff format` on `src/` and `tests/`, plus large-file, TOML,
YAML, end-of-file and whitespace checks. Code that isn't ruff-clean will be
rejected at commit time; write it to standard up front (see
[coding_standards.md](coding_standards.md)). `pydoclint` is installed but not a
hook yet.
