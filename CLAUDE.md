# ARIEL

Read PLAN.md first, then DESIGN.md, CODEMAP.md and the README of the module you will change. If a file and the code disagree, trust the code, fix the file, and say what was wrong. If a file and the user's instructions disagree, stop and ask.

## Where facts live

| File | Holds |
|---|---|
| CLAUDE.md | rules and commands for every session in this repo |
| PLAN.md | state of the work; goes stale as it progresses |
| CODEMAP.md | structure of the code, regenerable from it; never plans, decisions or measurements |
| DESIGN.md | reasoning and reference that stays true after a task is done |

Test: the code could regenerate it → CODEMAP.md. Stale once the work is done → PLAN.md. A rule for every session here → CLAUDE.md. Otherwise → DESIGN.md.

`.claude/dev/`, `.claude/user/` and `.claude/wiki/` are existing hand-written developer docs, user docs and a knowledge wiki sorted into topic folders (`.claude/SCHEMA.md` explains it; `tests/unit/test_wiki_structure.py` checks it). Link to them; do not copy them.

## Commands

From the repo root. Python 3.12 (`.python-version` 3.12.9), uv 0.13.1.

| Purpose | Command | Checked |
|---|---|---|
| Install everything | `uv sync --all-groups` | ~5 min first time, ~8.4 GB `.venv` |
| Lockfile matches pyproject | `uv lock --check` | passes, ~2 s, read-only |
| Unit tests | `uv run --no-sync pytest tests/unit -q -p no:cacheprovider` | 981 passed, ~37 s wall on 2026-10-18 (229 of them wiki structure cases; two tests start a process pool, ~15 s); for the gate's coverage run `--cov=src --cov=tests --cov-branch` (95.1% today, `fail_under = 54` in `pyproject.toml`, target 100, see PLAN.md GS8) |
| Lint | `uv run --no-sync ruff check src tests` | passes (src, tests; `ruff.toml` excludes `examples/`, `docs/` and `.claude/wiki`, `mypy.ini` excludes `examples/` and `docs/`), <1 s, read-only (`fix = false`); `ruff.toml` is the user's template (ruff 0.17, `preview = true`, banned-api for `Any`, `dataclass`, `NamedTuple`, `TypedDict`) with py312, those two exclusions and `fix = false` |
| Format check | `uv run --no-sync ruff format --check src tests` | passes (150 files), <1 s, read-only; also covers notebooks under `tests/`; ruff 0.17 also formats the code blocks in markdown (`.claude/`, README) |
| Types (mypy) | `uv run --no-sync mypy src` | passes (101 files), ~40 s, read-only; `mypy.ini` is the user's template (`disallow_any_explicit`, `pydantic.mypy` plugin, `follow_untyped_imports`) with py312 and `mujoco` kept untyped; 87 lines carry `# type: ignore[...]  # known bug`, listed in PLAN.md |
| Whole-repo gate commands | `uv run --no-sync ruff format --check . && uv run --no-sync ruff check --no-fix . && uv run --no-sync mypy .` | pass (293 files, 144 mypy sources) on 2026-10-18; the user's global Python gate runs these plus `pytest -q`; `examples/` and `docs/` are excluded in `ruff.toml` and `mypy.ini`; the gate still lists the missing coverage floor of 100 (PLAN.md, GS8) |
| Probe OpenGL backends | `uv run --no-sync python -m ariel.utils.gl_manager` | writes `mujoco_user_config.json` in the current folder, ~2 s here (all three backends fail on this machine); rerun after a driver change |
| Pre-commit (all hooks) | `uv run --no-sync pre-commit run --all-files` | passes; first run clones `pre-commit/pre-commit-hooks` into `~/.cache/pre-commit`; the whitespace/EOF hooks edit files, ruff hooks cover `src` and `tests` only |
| Types (pyright) | `uv run --no-sync pyright src tests` | fails today (211 errors), ~15 s, read-only; first run downloads Node into `~/.cache/pyright-python` (229 MB) |

Ruff is pinned to 0.17.* and does not fix by default; `ruff check --fix` is opt-in. Pytest with `--cov=ariel` adds ~4 s and writes `.coverage`. Failing results are listed in PLAN.md.

## Code rules

- Package `ariel` in `src/ariel/`; entry points are the modules and `examples/`. Imports point down: `ec` → `simulation`, `body_phenotypes`, `utils` → `parameters`; `parameters` imports no other ariel package. Nothing in `src/` imports from `examples`.
- Reach another module only through its package `__init__.py` exports.
- No compatibility shims. No abstraction before two real call sites. Clear names over clever one-liners. No options nobody asked for.
- House style (strict ruff `select = ["ALL"]`, 80 columns, NumPy docstrings, strict mypy, 100% coverage target) is in `.claude/dev/`; the config files win over those docs.

## Git and approvals

- Git has no `user.name` or `user.email` set in this repo. Commit with `git -c user.name=jmdm -c user.email=jacopo.michele.dimatteo@gmail.com commit ...` and do not change the git config. The user confirmed the name `jmdm` on 2026-10-17.
- Merge as the global file says (rebase, then `git merge --ff-only`). `main` was rebased to a linear history on 2026-10-17. The old tip with the merge commits is no longer tagged (the tag `backup/pre-linear` was deleted on 2026-10-18).
- Library behaviour changes only with the user's explicit approval. A lint or type fix that would change behaviour gets a line-level `# noqa: CODE` or `# type: ignore[code]  # known bug` and an entry in PLAN.md Known problems instead of a fix.
- Push and branch deletion wait for the user's say-so, as the global file says. Pushes go to `fork`, not `origin`. `main` is on `fork` (pushed 2026-10-17); it is ahead of `origin/main` (`git rev-list --count origin/main..main`, about 84); `fork` also holds `feature-dashboard` and `hardware`, and `main` there is behind the local one (`git rev-list --count fork/main..main`).
- Remotes: `origin` is `ci-group/ariel` (not ours to push to); `fork` is `git@github.com:Jacopo-DM/ariel.git`, the user's GitHub fork (created 2026-10-17), meant for experiments away from `main`; it holds `main`, `feature-dashboard`, `hardware` and `gh-pages`. `gh` is installed on this Linux machine and logged in as `Jacopo-DM`.

## Data and secrets

- Run outputs go to `__data__/` under the current working directory (`ariel.DATA`, git-ignored). Tests and dev tools must not write anywhere else or read other users' data.
- `mujoco_user_config.json` (and a `.lock` file while probing) is written in the current working folder by `ariel.utils.gl_manager`; it is git-ignored and machine-specific. `MUJOCO_GL`, if set, always wins over it. Tests preset `MUJOCO_GL=glfw` in `tests/conftest.py` so they never probe.
- No keys or environment variables are used, except `ARIEL_COMPILE_MYPYC=1` to opt in to the mypyc build (`setup.py`).

## Ignore list

| Path | Kind | What makes it |
|---|---|---|
| `__data__/` | generated | runs and examples |
| `.venv/` | generated | `uv venv` / `uv sync` |
| `.typings/` | hand-written stubs, 17 MB | mujoco stubs, tracked in git |
| `docs/_build/`, `docs/autoapi/`, `docs/apidocs/` | generated | sphinx |
| `uv.lock` | generated | `uv lock` |
| `3_spider_vision_new.npy` | binary, stray | tracked at the repo root; origin not recorded |
| `.claude/wiki/` | generated by the `/ingest` skill from source docs | LLM-maintained wiki |
| `mujoco_user_config.json`, `mujoco_user_config.json.lock` | generated, git-ignored | `ariel.utils.gl_manager` / `ensure_gl_backend` in the working folder |
| `.coverage` | generated, git-ignored | `pytest --cov` |

`.claude/settings.json` has no ignore or deny list and I left it unchanged. Its allow list holds Windows PowerShell entries (`c:\\Users\\John\\...`) from another developer; they do nothing here.

## Things that bite

- `ruff check --fix` is opt-in (config has `fix = false`), and the pre-commit ruff hook does not pass it either. The whitespace and end-of-file hooks do edit files.
- Suppressions use ruff 0.17 syntax, not `# noqa`: `# ruff: ignore[rule-name]  # reason` on the line, `# ruff: file-ignore[rule-name]  # reason` for a file; ruff reports `# noqa` (RUF105). Names, not codes (RUF201).
- `ruff check --select X --fix` with a narrow `--select` makes `RUF100` flag valid `# noqa` comments for the rules you did not select (it removed three needed `E712` ones). Never select `RUF100` alone; check with the full rule set.
- mypy only honours `# type: ignore[code]` when it is the first comment on the line (`  # type: ignore[code]  # noqa: X  # known bug`), and `ruff format` can move a trailing comment to another line when it re-wraps. Run mypy after formatting.
- `MUJOCO_GL` is read when `mujoco` is first imported. Code that decides the backend must run before that import (`ariel/__init__.py`, the top of `ariel/utils/renderers.py`, which is why it carries `# isort: split` and `# ruff: noqa: E402`). Do not import `mujoco` or any module that does above them.
- `CONTRIBUTING.md` (root) and `docs/source/contributing_guide/CONTRIBUTING.md` still tell you to run `nox --session=tests` / `pre-commit`; `noxfile.py` only has `docs`.
