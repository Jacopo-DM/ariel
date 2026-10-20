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
| Install everything | `uv sync --all-groups` | ~5 min first time, ~7.8 GB `.venv` (2026-10-20, after the dependency clean-up) |
| Lockfile matches pyproject | `uv lock --check` | passes, ~2 s, read-only |
| Unit tests | `uv run --no-sync pytest tests/unit -q -p no:cacheprovider` | 1233 passed of 1236 run (3 warnings), ~28 s wall on 2026-10-19 (231 of them wiki structure cases; two tests start a process pool, ~15 s); plain `uv run --no-sync pytest -q` also runs the coverage from the template `pytest.ini` (`--cov=src --cov=tests --cov-branch --cov-fail-under=100`, 100% on 2026-10-19) |
| Lint | `uv run --no-sync ruff check src tests` | passes (src, tests; `ruff.toml` excludes `examples/`, `docs/` and `.claude/wiki`, `mypy.ini` excludes `examples/` and `docs/`), <1 s, read-only (`fix = false`); `ruff.toml` is the user's template (ruff 0.17, `preview = true`, banned-api for `Any`, `dataclass`, `NamedTuple`, `TypedDict`) with py312, those two exclusions, `fix = false` and `assert` allowed in `tests/**/*.ipynb` (added 2026-10-19 with the user's approval, for the functional notebooks) |
| Format check | `uv run --no-sync ruff format --check src tests` | passes (228 files in src and tests, 264 for `.`, 2026-10-20), <1 s, read-only; also covers notebooks under `tests/`; ruff 0.17 also formats the code blocks in markdown (`.claude/`, README) |
| Types (mypy) | `uv run --no-sync mypy src` | passes (124 files, 2026-10-20), ~40 s, read-only; `mypy.ini` is the user's template (`disallow_any_explicit`, `pydantic.mypy` plugin, `follow_untyped_imports`) with py312 and `mujoco` kept untyped; 52 lines carry `# type: ignore[...]  # known bug`, listed in PLAN.md |
| Whole-repo gate commands | `uv run --no-sync ruff format --check . && uv run --no-sync ruff check --no-fix . && uv run --no-sync mypy .` | pass (263 files for ruff, 215 mypy sources) on 2026-10-20; the user's global Python gate runs these plus `pytest -q`; the hook now drops files that `mypy.ini` excludes before it calls mypy; `examples/` and `docs/` are excluded in `ruff.toml` and `mypy.ini` |
| MuJoCo settings report | `uv run --no-sync python -m ariel.simulators.mujoco [--out FILE]` | prints the derivation table, `10 of 156 differ` and the band count, <1 s here (2026-10-19); `--out` writes the MJCF file (only the differences); `--write-docs` rewrites `docs/source/mujoco_parameters.md`, which a test checks, so run it after a change of seeds, bands, settings or MuJoCo version |
| Structure numbers | `uv run --no-sync python tools/measure_structure.py` | read-only, <1 s; prints, for `src/ariel` and `tests`, files, lines, files over 400 lines, modules importing more than 5 others, and (src only) imports that skip a package `__init__`; baseline in PLAN.md |
| Floor pictures | `uv run --no-sync python tools/render_floors.py` | writes one PNG per world into `__data__/floors/` and prints the paths; needs a working OpenGL backend; the script imports `mujoco` before `ariel`, so MuJoCo ignores ariel's choice and the plain command fails on this machine (glfw, no display, `FatalError`): run it as `MUJOCO_GL=osmesa uv run --no-sync python tools/render_floors.py` (~3 s, works 2026-10-20; faked in `tests/unit/test_tools/test_render_floors.py`) |
| Run notebooks | `uv run --no-sync python tools/run_notebooks.py [FILE ...] [--slow] [--timeout SECONDS] [--save]` | one line per notebook (PASS, FAIL, SKIP), exit 1 on a FAIL; no arguments runs `tests/functional/*.ipynb` except those with `"slow": true` in their metadata (`speed_tests`, about 105 s); a notebook with `"requires": ["opengl"]` is SKIP without a working backend; `--save` writes the outputs into the notebooks; `cpg` passes in ~9 s here (2026-10-19) |
| Body overlap | `uv run --no-sync python tools/measure_overlap.py` | read-only, ~8 s; prints, for 60 bodies from each of three random sources (seed 42), how many have two non-adjacent modules penetrating deeper than 1% of the smaller module's thinnest side, measured with MuJoCo's `mj_geomDistance`, before and after `construct_mjspec_from_graph(graph, repair_overlaps=True)`, plus build and repair times; 30, 51, 32 overlapping bodies before and 0, 0, 0 after (goal no-overlap) |
| Benchmark | `uv run --no-sync python tools/benchmark.py [WORKLOAD ...] [--runs N]` | read-only, writes nothing in the repo; workloads `evolution`, `simulation`, `bodies`, `tests` (goal speed-up; definitions in DESIGN.md Specs); prints median and range of wall-clock time, median CPU time and median peak memory per workload; one run of each checked on 2026-10-20 (evolution 16.6 s, simulation 0.24 s at 5000 steps, bodies 2.2 s, tests about 62 s after speed-up S1 and S2), the 10-run table is in PLAN.md Baseline (about 25 minutes for all four workloads); run nothing else while it runs (DESIGN.md Gotchas) |
| Duplicated code | `npx --yes jscpd@5.4.1 --min-lines 8 --reporters console --output <scratch dir> src` | read-only, 25 ms; needs Node 22 (installed in `/usr/bin`); downloads a 46 MB Rust binary into `~/.npm` the first time; 4.64% of `src` lines on 2026-10-19; not part of any gate |
| Probe OpenGL backends | `uv run --no-sync python -m ariel.opengl.gl_manager` | writes `mujoco_user_config.json` in the current folder, ~2 s here (2026-10-20: egl and osmesa work, glfw fails, osmesa selected; before, all three failed); rerun after a driver change |
| Pre-commit (all hooks) | `uv run --no-sync pre-commit run --all-files` | passes; also runs the notebook hooks (`notebooks-commit` on the committed notebooks, `notebooks-push` at push only; install the push hook with `uv run --no-sync pre-commit install --hook-type pre-commit --hook-type pre-push`); first run clones `pre-commit/pre-commit-hooks` into `~/.cache/pre-commit`; the whitespace/EOF hooks edit files, ruff hooks cover `src` and `tests` only; the hooks are installed in this clone since 2026-10-20 (`.git/hooks/` is not in git, so a fresh clone must run the install command), every commit then runs the quick ones and every push the fast notebook set; the notebooks step prints nothing for about 90 s, do not interrupt it |
| Types (pyright) | `uv run --no-sync pyright src tests` | fails today (182 errors on 2026-10-20), ~15 s, read-only; first run downloads Node into `~/.cache/pyright-python` (229 MB) |

Ruff is pinned to 0.17.* and does not fix by default; `ruff check --fix` is opt-in. Pytest with `--cov=ariel` adds ~4 s and writes `.coverage`. Failing results are listed in PLAN.md.

## Code rules

- Package `ariel` in `src/ariel/`; entry points are the modules and `examples/`. Imports point down: `ec` → `simulation`, `body_phenotypes`, `rendering` → `opengl`; `simulation` may import `simulators` (back-end settings generators); `body_phenotypes`, `simulation`, `simulators` and `ec` → `parameters`; `opengl` and `parameters` import no other ariel package. Nothing in `src/` imports from `examples`.
- Reach another module only through its package `__init__.py` exports. Outside a top-level package, import only from `ariel.<package>` (for example `from ariel.simulation import Tracker`), never from a module below it; `tools/measure_structure.py` counts the exceptions in `src` (0 on 2026-10-19). Tests may import any module.
- Every package and sub-package of `src/ariel` has a row in `tests/functional/README.md`, with its functional notebook or `none yet`; add the row in the same commit as a new area, and add a notebook where the area can be seen working (`tests/unit/test_functional_notebooks.py` fails when a row is missing). A notebook closes each section with plain `assert` lines, imports `ariel.rendering` before `mujoco` if it renders, and carries `"requires": ["opengl"]` (or an in-notebook guard) when it needs a backend.
- No compatibility shims. No abstraction before two real call sites. Clear names over clever one-liners. No options nobody asked for.
- House style (strict ruff `select = ["ALL"]`, 80 columns, NumPy docstrings, strict mypy, 100% coverage target) is in `.claude/dev/`; the config files win over those docs.

## Git and approvals

- Git has no `user.name` or `user.email` set in this repo. Commit with `git -c user.name=jmdm -c user.email=jacopo.michele.dimatteo@gmail.com commit ...` and do not change the git config. The user confirmed the name `jmdm` on 2026-10-17.
- Merge as the global file says (rebase, then `git merge --ff-only`). `main` was rebased to a linear history on 2026-10-17. The old tip with the merge commits is no longer tagged (the tag `backup/pre-linear` was deleted on 2026-10-18).
- Library behaviour changes only with the user's explicit approval. A lint or type fix that would change behaviour gets a line-level `# noqa: CODE` or `# type: ignore[code]  # known bug` and an entry in PLAN.md Known problems instead of a fix.
- Push and branch deletion wait for the user's say-so, as the global file says. Pushes go to `fork`, not `origin`. `main` is on `fork` (pushed 2026-10-17); it is ahead of `origin/main` (`git rev-list --count origin/main..main`, 109 on 2026-10-18); `fork` also holds `feature-dashboard` and `hardware`, and `main` there is behind the local one (`git rev-list --count fork/main..main`).
- Remotes: `origin` is `ci-group/ariel` (not ours to push to); `fork` is `git@github.com:Jacopo-DM/ariel.git`, the user's GitHub fork (created 2026-10-17), meant for experiments away from `main`; it holds `main`, `feature-dashboard`, `hardware` and `gh-pages`. `gh` is installed on this Linux machine and logged in as `Jacopo-DM`.

## Data and secrets

- Run outputs go to `__data__/` under the current working directory (`ariel.DATA`, git-ignored). Tests and dev tools must not write anywhere else or read other users' data.
- `mujoco_user_config.json` (and a `.lock` file while probing) is written in the current working folder by `ariel.opengl.gl_manager`; it is git-ignored and machine-specific. `MUJOCO_GL`, if set, always wins over it. Tests preset `MUJOCO_GL=glfw` in `tests/conftest.py` so they never probe.
- No keys or environment variables are used, except `ARIEL_COMPILE_MYPYC=1` to opt in to the mypyc build (`setup.py`).

## Ignore list

| Path | Kind | What makes it |
|---|---|---|
| `__data__/` | generated | runs and examples |
| `.venv/` | generated | `uv venv` / `uv sync` |
| `.typings/` | generated stubs, 17 MB, git-ignored | mujoco, torch and other stubs kept on disk for pyright; untracked 2026-10-18 because the global gate type-checks every tracked file and they fail mypy |
| `docs/_build/`, `docs/autoapi/`, `docs/apidocs/` | generated | sphinx |
| `uv.lock` | generated | `uv lock` |
| `3_spider_vision_new.npy` | binary, stray | tracked at the repo root; origin not recorded |
| `.claude/wiki/` | generated by the `/ingest` skill from source docs | LLM-maintained wiki |
| `mujoco_user_config.json`, `mujoco_user_config.json.lock` | generated, git-ignored | `ariel.opengl.gl_manager` / `ensure_gl_backend` in the working folder |
| `.coverage` | generated, git-ignored | `pytest --cov` |

`.claude/settings.json` has no ignore or deny list and I left it unchanged. Its allow list holds Windows PowerShell entries (`c:\\Users\\John\\...`) from another developer; they do nothing here.

## Things that bite

- `ruff check --fix` is opt-in (config has `fix = false`), and the pre-commit ruff hook does not pass it either. The whitespace and end-of-file hooks do edit files.
- `ruff format <folder>` with an explicit path ignores the `exclude` in `ruff.toml` and reformats `examples/` and `docs/`; run it on `src tests tools` only. A bare `import ariel.rendering` or `ariel.opengl.gl_manager` run from the repo root probes the OpenGL backends the first time and writes `mujoco_user_config.json` there (git-ignored); preset `MUJOCO_GL=glfw` for ad-hoc imports, or delete the file after.
- Read the exit code of `uv run --locked pytest -q`, not the coverage table: the `Miss` column counts statements only, so a gap in branches (`BrPart`) still fails the floor and the TOTAL row looks clean. When the floor fails, pytest-cov's own error message crashes with `ModernTerminalReporter has no attribute write` (INTERNALERROR) instead of saying so; run `uv run --no-sync coverage report --fail-under=100` to see the gap.
- Suppressions use ruff 0.17 syntax, not `# noqa`: `# ruff: ignore[rule-name]  # reason` on the line, `# ruff: file-ignore[rule-name]  # reason` for a file; ruff reports `# noqa` (RUF105). Names, not codes (RUF201).
- `ruff check --select X --fix` with a narrow `--select` makes `RUF100` flag valid `# noqa` comments for the rules you did not select (it removed three needed `E712` ones). Never select `RUF100` alone; check with the full rule set.
- mypy only honours `# type: ignore[code]` when it is the first comment on the line (`  # type: ignore[code]  # noqa: X  # known bug`), and `ruff format` can move a trailing comment to another line when it re-wraps. Run mypy after formatting.
- `MUJOCO_GL` is read when `mujoco` is first imported. Code that decides the backend must run before that import (`ariel/__init__.py`, the top of `ariel/rendering/__init__.py`, which is why it carries `# isort: split` and a file-level ruff ignore). Do not import `mujoco` or any module that does above them.
- `ariel.opengl` must not import MuJoCo, directly or through another module: `ariel/__init__.py` imports it first and chooses `MUJOCO_GL` before MuJoCo loads. `tests/unit/test_opengl/test_opengl_entry_point.py` pins that `import ariel` and `import ariel.opengl` leave `mujoco` unloaded.
