# Functional notebooks

Each notebook here shows one feature area of ARIEL working end to end: a
tutorial to read, a picture or table to judge by eye, and a smoke test that
fails when any cell raises. They do not replace the unit tests in
`tests/unit`; those pin exact behaviour and carry the coverage number.

## Run them

```bash
uv run --no-sync python tools/run_notebooks.py
```

One line per notebook: PASS, FAIL or SKIP, and the seconds. A failing notebook
also prints the cell and the error. The exit code is 1 when any notebook fails
(SKIP is not a failure). With file arguments only those run. `--slow` adds the
notebooks marked `"slow": true` in their metadata, `--timeout SECONDS` sets the
limit per notebook (default 600), and `--save` writes the outputs back into the
notebooks.

A notebook that needs a working OpenGL backend carries `"requires": ["opengl"]`
in its metadata and prints SKIP, with the reason, on a machine without one.

## Hooks

Two pre-commit hooks run the notebooks (`.pre-commit-config.yaml`). At commit,
only the notebooks in the commit run, each with a 300 second limit (a slow
notebook named on the command line runs even without `--slow`, so editing
`speed_tests` takes about two minutes at commit). At push, every non-slow
notebook runs, so a library change that breaks a notebook nobody edited is
caught. pre-commit installs the push hook only when asked:

```bash
uv run --no-sync pre-commit install --hook-type pre-commit --hook-type pre-push
```

Skipping hooks with `--no-verify` skips this check too. There is no CI job: CI
has no OpenGL either, so the OpenGL notebooks would only SKIP there.

## When to run them

Before a release, and after changing the area a notebook covers. A FAIL means a
cell raised, a check inside it failed, or the kernel crashed; the line names the
cell. A SKIP means the machine lacks something the notebook declared.

## Coverage

One row per package and sub-package of `src/ariel`. The rule: every area has a
row, and a new area gets a row in the same commit. `none yet` marks an area with
no notebook. `tests/unit/test_functional_notebooks.py` fails when an area has no
row, a row names a notebook that does not exist, or a notebook has no row.

| Area | Notebook |
|---|---|
| `body_phenotypes` | `decoders` |
| `body_phenotypes/lynx_mjspec` | none yet |
| `body_phenotypes/robogen_lite` | `decoders` |
| `ec` | none yet |
| `ec/generators` | none yet |
| `ec/genotypes` | `nde` |
| `ec/optimizers` | none yet |
| `opengl` | none yet |
| `parameters` | none yet |
| `rendering` | `ariel_modules`, `sim_environments` |
| `simulation` | `speed_tests` |
| `simulation/controllers` | `cpg` |
| `simulation/environments` | `sim_environments`, `utils` |
| `simulation/tasks` | none yet |
| `simulators` | none yet |
| `simulators/mujoco` | none yet |
| `visualisation` | none yet |
| `visualisation/analysis` | none yet |
| (third-party libraries, not an ARIEL area) | `external_libs` |
