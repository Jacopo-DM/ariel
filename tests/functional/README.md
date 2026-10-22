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
row, and a new area gets a row in the same commit. A notebook lives in the
folder of its main package (`ec/nde.ipynb`) and the table names it by that
path; a notebook listed under several areas sits in the folder of one of them,
and only a notebook about no ARIEL area sits at the top. `none yet` marks an area with
no notebook. `tests/unit/test_functional_notebooks.py` fails when an area has no
row, a row names a notebook that does not exist, a notebook has no row, or a
notebook sits outside the folders of the areas that list it.

| Area | Notebook |
|---|---|
| `body_phenotypes` | `body_phenotypes/decoders`, `body_phenotypes/cppn_genome`, `body_phenotypes/morphology` |
| `body_phenotypes/lynx_mjspec` | none yet |
| `body_phenotypes/robogen_lite` | `body_phenotypes/ariel_modules`, `body_phenotypes/decoders`, `body_phenotypes/cppn_genome`, `body_phenotypes/prebuilt_robots`, `body_phenotypes/overlap_repair`, `body_phenotypes/body_density`, `body_phenotypes/body_density_repair_nde`, `body_phenotypes/body_density_repair_others` |
| `ec` | `ec/ec_engine` |
| `ec/generators` | `ec/ec_engine` |
| `ec/genotypes` | `ec/nde`, `ec/tree_genome`, `body_phenotypes/body_density`, `body_phenotypes/body_density_repair_nde`, `body_phenotypes/body_density_repair_others` |
| `ec/optimizers` | none yet |
| `opengl` | `opengl/opengl_and_video` |
| `parameters` | `parameters/parameters` |
| `rendering` | `body_phenotypes/ariel_modules`, `simulation/sim_environments`, `opengl/opengl_and_video`, `body_phenotypes/prebuilt_robots` |
| `simulation` | `simulation/speed_tests` |
| `simulation/controllers` | `simulation/cpg_properties`, `simulation/cpg_locomotion`, `simulation/cpg_steering`, `simulation/cpg_initialisation`, `simulation/cpg_initialisation_search` |
| `simulation/environments` | `simulation/sim_environments` |
| `simulation/tasks` | `simulation/tasks` |
| `simulators` | `simulators/mujoco_settings` |
| `simulators/mujoco` | `simulators/mujoco_settings` |
| `visualisation` | none yet |
| `visualisation/analysis` | none yet |
| (third-party libraries, not an ARIEL area) | `external_libs` |
