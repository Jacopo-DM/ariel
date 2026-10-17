# CODEMAP

Generated from `src/ariel/` on 2026-10-17. Ignore list: see CLAUDE.md.

## Entry points

- Package `ariel` (`src/ariel/__init__.py`): exports `DATA`, `SEED`, `RNG`, `console`, `log`.
- `examples/`: runnable scripts (`a_mujoco`, `b_robots`, `c_genotypes`, `re_book`, `z_ec_course`, `new_EC_engine_example.py`).
- `tests/unit/` mirrors `src/ariel/`; `tests/functional/` is notebooks.

## Modules

| Package | Purpose | Public interface |
|---|---|---|
| `ariel.ec` | evolutionary algorithm engine | `EA`, `EAOperation`, `EASettings`, `DBHandlingMode`, `config`, `Individual`, `Population`, `Archive`, `Crossover`, float and integer generators and mutators |
| `ariel.ec.genotypes.tree` | tree genome and operators | `tree_genome`, `operators`, `validation`, `io` |
| `ariel.ec.genotypes.cppn` | CPPN to morphology decoding | `cppn_decoding` |
| `ariel.ec.genotypes.nde` | neural developmental encoding (PyTorch) | `nde` |
| `ariel.body_phenotypes.robogen_lite` | modular robot bodies | `modules` (`Core`, `Brick`, `Hinge`, `Module`), `cppn_neat` (`Genome`, `IdManager`), `decoders`, `prebuilt_robots` (gecko, spider, john_set, random_20, spider_with_blocks), `constructor`, `config` |
| `ariel.body_phenotypes.lynx_mjspec` | Lynx arm body and `unified_pipeline` (evolve, replay, plot_from_logs) | `lynx_arm`, `table` |
| `ariel.simulation.environments` | MuJoCo worlds | `BaseWorld` and terrain worlds (`SimpleFlatWorld`, rugged, tilted, crater, amphitheatre, `OlympicArena`) |
| `ariel.simulation.controllers` | controllers | `Controller`, `SimpleCPG`, `NaCPG` variants |
| `ariel.simulation.tasks` | fitness functions | `targeted_locomotion`, `gait_learning`, `turning_in_place` |
| `ariel.simulation` | MuJoCo worker base class | `mujoco_worker` |
| `ariel.parameters` | shared types and params | `ariel_types`, `ariel_modules`, `mujoco_params` |
| `ariel.utils` | helpers | `Tracker`, renderers, `VideoRecorder`, `file_ops`, `mujoco_ops`, `runners`, `noise_gen`, `morphological_descriptor`, `optimizers.RevDE` |
| `ariel.visualisation` | dashboards and analysis | `dashboard/`, `analysis/` |

## Import direction

From `from ariel.X import` lines, checked on 2026-10-17: `ec` → `body_phenotypes`, `simulation`, `utils`; `simulation` → `parameters`, `utils`; `body_phenotypes` → `parameters`; `utils` → `parameters`; `parameters` → none. `examples` import from all. `visualisation` was not checked. No architecture test enforces this.
