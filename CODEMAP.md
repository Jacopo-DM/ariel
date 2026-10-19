# CODEMAP

Generated from `src/ariel/` on 2026-10-17. Ignore list: see CLAUDE.md.

## Entry points

- Package `ariel` (`src/ariel/__init__.py`): exports `DATA`, `SEED`, `RNG`, `console`, `log`; first reads `mujoco_user_config.json` to set `MUJOCO_GL` (before anything imports mujoco).
- `python -m ariel.utils.gl_manager`: probes the OpenGL backends and writes `mujoco_user_config.json` in the current folder.
- `examples/`: runnable scripts (`a_mujoco`, `b_robots`, `c_genotypes` (includes `cppn_genome_checks.py`, the old `cppn_neat/tests.py`), `re_book`, `z_ec_course`, `new_EC_engine_example.py`).
- `tests/unit/` mirrors `src/ariel/`; `tests/functional/` is notebooks; `tests/conftest.py` presets `MUJOCO_GL`; `tests/unit/test_wiki_structure.py` checks the knowledge wiki in `.claude/wiki/` (not `src`).
- `.claude/wiki/`: knowledge wiki in topic folders `mujoco/`, `mjx/`, `cma-es/`, `nevergrad/`, `ariel/`, `astropy/`, `pydantic/`, plus `index.md` and one `log.md`; layout rules in `.claude/SCHEMA.md`, commands `/ingest` and `/query` in `.claude/commands/`.

## Modules

| Package | Purpose | Public interface |
|---|---|---|
| `ariel.ec` | evolutionary algorithm engine | `EA`, `EAOperation`, `Operation` (protocol for pipeline elements), `EASettings`, `DBHandlingMode`, `config`, `Individual`, `Population`, `Archive`, `Crossover`, float and integer generators and mutators |
| `ariel.ec.genotypes.tree` | tree genome and operators | `tree_genome` (`TreeGenome`, `TreeNode`, `TreeEdge` pydantic models), `operators`, `validation` (checks a genome dict), `io` |
| `ariel.ec.genotypes.cppn` | CPPN to morphology decoding | `cppn_decoding` |
| `ariel.ec.genotypes.nde` | neural developmental encoding (PyTorch) | `nde` |
| `ariel.body_phenotypes.robogen_lite` | modular robot bodies | `modules` (`Core`, `Brick`, `Hinge`, `Module`), `cppn_neat` (`Genome`, `IdManager`), `decoders` (`HighProbabilityDecoder.probability_matrices_to_graph`, `VectorDecoder.vectors_to_graph`, `CPPN` best-first decoder, `Blueprint` base), `prebuilt_robots` (gecko, spider, john_set, random_20, spider_with_blocks), `constructor`, `config` |
| `ariel.body_phenotypes.lynx_mjspec` | Lynx arm body and `unified_pipeline` (common, evolve, replay, plot_from_logs); a package since GS8 | `lynx_arm` (`LynxArm`, `ArmConfig`), `table` |
| `ariel.simulators.mujoco` | generator of MuJoCo settings from physical seeds (goal mujoco-units; `seeds`, `table` and `settings` filled by MU3 to MU5, tasks MU6 to MU9 add the rest) | `seeds` (`Seeds`: frozen model of astropy quantities, `replace`, computed `timestep` and its limits), `table` (`Row`, `derivation_table`), `settings` (`Setting`, `engine_defaults`, `read_object`, `settings`) |
| `ariel.simulation.environments` | MuJoCo worlds | `BaseWorld` and terrain worlds (`SimpleFlatWorld`, rugged, tilted, crater, amphitheatre, `OlympicArena`) |
| `ariel.simulation.controllers` | controllers | `Controller`, `ControlCallback` (protocol), `SimpleCPG`, `NaCPG` (one class, after the paper) |
| `ariel.simulation.tasks` | fitness functions | `targeted_locomotion`, `gait_learning`, `turning_in_place` |
| `ariel.simulation` | MuJoCo worker base class | `mujoco_worker` |
| `ariel.parameters` | shared types and params | `ariel_types`, `ariel_modules`, `mujoco_params`, `base_model` (`FrozenModel`, `MutableModel`: the base classes of every pydantic record) |
| `ariel.utils` | helpers | `Tracker`, renderers, `VideoRecorder`, `file_ops`, `gl_backend` / `gl_manager` / `_gl_probe_worker` (OpenGL backend probe and `MUJOCO_GL` setup, import-order sensitive: `gl_backend` imports neither mujoco nor any ariel package, and `renderers` calls `ensure_gl_backend()` before `import mujoco`; `GLConfig` and `BackendResult` are the models of the saved result), `mujoco_ops`, `quaternions` (typed wrappers for the untyped numpy-quaternion), `runners`, `noise_gen`, `morphological_descriptor`, `optimizers.RevDE` |
| `ariel.visualisation` | analysis tooling (dashboards are on branch `feature-dashboard`) | `analysis/` |

## Import direction

From `from ariel.X import` lines, checked on 2026-10-17: `ec` → `body_phenotypes`, `simulation`, `utils`; `simulation` → `parameters`, `utils` (and `simulators`, planned: MU8); `simulators` → `parameters` (planned, no imports yet); `body_phenotypes` → `parameters`, `utils` (`quaternions`); `utils` → `parameters`; `parameters` → none. Every package may also import the `ariel` root (`log`, `console`, `DATA`), and `ariel/__init__.py` imports `ariel.utils.gl_backend`. `examples` import from all. `visualisation` imports nothing from ariel (only an empty `analysis/__init__.py` is left). No architecture test enforces this.
