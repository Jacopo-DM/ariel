# CODEMAP

Generated from `src/ariel/` on 2026-10-17. Ignore list: see CLAUDE.md.

## Entry points

- Package `ariel` (`src/ariel/__init__.py`): exports `DATA`, `SEED`, `RNG`, `console`, `log`; first reads `mujoco_user_config.json` to set `MUJOCO_GL` (before anything imports mujoco).
- `python -m ariel.opengl.gl_manager`: probes the OpenGL backends and writes `mujoco_user_config.json` in the current folder.
- `python tools/measure_structure.py`: prints size, import-count and entry-point numbers for `src/ariel` and `tests` (`tools/` is not a package; its test is `tests/unit/test_tools/`).
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
| `ariel.body_phenotypes.robogen_lite` | modular robot bodies | `modules` (`Core`, `Brick`, `Hinge`, `Module`), `cppn_neat` (`Genome`, `IdManager`), `decoders` (`HighProbabilityDecoder.probability_matrices_to_graph`, `VectorDecoder.vectors_to_graph`, `CPPN` best-first decoder, `Blueprint` base), `prebuilt_robots` (gecko, spider, john_set, random_20, spider_with_blocks), `constructor`, `config`; `robogen_lite/__init__` exports the config constants, `ModuleType`/`ModuleFaces`/`ModuleRotationsIdx`, `construct_mjspec_from_graph`, `Genome`, `IdManager`, `MorphologyDecoderBestFirst`, `CoreModule` |
| `ariel.body_phenotypes.lynx_mjspec` | Lynx arm body and `unified_pipeline` (common, evolve, replay, plot_from_logs); a package since GS8 | `lynx_arm` (`LynxArm`, `ArmConfig`), `table` |
| `ariel.simulators.mujoco` | generator of MuJoCo settings from physical seeds (goal mujoco-units; package exports `Seeds`, `build_spec`, `generate_xml`; `seeds`, `table`, `settings` (also `CHOICES`, `Choice`), `bands`, `mjcf`, `docs` and `__main__` filled by MU3 to MU9, the generated document is `docs/source/mujoco_parameters.md`) | `seeds` (`Seeds`: frozen model of astropy quantities, `replace`, computed `timestep` and its limits), `table` (`Row`, `derivation_table`), `settings` (`Setting`, `engine_defaults`, `read_object`, `read_spec`, `settings`), `bands` (`Band`, `BandResult`, `BANDS`, `evaluate_bands`, `warn_outside`, `summary`), `mjcf` (`build_spec`, `generate_xml`, `differences`, `summary`), `docs` (`render_markdown`, `write_docs`, `DOCS_PATH`), `__main__` (`main`: the report, `--out`, `--write-docs`) |
| `ariel.simulation.environments` | MuJoCo worlds | `BaseWorld` and terrain worlds (`SimpleFlatWorld`, rugged, tilted, crater, amphitheatre, `OlympicArena`) |
| `ariel.simulation.controllers` | controllers | `Controller`, `ControlCallback` (protocol), `SimpleCPG`, `NaCPG` (one class, after the paper); the package exports `Controller`, `NaCPG`, `create_fully_connected_adjacency` |
| `ariel.simulation.tasks` | fitness functions | `targeted_locomotion`, `gait_learning`, `turning_in_place` |
| `ariel.simulation` | MuJoCo worker base class and tools that run or watch a simulation | `mujoco_worker`, `tracker` (`Tracker`), `mujoco_ops` (`euler_to_quat_conversion`, `has_self_collision`), `runners` |
| `ariel.parameters` | shared types and params | `ariel_types`, `ariel_modules`, `mujoco_params`, `base_model` (`FrozenModel`, `MutableModel`: the base classes of every pydantic record); the package exports the types of `ariel_types`, `ArielModulesConfig`, `MujocoConfig`, `ArielConfig` |
| `ariel.opengl` | OpenGL backend probe and `MUJOCO_GL` setup | exports `configure_from_file`, `ensure_gl_backend`; `gl_backend` (imports neither mujoco nor any ariel package; `GLConfig` and `BackendResult` are the models of the saved result), `gl_manager` (the probe command), `_gl_probe_worker`; import-order sensitive: `ariel/__init__` imports it first, and `renderers` calls `ensure_gl_backend()` before `import mujoco` (a test pins that `import ariel` and `import ariel.opengl` do not load mujoco) |
| `ariel.utils` | helpers | renderers, `VideoRecorder`, `file_ops`, `quaternions` (typed wrappers for the untyped numpy-quaternion), `noise_gen`, `morphological_descriptor`, `optimizers.RevDE`; the package exports only `NormMethod` and `PerlinNoise` (being dissolved, goal restructure R2d) |
| `ariel.visualisation` | analysis tooling (dashboards are on branch `feature-dashboard`) | `analysis/` |

## Import direction

From `from ariel.X import` lines, checked on 2026-10-17: `ec` → `body_phenotypes`, `simulation`, `utils`; `simulation` → `parameters`, `utils`, `simulators` (`environments/_base_world.py` imports `Seeds` and `build_spec`); `simulators` → `parameters` and the `ariel` root (`console`); `body_phenotypes` → `parameters`, `utils` (`quaternions`); `utils` → `parameters`; `parameters` → none. Every package may also import the `ariel` root (`log`, `console`, `DATA`), and `ariel/__init__.py` imports `ariel.opengl` (which imports no other ariel package). `examples` import from all. `visualisation` imports nothing from ariel (only an empty `analysis/__init__.py` is left). No architecture test enforces this.
