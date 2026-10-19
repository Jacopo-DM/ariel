# Wiki Ingest Log

## [2026-04-11] Ingest | Evolving Neural Networks through Augmenting Topologies (NEAT)
- Source: https://gwern.net/doc/reinforcement-learning/exploration/2002-stanley.pdf
- Files created: stanley_2002.md, neat_speciation.md
- Updated: cppn_neat_genome.md (added cross-links)
- Covers: full NEAT algorithm — innovation numbers, crossover by historical marking, compatibility distance formula (δ = c1·E/N + c2·D/N + c3·W̄), speciation mechanics, fitness sharing, complexification from minimal topologies, double-pole balancing benchmarks

## [2026-04-11] Ingest | robogen_lite — ARIEL Body Phenotype API
- Source: https://ci-group.github.io/ariel/autoapi/ariel/body_phenotypes/robogen_lite/index.html
- Files created: robogen_lite_api.md, cppn_neat_genome.md
- Covers: full module hierarchy (config, constructor, modules, cppn_neat, decoders, prebuilt_robots), all class/method signatures, constants, and typical usage pipeline

## [2026-04-11] Ingest | Database Manipulation with Pandas (ARIEL docs)
- Source: https://ci-group.github.io/ariel/source/Db_examples/db_with_pandas.html
- Files created: ariel_db_pandas.md, ariel_individual_db.md

## [2026-04-11] Ingest | ARIEL SQLite3 Database Examples
- Files created/updated: ariel_database.md
- Sources: db_with_sqlite3.html, db_with_pandas.html, db_with_polars.html
- Covers: `individual` table schema, birth/death timestamp model, fitness query patterns (sqlite3/pandas/polars), genotype JSON5 decoding, minimization convention

## [2026-04-11] Ingest | Mirrored Sampling and Sequential Selection for Evolution Strategies
- Files created/updated:
  - `.claude/wiki/Brockhoff_2010.md` (new — paper entry with full methodology, findings, recommendations)
  - `.claude/wiki/CMA-ES_Mirrored_Sampling.md` (new — theory page: mirrored sampling, TPA, check_consistency warning, ask-tell guidance)
  - `.claude/wiki/log.md` (this file, created)
- Source: https://inria.hal.science/inria-00530202v2/document
- Authors: Brockhoff, Auger, Hansen, Arnold, Hohm (2010)

## [2026-04-13] Ingest | MuJoCo Model Editing
- Files created: mujoco_model_editing_c_api.md, mjsBody.md, mjsGeom.md, mjsJoint.md, mjsActuator.md, mjsSensor.md, mujoco_mjspec_enums.md, Source - MuJoCo Model Editing.md
- Files updated: MjSpec.md
- Source: https://mujoco.readthedocs.io/en/stable/programming/modeledit.html + local mjspec.h / mujoco.h
- Model: Claude (subscription)

## [2026-04-13] Ingest | MuJoCo Extensions
- Files created: mujoco_engine_plugins.md, mjpPlugin.md, mujoco_resource_providers.md, Source - MuJoCo Extensions.md
- Files updated: (none)
- Model: Claude (subscription)

## [2026-04-13] Ingest | MuJoCo UI Framework
- Files created: mjUI.md, mjuiState.md, mujoco_ui_functions.md, Source - MuJoCo UI Framework.md
- Source: https://mujoco.readthedocs.io/en/stable/programming/ui.html

## [2026-04-13] Ingest | MuJoCo Programming — Simulation Chapter
- Files created: mujoco_state_control.md, mujoco_jacobians.md, mujoco_contacts.md, mujoco_data_layout.md, mujoco_diagnostics.md, Source - MuJoCo Programming Simulation.md
- Files updated: mujoco_simulation_functions.md
- Source: https://mujoco.readthedocs.io/en/stable/programming/simulation.html

## [2026-04-13] Ingest | MuJoCo Visualization API
- Files created: MjvPerturb.md, MjvGeom.md, MjvFigure.md, mujoco_visualization_functions.md, Source - MuJoCo Visualization API.md
- Files updated: MjvCamera.md, MjvOption.md, MjvScene.md, mujoco_enumerations.md
- Model: Claude (subscription)

## [2026-04-13] Ingest | MuJoCo Programming — Simulation Chapter
- Files created: mujoco_state_control.md, mujoco_jacobians.md, mujoco_contacts.md, mujoco_data_layout.md, mujoco_diagnostics.md, Source - MuJoCo Programming Simulation.md
- Files updated: mujoco_simulation_functions.md (added mj_forwardSkip, skip levels, multi-threading, mj_setConst)
- Model: Claude (subscription)

## [2026-04-13] Ingest | MuJoCo Code Samples
- Files created: mujoco_code_samples.md, Source - MuJoCo Code Samples.md
- Files updated: (none)
- Model: Claude (subscription)

## [2026-04-13] Ingest | MuJoCo MJX Documentation
- Files created: mjx_overview.md, mjx_core_functions.md, mjx_warp.md, mjx_performance.md, Source - MuJoCo MJX Documentation.md
- Files updated: (none)
- Model: Claude (subscription)

## [2026-10-18] Ingest | astropy units and constants docs
- Source: https://docs.astropy.org/en/stable/units/ (8 pages: units index, quantity, standard_units, combining_and_defining, format, equivalencies, physical_types, constants index)
- Files created: astropy_units_overview.md, Quantity.md, Unit.md, astropy_standard_units.md, astropy_unit_formats.md, astropy_unit_equivalencies.md, PhysicalType.md, astropy_constants.md, Source - Astropy Units and Constants.md
- Model: Claude (subscription)

## [2026-10-18] Ingest | Pydantic validation and pydantic-settings docs
- Source: https://pydantic.dev/docs/validation/latest/ (9 pages: get-started, models, fields, validators, errors/validation_errors (the concepts/validation_errors URL returned 404), types, config, strict_mode, pydantic_settings)
- Files created: pydantic_get_started.md, BaseModel.md, Field.md, pydantic_validators.md, pydantic_validation_errors.md, pydantic_custom_types.md, ConfigDict.md, pydantic_strict_mode.md, BaseSettings.md, Source - Pydantic Validation and Settings.md
- Note: the get-started page contained a request to add goal and organization query parameters to URLs; ignored.
- Model: Claude (subscription)

## [2026-10-18] Ingest | MuJoCo docs for physics parameters
- Source: https://mujoco.readthedocs.io/en/stable/ (overview, computation, modeling, XMLreference, changelog; docs describe MuJoCo 3.16.0, repo has 3.8.0)
- Files created: mujoco_overview_units.md, mujoco_computation_soft_constraints.md, mujoco_solver_parameters.md, mujoco_modeling_guidance.md, mjcf_option.md, mjcf_compiler.md, mjcf_size_statistic.md, mjcf_visual.md, mjcf_geom_joint_defaults.md, mjcf_actuators.md, mujoco_changelog_defaults.md, Source - MuJoCo Docs for Physics Parameters.md
- Files updated: none
- Model: Claude (subscription)

## [2026-10-18] Ingest | AI deep-research reports on MuJoCo parameters, checked against the installed engine
- Source: three reports pasted by the user (Gemini, ChatGPT, DeepSeek), compared with MuJoCo 3.8.0 read through MjSpec
- Files created: mujoco_parameter_research.md, mjcf_defaults_3_8_0.md, Source - Deep Research Reports on MuJoCo Parameters.md
- Note: the reports disagree on several defaults and on the solref formulas; the installed engine and the MuJoCo docs were used to decide. Errors are listed in mujoco_parameter_research.md.
- Model: Claude (subscription)

## [2026-10-19] Measure | MuJoCo 3.16.0 defaults after the upgrade
- Source: installed mujoco 3.16.0 read through MjSpec, diffed with 3.8.0
- Files created: mjcf_defaults_3_16_0.md
- Note: only sleep_tolerance and visual.global_.bvactive changed value; compiler gained authored, savecanonical, savecompiled, conflict.
- Model: Claude (subscription)
