---
type: api_reference
tags: [mujoco, mjcf, physics]
source: https://mujoco.readthedocs.io/en/stable/overview.html
date_ingested: 2026-10-18
---
# mujoco_overview_units

What MuJoCo is, its model objects, and its statement on units. **Version note.** The `/stable/` docs on 2026-10-18 describe MuJoCo 3.16.0 (changelog of 2026-10-14). This repo has MuJoCo 3.8.0 installed (`uv run python -c 'import mujoco; print(mujoco.__version__)'`) and `pyproject.toml` asks for `>=3.3.6`. Defaults and attributes below may differ for 3.8.0 (see [[mujoco_changelog_defaults]]); verify against the installed version before relying on a value.

## Units

- "MuJoCo does not specify basic physical units."
- "The user may interpret the system of units as they choose, as long as it is consistent."
- "users are encouraged to use MKS, as there are two places where MuJoCo uses MKS-like default values": gravity `(0, 0, -9.81)` ("Earth surface gravity in MKS") and geom density 1000 ("the density of water in MKS").
- "all angular quantities mjModel and mjData are expressed in radians." Angles in MJCF may be given in degrees, which is the compiler default (see [[mjcf_compiler]]).

## Main objects

| Object | Role |
|---|---|
| `mjModel` | compiled, low-level model, constant after compilation; built by the compiler |
| `mjData` | runtime state and intermediate results, allocated from an mjModel, no allocation while stepping |
| `mjOption` | all options that affect the physics (algorithms, parameters, flags, gravity); see [[mjcf_option]] |
| `mjVisual`, `mjStatistic` | visualisation settings; compiler-computed model statistics ([[mjcf_visual]], [[mjcf_size_statistic]]) |
| MJCF (XML) | native model format with a CSS-like default mechanism |
| `mjSpec` | in-memory high-level model, in one-to-one correspondence with MJCF; the Python `MjSpec` ([[MjSpec]]) |
| MJZ, MJB | zip archive of a model; binary version-specific model (not decompilable) |

Pipelines to an mjModel: MJCF/URDF file, parser, mjSpec, compiler, mjModel; or user code, mjSpec, compiler, mjModel. At run time `mj_step(m, d)` advances the simulation.

```
void mj_step(const mjModel* m, mjData* d);
```

## Further chapters

Computation (with a Fluid forces subpage), Modeling, XML Reference, Programming, API Reference, Python, MJX, MuJoCo Warp, Unity plug-in, OpenUSD, Model Gallery, Changelog.

Related: [[mujoco_computation_soft_constraints]], [[mujoco_solver_parameters]], [[mujoco_modeling_guidance]].
