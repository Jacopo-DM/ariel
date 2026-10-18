---
type: api_reference
tags: [mujoco, mjcf, physics]
source: https://mujoco.readthedocs.io/en/stable/XMLreference.html
date_ingested: 2026-10-18
---
# mjcf_actuators

Actuator elements and their shortcut defaults. **Version note.** The `/stable/` docs on 2026-10-18 describe MuJoCo 3.16.0 (changelog of 2026-10-14). This repo has MuJoCo 3.8.0 installed (`uv run python -c 'import mujoco; print(mujoco.__version__)'`) and `pyproject.toml` asks for `>=3.3.6`. Defaults and attributes below may differ for 3.8.0 (see [[mujoco_changelog_defaults]]); verify against the installed version before relying on a value.

## general (base for the shortcuts)

`gear` "1 0 0 0 0 0", `ctrllimited`, `forcelimited`, `actlimited` auto, `ctrlrange` "0 0", `forcerange` "0 0", `gaintype` fixed, `gainprm` "1 0 ... 0" (10 values), `biastype` none, `biasprm` zeros, `dyntype` none, `dynprm` "1 0 ... 0", `damping` "0 0 0", `armature` 0, `delay` 0, `nsample` 0. Units are not stated.

## Shortcuts

| Element | Attributes (defaults) | Sets |
|---|---|---|
| `motor` | none | gainprm "1 0 0", biasprm "0 0 0" |
| `position` | `kp` 1, `kv` 0, `dampratio` 0 (exclusive with kv), `timeconst` 0 (> 0 uses filterexact), `inheritrange` 0 | gainprm "kp 0 0", biasprm "0 -kp -kv" |
| `velocity` | `kv` 1 | gainprm "kv 0 0", biasprm "0 0 -kv" |
| `intvelocity` | `kp` 1, `kv` 0, `dampratio` 0, `inheritrange` 0, `actrange` "0 0" | integrator dynamics, biasprm "0 -kp -kv" |
| `damper` | `kv` 1 (nonnegative); `ctrlrange` required and nonnegative | force F = -kv * velocity * control |
| `cylinder` | `timeconst` 1, `area` 1, `diameter`, `bias` "0 0 0" | filter dynamics, pressure input |
| `muscle` | `timeconst` "0.01 0.04", `range` "0.75 1.05", `scale` 200, `lmin` 0.5, `lmax` 1.6, `vmax` 1.5, `fpmax` 1.3, `fvmax` 1.2 | muscle dynamics, gain and bias |
| `pid`, `orientation`, `adhesion`, `dcmotor` | listed, not fully read | |

For `dcmotor` (new): `saturation` "0 0 0", `cogging` "0 0 0", `lugre` "0 0 0 0 0", `input` voltage, `controller` "0 0 0 0 0 0". Changelog 3.16: shortcut gains come only from a default of the same shortcut or from `general`; a `pid` without `kp` now has gain 1 ([[mujoco_changelog_defaults]]).

Related: [[mjcf_geom_joint_defaults]], [[mujoco_modeling_guidance]].
