---
type: api_reference
tags: [mujoco, mjcf, physics]
source: https://mujoco.readthedocs.io/en/stable/XMLreference.html
date_ingested: 2026-10-18
---
# mjcf_compiler

The `<compiler>` element. **Version note.** The `/stable/` docs on 2026-10-18 describe MuJoCo 3.16.0 (changelog of 2026-10-14). This repo has MuJoCo 3.8.0 installed (`uv run python -c 'import mujoco; print(mujoco.__version__)'`) and `pyproject.toml` asks for `>=3.3.6`. Defaults and attributes below may differ for 3.8.0 (see [[mujoco_changelog_defaults]]); verify against the installed version before relying on a value.

| Attribute | Type | Default | Meaning |
|---|---|---|---|
| `autolimits` | bool | true | infer limited, forcelimited, ctrllimited, actlimited from the presence of a range |
| `boundmass` | real | 0 | M. Lower bound on body mass (not world) |
| `boundinertia` | real | 0 | M L^2. Lower bound on diagonal inertia |
| `settotalmass` | real | -1 | M. Deprecated (3.16); scales total mass when positive |
| `balanceinertia` | bool | false | average the diagonal inertia when the triangle inequality is violated |
| `angle` | degree, radian | degree (MJCF), radian (URDF) | unit of angles in the file |
| `eulerseq` | string | "xyz" | three characters from x,y,z,X,Y,Z; uppercase is extrinsic |
| `fitaabb` | bool | false | fit primitives to the AABB instead of the equivalent inertia box |
| `inertiafromgeom` | false, true, auto | auto | infer mass and inertia from geoms |
| `inertiagrouprange` | int(2) | "0 5" | geom groups used for inertial inference |
| `alignfree` | bool | false | align free-joint bodies with their inertial frame |
| `fusestatic` | bool | false (true for URDF) | fuse static bodies into their parents |
| `discardvisual` | bool | false (true for URDF) | drop visual-only elements |
| `usethread` | bool | true | multi-threaded length-range computation and mesh loading |
| `saveinertial`, `savecompiled`, `savecanonical` | bool | false, true, true | what saving an MjSpec writes |
| `conflict` | warning, merge, error | warning | conflicts during `mjs_attach` |
| `meshdir`, `texturedir`, `assetdir`, `strippath`, `coordinate` | | | paths; `coordinate="global"` is no longer supported |

`<compiler><lengthrange>` (muscle length ranges): `mode` muscle, `useexisting` true, `uselimit` false, `accel` 20, `maxforce` 0, `timeconst` 1 s, `timestep` 0.01 s (0 uses the model timestep), `inttotal` 10 s, `interval` 2 s, `tolrange` 0.05.

Note for ARIEL: in the Python `MjSpec` this setting is the boolean `spec.compiler.degree` (ARIEL sets it from `MujocoConfig.degree`, default `False`). MuJoCo's own default is degrees, while `mjModel` is always radians ([[mujoco_overview_units]]).

Related: [[mjcf_option]], [[mujoco_modeling_guidance]].
