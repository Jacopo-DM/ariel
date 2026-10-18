---
type: api_reference
tags: [mujoco, mjspec, defaults, physics]
source: installed mujoco 3.8.0, mujoco.MjSpec().compile()
date_ingested: 2026-10-18
---
# mjcf_defaults_3_8_0

Defaults measured on 2026-10-18 from the engine installed in this repo (MuJoCo 3.8.0), by reading `spec = mujoco.MjSpec()` and `spec.compile().opt`. This is the authority for ARIEL: the web docs at `/stable/` describe 3.16.0 ([[mujoco_changelog_defaults]]) and three AI research reports disagreed with each other ([[mujoco_parameter_research]]). Script: a short Python file that reads each attribute by name (looping over `dir()` on a spec crashes MuJoCo with a segfault). Enum values are the integers MuJoCo returns.

## option

| Attribute | 3.8.0 default | Note |
|---|---|---|
| `timestep` | 0.002 | s |
| `impratio` | 1.0 | |
| `tolerance` | 1e-8 | |
| `ls_tolerance` | 0.01 | |
| `noslip_tolerance` | 1e-6 | |
| `ccd_tolerance` | 1e-6 | |
| `sleep_tolerance` | 1e-4 | docs `/stable/` say 1e-3 (changed in 3.11) |
| `gravity` | 0 0 -9.81 | |
| `wind` | 0 0 0 | |
| `magnetic` | 0 -0.5 0 | |
| `density`, `viscosity` | 0, 0 | |
| `o_margin` | 0 | |
| `o_solref` | 0.02 1 | |
| `o_solimp` | 0.9 0.95 0.001 0.5 2 | |
| `o_friction` | 1 1 0.005 0.0001 0.0001 | five values |
| `integrator` | 0 | Euler |
| `cone` | 0 | pyramidal |
| `jacobian` | 2 | auto |
| `solver` | 2 | Newton |
| `iterations`, `ls_iterations`, `noslip_iterations` | 100, 50, 0 | |
| `ccd_iterations` | 35 | |
| `sdf_iterations`, `sdf_initpoints` | 10, 40 | |
| `disableflags`, `enableflags`, `disableactuator` | 0, 0, 0 | |

## compiler (spec.compiler)

`autolimits` 1, `boundmass` 0, `boundinertia` 0, `settotalmass` -1, `balanceinertia` 0, `fitaabb` 0, `degree` 1 (angles in degrees; ARIEL sets 0), `eulerseq` xyz, `inertiafromgeom` 2 (auto), `inertiagrouprange` 0 5, `alignfree` 0, `discardvisual` 0, `fusestatic` 0, `saveinertial` 0.

## default geom and joint

| Attribute | 3.8.0 default |
|---|---|
| geom `friction` | 1 0.005 0.0001 |
| geom `solref`, `solimp` | 0.02 1; 0.9 0.95 0.001 0.5 2 |
| geom `margin`, `gap`, `density`, `condim`, `priority`, `solmix` | 0, 0, 1000, 3, 0, 1 |
| joint `damping` | 0 0 0 |
| joint `armature`, `frictionloss` | 0, 0 |
| joint `stiffness` | 0 0 0 |
| joint `range` | 0 0 |
| joint limit and friction `solref` | 0.02 1 |
| joint limit `solimp` | 0.9 0.95 0.001 0.5 2 |

## size and visual

`memory` -1, `nconmax` -1, `njmax` -1, `nstack` -1; `visual.global_.offwidth` 640, `offheight` 480; `visual.quality.shadowsize` 4096.

Not measured: actuator, tendon, site, camera, light and material defaults, and the remaining `visual` groups. Related: [[mjcf_option]], [[mjcf_compiler]], [[mjcf_geom_joint_defaults]].
