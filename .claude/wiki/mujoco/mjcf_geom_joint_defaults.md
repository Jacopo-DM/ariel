---
type: api_reference
tags: [mujoco, mjcf, physics]
source: https://mujoco.readthedocs.io/en/stable/XMLreference.html
date_ingested: 2026-10-18
---
# mjcf_geom_joint_defaults

Geom, joint and contact-pair attributes with defaults that carry physical meaning, plus the `<default>` element. **Version note.** The `/stable/` docs on 2026-10-18 describe MuJoCo 3.16.0 (changelog of 2026-10-14). This repo has MuJoCo 3.8.0 installed (`uv run python -c 'import mujoco; print(mujoco.__version__)'`) and `pyproject.toml` asks for `>=3.3.6`. Defaults and attributes below may differ for 3.8.0 (see [[mujoco_changelog_defaults]]); verify against the installed version before relying on a value.

## default element

Only attribute `class` (string; the top level is named "main" if omitted). Child elements can inherit class attributes ([[mujoco_modeling_guidance]]).

## geom

| Attribute | Default | Meaning |
|---|---|---|
| `type` | sphere | plane, hfield, sphere, capsule, ellipsoid, cylinder, box, mesh, sdf |
| `size` | "0 0 0" | positive; box and ellipsoid take half-sizes |
| `contype`, `conaffinity` | 1, 1 | bitmasks; two geoms collide if one's contype and the other's conaffinity share a bit |
| `condim` | 3 | 1, 3, 4 or 6 |
| `priority` | 0 | which geom's contact parameters win |
| `friction` | "1 0.005 0.0001" | sliding, torsional, rolling; combined with `solmix` and `priority` |
| `density` | 1000 | kg per volume (mass per area if `shellinertia`); 1000 is water in SI |
| `mass` | none | overrides density |
| `solmix` | 1 | weight when averaging contact parameters |
| `margin`, `gap` | 0, 0 | contact activates at distance margin; gap is extra detection distance |
| `solref`, `solimp` | "0.02 1", "0.9 0.95 0.001 0.5 2" | see [[mujoco_solver_parameters]] |
| `adhesion` | 0 | pull-off force per contact |
| `fluidcoef` | "0.5 0.25 1.5 1.0 1.0" | blunt drag, slender drag, angular drag, Kutta lift, Magnus lift |

## joint

| Attribute | Default | Meaning |
|---|---|---|
| `type` | hinge | free, ball, slide, hinge |
| `axis` | "0 0 1" | hinge axis or slide direction, normalised |
| `range` | "0 0" | limits in degrees or radians per the compiler `angle`; ball joints use only the second value |
| `limited` | auto | auto enables limits if `range` is set (with `autolimits`) |
| `stiffness` | "0 0 0" | coefficients a, b, c of `f = -(a x + b x^2 + c x^3)` from `springref` |
| `damping` | "0 0 0" | coefficients of `f = -(a v + b v|v| + c v^3)`; large values can destabilise explicit integrators; Euler handles damping implicitly |
| `armature` | 0 | extra rotor inertia; reflected inertia = rotor inertia x gear^2; positive values improve stability |
| `frictionloss` | 0 | dry friction; positive enables |
| `margin` | 0 | distance below which limits activate |
| `springdamper` | "0 0" | if both positive, overrides stiffness and damping to give a time constant and damping ratio |
| `solreflimit`, `solimplimit`, `solreffriction`, `solimpfriction` | | limit and friction solver parameters |
| `actuatorfrcrange`, `actuatorfrclimited` | "0 0", auto | clamp the summed actuator force (scalar joints) |

`freejoint` is a shortcut with stiffness, damping, frictionloss and armature all 0.

## contact pair

`condim` 3, `friction` real(5) "1 1 0.005 0.0001 0.0001" (tangential, torsional, rolling), `solref`, `solimp`, `solreffriction` (default "0 0"), `margin`, `gap`, `adhesion`.

Related: [[mjcf_actuators]], [[mjcf_option]].
