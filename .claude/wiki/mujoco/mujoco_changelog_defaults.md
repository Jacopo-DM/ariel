---
type: api_reference
tags: [mujoco, mjcf, physics]
source: https://mujoco.readthedocs.io/en/stable/changelog.html
date_ingested: 2026-10-18
---
# mujoco_changelog_defaults

Changes to defaults and default behaviour, from the changelog, newest first. Only the first 100000 of 338188 characters were read (versions 3.9.0 to 3.16.0); earlier releases, including 3.8.0 and everything before it, are not covered. **Version note.** The `/stable/` docs on 2026-10-18 describe MuJoCo 3.16.0 (changelog of 2026-10-14). This repo has MuJoCo 3.8.0 installed (`uv run python -c 'import mujoco; print(mujoco.__version__)'`) and `pyproject.toml` asks for `>=3.3.6`. Defaults and attributes below may differ for 3.8.0 (see [[mujoco_changelog_defaults]]); verify against the installed version before relying on a value.

| Version | Date | Change |
|---|---|---|
| 3.16.0 | 2026-10-14 | compiler `savecompiled`, `savecanonical` default true; `settotalmass` deprecated (warns); `fusestatic` applied as a spec operation, new body attribute `fuse`; geom/site/camera `quat` no longer ignored when the default class sets another orientation; shortcut actuators take gains only from the same shortcut default or `general`; `pid` without `kp` has gain 1 not 0 |
| 3.15.0 | 2026-10-05 | `dampratio` uses operational-space reflected inertia; multi-link actuators may show lower damping |
| 3.14.0 | 2026-09-22 | tendon `actuatorfrclimited` defaults to auto (was false) |
| 3.13.0 | 2026-09-08 | implicit flex special case removed; flex elasticity or passive flex contact needs `integrator="discrete"` |
| 3.12.0 | 2026-08-20 | passive flex collision stiffness is a mass-scaled natural frequency; visual `bvactive` default true to false; mocap bodies are the root of their own weld group; light `softness` default 0.2 |
| 3.11.0 | 2026-07-27 | option `sleep_tolerance` default 1e-4 to 1e-3; `implicitfast` uses gyroscopic derivatives for free bodies; `intvelocity` `actlimited` defaults to auto |
| 3.10.0 | 2026-06-22 | CG solver uses Hager-Zhang update; compiler `conflict` default warning (will become merge) |
| 3.9.0 | 2026-05-27 | PGS visitation order uses a fixed seed; flexes may sleep |

For ARIEL (3.8.0 installed): any default above that changed after 3.8.0 (`sleep_tolerance`, `bvactive`, the 3.16 actuator rules) differs between the docs and the installed engine. The mujoco-units generator should read defaults from the installed engine, not from these docs.

Related: [[mjcf_option]], [[mjcf_compiler]], [[mjcf_visual]], [[mjcf_actuators]].
