---
type: api_reference
tags: [mujoco, mjcf, physics]
source: https://mujoco.readthedocs.io/en/stable/XMLreference.html
date_ingested: 2026-10-18
---
# mjcf_visual

The `<visual>` element (maps to `mjVisual`). Only read in the part of the page that covers `global`, `quality`, `headlight`, `map`, `scale` and `rgba`. **Version note.** The `/stable/` docs on 2026-10-18 describe MuJoCo 3.16.0 (changelog of 2026-10-14). This repo has MuJoCo 3.8.0 installed (`uv run python -c 'import mujoco; print(mujoco.__version__)'`) and `pyproject.toml` asks for `>=3.3.6`. Defaults and attributes below may differ for 3.8.0 (see [[mujoco_changelog_defaults]]); verify against the installed version before relying on a value.

| Group | Defaults |
|---|---|
| `global` | cameraid -1 (free camera), orthographic false, fovy 45 (degrees), ipd 0.068, azimuth 90, elevation -45 (degrees), linewidth 1, glow 0.3, realtime 1 (>0), offwidth 640, offheight 480 (pixels), ellipsoidinertia false, bvactive false |
| `quality` | shadowsize 4096, offsamples 4, numslices 28, numstacks 16, numquads 4 |
| `headlight` | ambient 0.1 0.1 0.1, diffuse 0.4 0.4 0.4, specular 0.5 0.5 0.5, active 1 |
| `map` | stiffness 100, stiffnessrot 500, force 0.005, torque 0.1, alpha 0.3, fogstart 3, fogend 10, znear 0.01, zfar 50, haze 0.3, shadowclip 1, shadowscale 0.6, actuatortendon 2 (multipliers of model extent or mean body mass, not physical units) |
| `scale` | forcewidth 0.1, contactwidth 0.3, contactheight 0.1, connect 0.2, com 0.4, camera 0.3, light 0.3, selectpoint 0.2, jointlength 1.0, jointwidth 0.1, actuatorlength 0.7, actuatorwidth 0.2, framelength 1.0, framewidth 0.1, constraint 0.1, slidercrank 0.2, frustum 10 (multipliers of mean body size) |
| `rgba` | fog 0 0 0 1; haze 1 1 1 1; force 1 0.5 0.5 1; inertia 0.8 0.2 0.2 0.6; joint 0.2 0.6 0.8 1; actuator 0.2 0.25 0.2 1; com 0.9 0.9 0.9 1; contactpoint 0.9 0.6 0.2 1; contactforce 0.7 0.9 0.9 1; constraint 0.9 0 0 1; others in the reference |

The changelog notes `bvactive` default changed from true to false in 3.12 ([[mujoco_changelog_defaults]]). None of these are physical quantities, so they are not astropy seeds. ARIEL overrides some in `BaseWorld._init_spec` (`offwidth` 1280, `offheight` 960, `shadowsize` 8192, headlight and scale values).

Related: [[mjcf_size_statistic]], [[MjvOption]].
