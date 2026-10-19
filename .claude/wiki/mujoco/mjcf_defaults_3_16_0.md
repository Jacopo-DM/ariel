---
type: api_reference
tags: [mujoco, mjspec, defaults, physics]
source: installed mujoco 3.16.0, mujoco.MjSpec()
date_ingested: 2026-10-19
---
# mjcf_defaults_3_16_0

Defaults measured on 2026-10-19 from the engine installed in this repo after the upgrade from 3.8.0 to 3.16.0 (task MU1 of mujoco-units), by reading `spec = mujoco.MjSpec()` and comparing every `option`, `compiler`, `visual` and `stat` attribute with [[mjcf_defaults_3_8_0]]. Everything not listed here is unchanged from 3.8.0.

| Attribute | 3.8.0 | 3.16.0 | Note |
|---|---|---|---|
| `option.sleep_tolerance` | 1e-4 | 1e-3 | the change the `/stable/` docs describe ([[mujoco_changelog_defaults]]) |
| `visual.global_.bvactive` | 1 | 0 | bounding-volume visualisation off |
| `compiler.authored` | absent | 0 | new attribute |
| `compiler.savecanonical`, `compiler.savecompiled` | absent | true | new attributes |
| `compiler.conflict` | absent | `mjCONFLICT_WARNING` (0) | new enum; values `WARNING` 0, `MERGE` 1, `ERROR` 2 |
| `compiler.inertiafromgeom` | int 2 | enum `mjtInertiaFromGeom`, `AUTO` (2) | the Python type changed from a plain int to an enum (`FALSE` 0, `TRUE` 1, `AUTO` 2); the value is the same |

The timestep stays 0.002 s, so a world built without a generated MJCF behaves as before except for the sleep tolerance, which only matters when sleeping is switched on.
