---
type: api_reference
tags: [mujoco, mjcf, physics]
source: https://mujoco.readthedocs.io/en/stable/XMLreference.html
date_ingested: 2026-10-18
---
# mjcf_size_statistic

The `<size>` and `<statistic>` elements. **Version note.** The `/stable/` docs on 2026-10-18 describe MuJoCo 3.16.0 (changelog of 2026-10-14). This repo has MuJoCo 3.8.0 installed (`uv run python -c 'import mujoco; print(mujoco.__version__)'`) and `pyproject.toml` asks for `>=3.3.6`. Defaults and attributes below may differ for 3.8.0 (see [[mujoco_changelog_defaults]]); verify against the installed version before relying on a value.

## size (structural, do not change after compilation)

| Attribute | Default | Meaning |
|---|---|---|
| `memory` | "-1" | arena size in bytes; suffixes K, M, G, T, P, E; -1 lets the compiler guess; setting both `memory` and `njmax`/`nstack` is an error |
| `njmax`, `nconmax`, `nstack` | -1 | legacy; `nconmax` caps runtime contacts, extras are discarded with a warning |
| `nuserdata` | 0 | |
| `nkey` | 0 | the allocated keyframe count is the larger of this and the number of key elements |
| `nuser_body`, `nuser_jnt`, `nuser_geom`, `nuser_site`, `nuser_cam`, `nuser_tendon`, `nuser_actuator`, `nuser_sensor` | -1 | custom user parameters per element |

## statistic (optional overrides of compiler-computed values)

| Attribute | Dimension | Meaning |
|---|---|---|
| `meanmass` | M | average body mass excluding world; scales the perturbation force |
| `meaninertia` | M L^2 | average diagonal joint-space inertia at `qpos0`; scales solver cost and gradient for early termination |
| `meansize` | L | average body radius heuristic; multiplies visual scale and acts as the length unit |
| `extent` | L | half the side of the bounding box, strictly positive; initial free camera distance is 1.5 times this |
| `center` | L | bounding-box centre, used to centre the free camera |

Related: [[mjcf_option]], [[mjcf_visual]], [[MjStatistic]].
