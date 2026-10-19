---
type: source_summary
tags: [source, mujoco, simulation, visualization, rendering]
source: https://mujoco.readthedocs.io/en/stable/programming/samples.html
author: DeepMind / MuJoCo team
date_ingested: 2026-10-19
---
# Source - MuJoCo Code Samples

Official MuJoCo documentation page describing five bundled C++ code samples (`testspeed`, `simulate`, `compile`, `basic`, `record`). Covers simulation timing, interactive visualization, model format conversion, and offscreen video recording — all key patterns for ariel's rendering and simulation infrastructure.

## Entity Pages Created

- [[mujoco_code_samples]] — api_reference covering all five samples: command-line args, controls, architecture notes, OpenGL backend selection, ffmpeg pipeline for `record`

## Refresh 2026-10-19

Rechecked against MuJoCo 3.16.0 (installed headers and the page fetched 2026-10-19). Updated: mujoco_code_samples: --sleep_tolerance and compiler timers added.
