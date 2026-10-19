---
type: source_summary
tags: [source, mujoco, mjx, jax, gpu]
source: https://mujoco.readthedocs.io/en/stable/mjx.html
author: Google DeepMind
date_ingested: 2026-10-19
---

# Source - MuJoCo MJX Documentation

Official MuJoCo XLA (MJX) documentation covering the JAX-based hardware-accelerated MuJoCo API. Relevant to ariel for GPU-accelerated batched robot simulation during evolutionary search.

## Entity Pages Created

- [[mjx_overview]] — Installation, two implementations (MJX-JAX vs MJX-Warp), minimal example, full feature parity table
- [[mjx_core_functions]] — Core API: `put_model`, `make_data`, `put_data`, `step`; struct semantics for `mjx.Model`/`mjx.Data`
- [[mjx_warp]] — MJX-Warp: `GraphMode` enum, CUDA graph tuning, batch rendering API (`create_render_context`, `render`, `get_rgb`), multi-GPU with `pmap`
- [[mjx_performance]] — MJX-JAX performance tuning parameters, broadphase config, GPU env vars, sharp bits (single-scene slowdown, mesh vertex limits, contact scaling)

## Refresh 2026-10-19

Rechecked against MuJoCo 3.16.0 (installed headers and the page fetched 2026-10-19). Updated: mjx_performance: structural arrays and per-environment reset added.
