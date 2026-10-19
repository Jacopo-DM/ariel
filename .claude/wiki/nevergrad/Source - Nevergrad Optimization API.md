---
type: source_summary
tags: [nevergrad, optimization, cma-es, ask-tell, parametrization]
source: https://facebookresearch.github.io/nevergrad/optimization.html
date_ingested: 2026-10-19
title: "Nevergrad Optimization API"
---

# Source - Nevergrad Optimization API

Nevergrad is Facebook Research's gradient-free optimization library. Covers optimizer selection, ask/tell interface, parametrization, parallel evaluation, constraints, and multiobjective optimization.

## Entity pages created

- [[Nevergrad_Optimizers]] — Optimizer classes (CMA, NGOpt, DE, etc.), selection guide, batch mode, executor integration
- [[Nevergrad_Parametrization]] — ng.p.* classes (Array, Scalar, Tuple, Dict, Instrumentation, Choice, Log), set_mutation, set_bounds, composition patterns
- [[Nevergrad_ML_Patterns]] — Optimizer selection by problem type, synchronous/async ask-tell, budget guidelines, RL and noisy objective advice

## Refresh 2026-10-19

Rechecked against MuJoCo 3.16.0 (installed headers and the page fetched 2026-10-19). Updated: Nevergrad_Optimizers: registry, rules of thumb, inoculation and callbacks added.
