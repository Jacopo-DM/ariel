---
type: api_reference
tags: [mujoco, mjcf, physics]
source: https://mujoco.readthedocs.io/en/stable/computation/index.html
date_ingested: 2026-10-18
---
# mujoco_computation_soft_constraints

The Computation chapter: soft contact model, integrators, the constraint model and the solvers. **Version note.** The `/stable/` docs on 2026-10-18 describe MuJoCo 3.16.0 (changelog of 2026-10-14). This repo has MuJoCo 3.8.0 installed (`uv run python -c 'import mujoco; print(mujoco.__version__)'`) and `pyproject.toml` asks for `>=3.3.6`. Defaults and attributes below may differ for 3.8.0 (see [[mujoco_changelog_defaults]]); verify against the installed version before relying on a value.

## Soft contact model

MuJoCo drops the strict complementarity constraint of the LCP formulation, so contacts are soft (a convex approximation). The model is in continuous time (forces and accelerations); the opt-in `discrete` integrator solves constraints in discrete time and keeps stiff elements unconditionally stable. Frictional LCP is NP-hard in the worst case; for typical robot models 10 PGS sweeps are practically the global minimum.

## Equations

```
M dv/dt + c = tau + J^T f          (equation of motion)
forward:  dv/dt = M^-1 (tau + J^T f - c)
inverse:  tau = M dv/dt + c - J^T f
```

Constraint reference acceleration and regularizer:

```
a_ref,i = -b_i (J v)_i - k_i r_i            (spring-damper; b > 0 damping, k >= 0 stiffness)
R_ii    = ((1 - d_i) / d_i) * A^_ii          (d: impedance, 0 < d < 1; A^ approximate diagonal of A = J M^-1 J^T)
a_c     = A (A+R)^-1 a_ref + R (A+R)^-1 a_u  (R -> 0: hard constraint a_c = a_ref; R -> inf: constraint absent)
```

b and k come from `solref` and `solimp`, see [[mujoco_solver_parameters]]. For friction loss and elliptic friction dimensions r = 0 and k = 0 (pure damping). A^ uses "end-effector" inertias at `qpos0`; the `diagexact` flag uses the exact diagonal.

## Integrators

| Integrator | Notes |
|---|---|
| Euler | semi-implicit with implicit joint damping; only for compatibility with older models; explicit Euler is not supported |
| implicitfast | implicit terms except Coriolis and centripetal derivatives; "the recommended integrator for most models", similar cost to Euler, more stable |
| implicit | all derivatives except constraint forces, restricted to the sparsity of M; LU factorisation; helps coupled rotational systems such as multi-link pendula |
| RK4 | fixed-step 4th order; better than single-step methods for energy-conserving systems even with a 4x smaller step; for large velocity-dependent forces an implicit method can be more stable |
| discrete | new (September 2026), under development; constraints in discrete time with position stiffness; suggested for stiff springs, position servos or strong damping interacting with contacts |

Timestep: "perhaps the single most important parameter"; too large is unstable, too small wastes CPU; no numeric default or rule is given in this chapter (the XML reference default is 0.002 s, see [[mjcf_option]]). Euler-integrated filters diverge when their time constant is below the step; `filterexact` filters are stable for any positive time constant.

## Constraint model

Constraint types are ordered equality, friction loss, limit, contact. Contact parameters: `condim` (1, 3, 4, 6), `friction`, `margin`, `gap`, `solref`, `solimp`.

| condim | Elliptic | Pyramidal | Meaning |
|---|---|---|---|
| 1 | 1 | 1 | frictionless |
| 3 | 3 | 4 | tangential friction |
| 4 | 4 | 6 | adds torsional friction (units of length) |
| 6 | 6 | 10 | adds rolling friction (units of length) |

Margin and gap add across the two geoms. Distance > margin+gap: no contact; margin < distance <= margin+gap: inactive; distance <= margin: active, impedance applied to `distance - margin`. Negative margins are allowed if margin+gap >= 0. Friction cones: elliptic `f1 >= 0, f1^2 >= sum f_i^2 / mu_(i-1)^2`; pyramidal `f >= 0` (self-dual).

## Solvers

| Solver | Notes |
|---|---|
| Newton (default) | exact Newton with analytic second derivatives and Cholesky; stops early on cost improvement, gradient norm or Newton decrement below `tolerance` |
| CG | nonlinear conjugate gradient (Hager-Zhang), exact line search, no setup cost |
| PGS | projected Gauss-Seidel on the dual, first-order convergence, one sweep about one matrix-vector product; used to be the default |

All three work with both cones and with dense or sparse Jacobians. NoSlip (`noslip_iterations` > 0) is a post-processing pass that re-solves the friction dimensions with hard constraints; ad hoc, can destabilise complex multi-contact cases. Fluid forces are computed in the passive-forces stage.

Related: [[mjcf_option]], [[mujoco_modeling_guidance]].
