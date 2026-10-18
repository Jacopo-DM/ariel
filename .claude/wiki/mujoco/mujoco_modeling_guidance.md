---
type: api_reference
tags: [mujoco, mjcf, physics]
source: https://mujoco.readthedocs.io/en/stable/modeling.html
date_ingested: 2026-10-18
---
# mujoco_modeling_guidance

Practical advice from the Modeling chapter. Only the first 100000 of 137076 characters were read for the first two topics, then a targeted second pass for the solver formulas ([[mujoco_solver_parameters]]). **Version note.** The `/stable/` docs on 2026-10-18 describe MuJoCo 3.16.0 (changelog of 2026-10-14). This repo has MuJoCo 3.8.0 installed (`uv run python -c 'import mujoco; print(mujoco.__version__)'`) and `pyproject.toml` asks for `>=3.3.6`. Defaults and attributes below may differ for 3.8.0 (see [[mujoco_changelog_defaults]]); verify against the installed version before relying on a value.

## Default classes

Classes nest; a child inherits every attribute of its parent and may override any. An element uses its own `class`, else the nearest ancestor body's `childclass`, else the top-level class. An attribute defined in a class cannot be set back to undefined in that class or its children. Body inertia can be left undefined so the compiler infers it from the geoms.

## Timestep and integrator

- The timestep is "the single most important parameter in any model". The default is chosen for stability, not efficiency: "increase it until divergence occurs, then back off to the largest value where divergence is never or very rarely seen."
- The recommended integrator is `implicitfast`. Explicit damping can feed vibration, so prefer implicit or implicitfast.

## Solver settings

- Jacobian `auto`: dense below 60 degrees of freedom, sparse above.
- Newton is the best choice for most models: usually about 5 iterations, rarely over 20, tolerances around 1e-10. CG when Newton gets slow (large models, elliptic cones, many slipping contacts). PGS when degrees of freedom exceed the constraint count.
- To suppress slip: elliptic cones, a large `impratio`, Newton, a very small tolerance; if slip persists enable NoSlip. With elliptic cones a larger `impratio` makes friction harder relative to the normal direction without raising the coefficient; it does not guarantee exact sticking.

## Joints and actuators

- `armature` (reflected rotor inertia = rotor inertia x gear^2, gear 3 gives 9): positive values improve stability even when small; needed on one of two split joints, otherwise the joint-space inertia matrix is singular.
- Force limits: `ctrlrange` clamps the input, `forcerange` clamps the actuator output, `actuatorfrcrange` on a joint or tendon clamps the summed input. Position actuators usually also need control range clamping.
- `position` creates a servo with gain `kp`; the parser sets the general gain and bias parameters. Shortcut parameters inherit only from defaults written with the same shortcut or a matching general default.
- `actrange` for `intvelocity` is in radians whatever the joint angle unit.
- Muscles: default timeconst (0.01, 0.04), range (0.75, 1.05); start with the defaults and adjust `scale` in a default class.
- A `delay` longer than `timestep * nsample` reads non-causal data without an error.

## Unstable or vibrating simulations

Visualise contact forces in simulate. Usual fixes: reduce the timestep and/or add armature to the relevant joints; use `implicit` or `implicitfast`; avoid explicit damping as a vibration source. Constraint accuracy: impedance uses a compile-time diagonal approximation from `qpos0`; symptoms are `badqacc` warnings, excess penetration and unrealistic slip; check the `fwdinv` flag for a large forward-inverse gap and enable `diagexact` for the exact diagonal. Near-perfect restitution: large stiffness, zero damping, for example `solref="-1000 0"`.

Related: [[mjcf_option]], [[mjcf_compiler]], [[mjcf_actuators]].
