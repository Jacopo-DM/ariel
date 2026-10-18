---
type: api_reference
tags: [mujoco, mjcf, physics]
source: https://mujoco.readthedocs.io/en/stable/modeling.html
date_ingested: 2026-10-18
---
# mujoco_solver_parameters

`solref` and `solimp`: how contact, limit and equality softness is parameterised. Source: the "Solver parameters" section of the Modeling chapter. **Version note.** The `/stable/` docs on 2026-10-18 describe MuJoCo 3.16.0 (changelog of 2026-10-14). This repo has MuJoCo 3.8.0 installed (`uv run python -c 'import mujoco; print(mujoco.__version__)'`) and `pyproject.toml` asks for `>=3.3.6`. Defaults and attributes below may differ for 3.8.0 (see [[mujoco_changelog_defaults]]); verify against the installed version before relying on a value.

## solimp (impedance), real(5), default "0.9 0.95 0.001 0.5 2"

Values `d0`, `dw`, `width`, `midpoint`, `power` parameterise the impedance `d(r)` as a function of the constraint violation `r`.

- `d(0) = d0`, `d(width) = dw`; `d` is a smooth sigmoid-like interpolation controlled by `midpoint` (0 to 1, in units of `width`) and `power` (>= 1; power 1 is linear whatever the midpoint). The page does not give the explicit spline.
- `d` is clamped to [mjMINIMP, mjMAXIMP], currently [0.0001, 0.9999].
- For smooth dynamics set `solimp[0] = 0` for limits and contacts.
- The constraint relation: `ac + d (b v + k r) = (1 - d) au`; the solver interpolates between the unforced acceleration `au` and the reference acceleration `ar`.
- Friction constraints have zero position violation: impedance is always `solimp[0]`, stiffness is zero, `solimp[1]` still affects damping.

## solref (reference), real(2), default "0.02 1"

Positive format `(timeconst, dampratio)`:

```
b = 2 / (dw * timeconst)
k = d(r) / (dw^2 * timeconst^2 * dampratio^2)
resting penetration (constant d = d0 = dw):  r = au * (1 - d) * timeconst^2 * dampratio^2
```

`dampratio = 1` is critical damping.

Direct format, both numbers negative `(-stiffness, -damping)`:

```
b = damping / dw
k = stiffness * d(r) / dw^2
resting penetration:  r = au * (1 - d) / stiffness
```

The two forms relate by `dampratio = 1` being `damping = 2 * sqrt(stiffness)`.

## Timestep rule

"The timeconst parameter should be at least two times larger than the simulation time step", "otherwise the system can become too stiff relative to the numerical integrator". It is enforced internally unless the `refsafe` flag is disabled; the option page defines refsafe as using `max(solref[0], 2 * timestep)` per constraint ([[mjcf_option]]).

## Combining the two geoms of a contact

| Parameter | Rule |
|---|---|
| `condim` | higher `priority` wins, else the maximum |
| `friction` | higher priority wins, else element-wise maximum |
| `margin`, `gap` | summed, priority ignored |
| `solref`, `solimp` | higher priority wins; at equal priority a weighted average with weights from `solmix`; if either `solref` is non-positive (direct format) the element-wise minimum is used |

The `override` flag with `o_margin`, `o_solref`, `o_solimp` changes contact parameters for all pairs without editing the model.

Related: [[mujoco_computation_soft_constraints]], [[mjcf_geom_joint_defaults]].
