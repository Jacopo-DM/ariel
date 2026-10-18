---
type: api_reference
tags: [mujoco, mjcf, physics]
source: https://mujoco.readthedocs.io/en/stable/XMLreference.html
date_ingested: 2026-10-18
---
# mjcf_option

The `<option>` element of the XML reference (maps to `mjOption`, see [[MjOption]] and [[mujoco_overview_units]]). **Version note.** The `/stable/` docs on 2026-10-18 describe MuJoCo 3.16.0 (changelog of 2026-10-14). This repo has MuJoCo 3.8.0 installed (`uv run python -c 'import mujoco; print(mujoco.__version__)'`) and `pyproject.toml` asks for `>=3.3.6`. Defaults and attributes below may differ for 3.8.0 (see [[mujoco_changelog_defaults]]); verify against the installed version before relying on a value.

MuJoCo assigns no units; the dimension column is from the reference's dimension table (L length, T time, M mass, 1 dimensionless).

| Attribute | Type | Default | Dimension and meaning |
|---|---|---|---|
| `timestep` | real | 0.002 | T. Step size. For real time it must exceed CPU time per step (4x larger with RK4) |
| `impratio` | real | 1 | friction-to-normal impedance ratio for elliptic cones; above 1 reduces slow slippage |
| `gravity` | real(3) | "0 0 -9.81" | L T^-2. Z-up recommended for the GUI |
| `wind` | real(3) | "0 0 0" | L T^-1. Medium velocity subtracted before viscous, lift and drag forces |
| `magnetic` | real(3) | "0 -0.5 0" | global magnetic flux for magnetometers, in the site frame |
| `density` | real | 0 | M L^-3. Medium density for lift and drag; "about 1.2 for air and 1000 for water in SI"; 0 disables |
| `viscosity` | real | 0 | medium viscosity; "about 2e-5 for air and 9e-4 for water in SI"; 0 disables |
| `o_margin` | real | 0 | L. Replaces contact margin under the override flag |
| `o_solref`, `o_solimp`, `o_friction` | real arrays | not stated | replace solref (T,1 or T^-2,T^-1), solimp (1,1,L,1,1), friction (1,1,L,L,L) under override |
| `integrator` | Euler, RK4, implicit, implicitfast, discrete | "Euler" | numerical integrator |
| `cone` | pyramidal, elliptic | "pyramidal" | friction cone; elliptic is more physical, pyramidal can be faster and more robust |
| `jacobian` | dense, sparse, auto | "auto" | dense up to 60 degrees of freedom |
| `solver` | PGS, CG, Newton | "Newton" | constraint solver |
| `iterations` | int | 100 | maximum solver iterations |
| `tolerance` | real | 1e-8 | early termination; 0 disables |
| `ls_iterations` | int | 50 | line-search iterations (CG, Newton) |
| `ls_tolerance` | real | 0.01 | line-search early termination |
| `noslip_iterations` | int | 0 | NoSlip passes; 0 disables |
| `noslip_tolerance` | real | 1e-6 | NoSlip early termination |
| `ccd_iterations` | int | 35 | convex collision iterations |
| `ccd_tolerance` | real | 1e-6 | L. Convex collision early termination |
| `sleep_tolerance` | real | 1e-3 | L T^-1. Velocity below which sleeping is allowed |
| `sdf_iterations`, `sdf_initpoints` | int | 10, 40 | SDF collision |
| `actuatorgroupdisable` | int(31) | none | actuator groups (0 to 30) producing no force |

The integrator default in the reference is Euler while the Computation chapter recommends implicitfast ([[mujoco_computation_soft_constraints]]); ARIEL sets implicitfast in `MujocoConfig`.

## option/flag

Each flag is `[disable, enable]`. Default enable: `constraint`, `equality`, `frictionloss`, `limit`, `contact`, `spring`, `damper`, `gravity`, `clampctrl`, `warmstart`, `filterparent`, `actuation`, `refsafe`, `sensor` (description cut off), `midphase`, `nativeccd`, `island`, `eulerdamp`, `autoreset`, `multiccd`. Default disable: `override`, `energy`, `fwdinv`, `invdiscrete`, `sleep`, `diagexact`, `ipc` (experimental, `discrete` integrator and CG only).

- `refsafe`: uses `max(solref[0], 2*timestep)` per constraint; under `discrete` it replaces rows the timestep cannot resolve.
- `gravity` flag: sets gravity to zero at run time without changing the stored value.
- `eulerdamp` off disables implicit joint damping in Euler.

Related: [[mujoco_solver_parameters]], [[mjcf_compiler]], [[mjcf_size_statistic]].
