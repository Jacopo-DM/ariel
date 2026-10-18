---
type: concept_reference
tags: [mujoco, physics, astropy, research, concept]
source: three AI deep-research reports (Gemini, ChatGPT, DeepSeek) checked against the MuJoCo docs and the installed engine
date_ingested: 2026-10-18
---
# mujoco_parameter_research

Reconciled findings from three AI research reports on MuJoCo parameters, for the mujoco-units goal. The reports are secondary sources and contain errors. Every claim below is labelled: documented (MuJoCo docs, see [[mujoco_solver_parameters]], [[mjcf_option]]), measured (installed 3.8.0, see [[mjcf_defaults_3_8_0]]), consensus (all three agree, no primary page), judgement (an engineering proposal), or unverified.

## What the reports got wrong (checked against the installed engine)

| Parameter | Installed 3.8.0 | Wrong in |
|---|---|---|
| `ls_tolerance` | 0.01 | Gemini (1e-6) |
| `noslip_tolerance` | 1e-6 | Gemini (1e-5) |
| `ccd_iterations` | 35 | Gemini (50), DeepSeek (0) |
| `sleep_tolerance` | 1e-4 | Gemini (0.0, joules), ChatGPT (1e-3, the 3.11 value) |
| `magnetic` | 0 -0.5 0 | DeepSeek (0 0 0) |
| `compiler.degree` | 1 (degrees) | DeepSeek (false) |
| `o_friction` | five values 1 1 0.005 0.0001 0.0001 | Gemini (-1s), DeepSeek (three values) |
| `fusestatic` | 0 | ChatGPT (true) |

Formula errors: Gemini wrote `k = 1/(d0 tc^2 dr^2)` and `b = 2/(d0 tc)`; the docs have `k = d(r)/(dw^2 tc^2 dr^2)` and `b = 2/(dw tc)` with `dw = solimp[1]` ([[mujoco_solver_parameters]]). Gemini's piecewise `d(r)` uses an invented `dmax = d0 + d_width`; the docs say only `d(0)=d0`, `d(width)=dw`. ChatGPT's `k` omits `d(r)` but then scales by `d`, ending equivalent. DeepSeek's `k`, `b` match the docs. Gemini's convergence rule `||g|| <= tol * trace(M)` and its contact-stiffness chain `tc = 2 pi sqrt(m_eff / k_contact)` have no source.

## Agreed and documented

- `tc >= 2 * timestep` (documented, enforced by `refsafe`, `max(solref[0], 2*timestep)`). All three agree.
- Standard gravity 9.80665 m/s^2 (consensus; astropy `g0`). Air density about 1.204 kg/m^3 and viscosity about 1.81e-5 Pa s at 20 C (consensus; the MuJoCo option page gives about 1.2 and 2e-5 for air).
- `implicitfast` is the documented general recommendation (documented). The `Euler` default stays for compatibility.
- Elliptic cone with `impratio` above 1 reduces slip (documented, see [[mujoco_modeling_guidance]]); the reports differ on whether to use it by default (ChatGPT and DeepSeek: compare, Gemini: use 5.0). Judgement.
- MuJoCo 3.x sizes `nconmax`, `njmax`, `nstack` automatically (-1); MJX and Warp need fixed sizes. Consensus.
- The 14 ms robot-to-robot contact time constant is an ARIEL calibration (from the self-contact fix on `ci-group/ariel`, see PLAN.md), not a published value.

## Proposed derivation chain for the seeds (judgement, needs the user's approval)

Seeds: length scale `L` (0.05 m, a brick), top speed `v_max`, gravity `g0`, air density, air viscosity, sliding friction.

```
h <= alpha * L / v_max          advection limit, alpha = 0.05 (ChatGPT warns above 0.05, DeepSeek uses 0.2)
tc >= 2 * h                     documented (refsafe)
tc  = kappa * L / v_max         kappa about 0.28 reproduces 14 ms for L = 0.05 m, v_max = 1 m/s
h   = min(alpha * L / v_max, tc / 2, gamma / omega_max)
```

With `L = 0.05 m` and `v_max = 1 m/s`: `alpha*L/v = 2.5 ms`, `tc = 14 ms`, `tc/2 = 7 ms`. The third term needs the stiffest joint frequency `omega_max = sqrt(k / I_eff)` (position servo: `kp` and reflected inertia); `gamma` about 0.5 (DeepSeek) is a judgement. The constants `alpha`, `kappa`, `gamma` have no primary source; a convergence study (halve the timestep, compare trajectories and contact forces) should set them. Backed by: all three reports propose a minimum of contact, advection and joint limits; none gives a published universal formula.

## Plausible bands (warning only)

| Quantity | Band | Basis |
|---|---|---|
| timestep | 0.5 to 5 ms | judgement (agreed by the reports, DMC 2 to 5 ms) |
| `h / tc` | 0.01 to 0.5 | documented upper limit (0.5), lower is judgement |
| `h * v_max / L` | below 0.05 | judgement (ChatGPT) |
| gravity magnitude | 9 to 10 m/s^2 for Earth | judgement |
| air density, viscosity | 0.8 to 1.4 kg/m^3; 1.5e-5 to 2.2e-5 Pa s | judgement around the reference values |
| sliding friction | 0.2 to 1.5 | judgement; no primary tribology source for ABS or PLA on rubber, wood, concrete or carpet |
| `solref` damping ratio | 0.7 to 1.5 | judgement |
| `solimp` d0 | 0.8 to 0.98 | judgement (documented hard range 0.0001 to 0.9999) |
| joint damping, armature | 1e-3 to 0.1 N m s/rad; 0 to 1e-3 kg m^2 | judgement, depends on the servo |

## Reference servo values (unverified)

The reports disagree: SG90 class stall torque 0.18 N m (Gemini) against 0.1 to 0.3 N m class (ChatGPT); standard servo 1 N m (Gemini) against 0.2 to 0.6 N m (ChatGPT). ARIEL uses measured masses (stator 20 g, rotor 40 g, brick 55 g, core about 250 g in the Gemini report only). Do not use these as seeds without a datasheet.

## Open

- A pinned MuJoCo 3.8 changelog and XML reference (`/en/3.8.0/`) were cited by ChatGPT but not fetched; the installed engine replaces them for defaults.
- Whether `alpha`, `kappa`, `gamma` are seeds or fixed constants with a convergence test, and their values (PLAN.md, Open questions).

See also [[Source - Deep Research Reports on MuJoCo Parameters]].
