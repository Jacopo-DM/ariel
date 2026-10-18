---
type: api_reference
tags: [astropy, constants, physics]
source: https://docs.astropy.org/en/stable/constants/index.html
date_ingested: 2026-10-18
---
# astropy_constants

`astropy.constants`: physical and astronomical constants. A `Constant` is a [[Quantity]] with metadata: name, value, uncertainty, unit, reference (for example CODATA 2022 for `G`).

## Examples

```python
from astropy.constants import G
from astropy import constants as const

F = (const.G * 3.0 * const.M_sun * 100 * u.kg) / (2.2 * u.au) ** 2
print(F.to(u.N))
print(const.c.to("km/s"))
const.c.cgs
100 * const.e.esu
```

Some constants have other dimensions in CGS; using them without choosing a system (`.esu`, `.emu`) raises `TypeError`.

## Selected constants

| Symbol | Unit | Meaning |
|---|---|---|
| `g0` | m / s2 | standard acceleration of gravity |
| `G` | m3 / (kg s2) | gravitational constant |
| `c` | m / s | speed of light |
| `R` | J / (K mol) | gas constant |
| `k_B` | J / K | Boltzmann constant |
| `N_A` | 1 / mol | Avogadro's number |
| `atm` | Pa | standard atmosphere |
| `M_earth`, `R_earth` | kg, m | Earth mass and equatorial radius |
| `GM_earth` | m3 / s2 | nominal Earth mass parameter |

Not on the page: air density and viscosity. Those are not astropy constants.

## Version sets

- Modules: `astropyconst13` and `astropyconst20` (the page does not mention `astropyconst40`), CODATA (`codata2010`, `codata2014`, `codata2018`) and IAU (`iau2012`, `iau2015`) sets.
- `physical_constants.set()` and `astronomical_constants.set()` must run before importing `astropy.constants` or `astropy.units`, otherwise a `RuntimeError` is raised. `astropyconst13` selects `codata2010` and `iau2012`.
- For ARIEL: gravity should come from `g0` rather than a typed 9.81.
