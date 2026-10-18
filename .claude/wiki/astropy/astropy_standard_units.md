---
type: api_reference
tags: [astropy, units, si]
source: https://docs.astropy.org/en/stable/units/standard_units.html
date_ingested: 2026-10-18
---
# astropy_standard_units

Built-in units and prefixes in `astropy.units`. All units are defined in terms of basic irreducible units; units built from them are `CompositeUnit` instances.

## Irreducible units

| Quantity | Unit |
|---|---|
| Length | meter |
| Time | second |
| Mass | kilogram |
| Current | ampere |
| Temperature | Kelvin |
| Angular distance | radian |
| Solid angle | steradian |
| Luminous intensity | candela |
| Stellar magnitude | mag |
| Amount of substance | mole |
| Photon count | photon |

The page does not list force, pressure, density, viscosity, velocity or acceleration units; use composite units (`u.kg / (u.m * u.s)`) and check them with [[PhysicalType]].

## Prefixes

Decimal prefixes run from quetta (1e30) to quecto (1e-30), for example k (1e3), M (1e6), G (1e9), c (1e-2), m (1e-3), u (1e-6), n (1e-9). Binary prefixes (Ki = 2**10 up to Yi = 2**80) apply to `bit` and `byte`.

## find_equivalent_units

```python
>>> u.g.find_equivalent_units()
  Primary name | Unit definition | Aliases
[
  M_e          | 9.10938e-31 kg  |                                  ,
  g            | 0.001 kg        | gram                             ,
  kg           | irreducible     | kilogram                         ,
  t            | 1000 kg         | tonne                            ,
  ...
]
```

By default only SI, CGS and astrophysical units are searched. Enable more with `imperial.enable()` or `add_enabled_units()` (see [[Unit]]).

## Notes

- Related: [[Quantity]], [[astropy_constants]].
