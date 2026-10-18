---
type: api_reference
tags: [astropy, units, overview]
source: https://docs.astropy.org/en/stable/units/index.html
date_ingested: 2026-10-18
---
# astropy_units_overview

`astropy.units` defines, converts between and does arithmetic on physical quantities, including logarithmic units (magnitude, dex). It does not handle spherical geometry or sexagesimal units (see `astropy.coordinates`).

## Sub-pages

| Page | Content | Wiki page |
|---|---|---|
| Quantity | create, convert, compare, NumPy, QTable, subclassing | [[Quantity]] |
| Standard Units | SI and other unit sets, prefixes | [[astropy_standard_units]] |
| Combining and Defining Units | compound units, fractional powers, `def_unit` | [[Unit]] |
| String Representations | formats and parsing | [[astropy_unit_formats]] |
| Equivalencies | built-in and custom equivalencies | [[astropy_unit_equivalencies]] |
| Physical Types | dimensional analysis | [[PhysicalType]] |
| Constants | `astropy.constants` | [[astropy_constants]] |

Not ingested: type annotations (`astropy.units.typing`), decomposing and composing units, logarithmic units, structured units, prior constant versions, low-level conversion, the API reference.

## Examples

```python
>>> 15.1 * u.meter / (32.0 * u.second)
<Quantity 0.471875 m / s>
>>> (3.0 * u.kilometer / (130.51 * u.meter / u.second)).decompose()
<Quantity 22.986744310780782 s>
>>> u.m / u.m
Unit(dimensionless)
>>> (1.0 * u.Pa).cgs
<Quantity 10. Ba>
>>> (1000 * u.nm).to(u.Hz)
UnitConversionError: 'nm' (length) and 'Hz' (frequency) are not convertible
>>> (u.s ** -1).compose()
[Unit("Bq"), Unit("Hz"), Unit("2.7027e-11 Ci")]
```

## Performance (page benchmark, array of 10 million floats)

| Expression | Time |
|---|---|
| `array * u.m / u.s / u.kg / u.sr` | 92.5 ms |
| `array * (u.m / u.s / u.kg / u.sr)` | 21.5 ms |
| `array << u.m / u.s / u.kg / u.sr` | 47.1 microseconds |
| `array << UNIT_MSKGSR` (precomputed) | 6.51 microseconds |

Parenthesise or precompute composite units, and use `<<` to avoid copying.
