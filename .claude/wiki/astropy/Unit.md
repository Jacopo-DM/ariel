---
type: api_reference
tags: [astropy, units, python]
source: https://docs.astropy.org/en/stable/units/combining_and_defining.html
date_ingested: 2026-10-18
---
# Unit

Units in `astropy.units`: how to combine them and how to define new ones. A [[Quantity]] is a number with a unit. Built-in units are listed in [[astropy_standard_units]].

## Signature

The page gives no full signatures for `Unit` or `def_unit` (see the astropy API reference). Forms it shows:

```python
u.Unit("erg / (s cm2)")                       # from a string
u.def_unit('bakers_fortnight', 13 * u.day)    # compound unit: name, definition
u.def_unit('titter')                          # new fundamental unit, no definition
u.def_unit(['Fo', 'foo'], prefixes=True, namespace=new_units)  # names, prefixes, namespace dict
u.add_enabled_units(new_units)                # make custom units findable and parseable
```

## Examples

```python
>>> fluxunit = u.erg / (u.cm ** 2 * u.s)
>>> fluxunit
Unit("erg / (s cm2)")
>>> 52.0 * fluxunit / u.s
<Quantity  52. erg / (cm2 s2)>

>>> from fractions import Fraction
>>> Franklin = u.g ** Fraction(1, 2) * u.cm ** Fraction(3, 2) * u.s ** -1

>>> chuckle = u.def_unit('chuckle', 5 * titter)
>>> rofl.represents
Unit("4 guffaw")
>>> rofl.decompose()
Unit("240 titter")

>>> kmph = u.def_unit('kmph', u.km / u.h)
>>> u.add_enabled_units([kmph])
>>> print(u.Unit("megafoo").to(u.Unit("kFo")))
1000.0
```

## Notes

- Units combine with the usual Python operators. Use `fractions.Fraction` for fractional powers; a float such as `u.m ** (1. / 3.)` with a denominator under 10 becomes a `Fraction` automatically.
- Custom units are left out of searches such as `find_equivalent_units()` until enabled with `add_enabled_units()`.
- `represents` shows the definition; `decompose()` reduces to irreducible units.
- The page does not mention `dimensionless_unscaled`; `u.m / u.m` gives `Unit(dimensionless)` (see [[astropy_units_overview]]).
- Related: [[astropy_unit_formats]] for strings, [[PhysicalType]].
