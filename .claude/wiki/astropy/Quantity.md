---
type: api_reference
tags: [astropy, units, quantity, python]
source: https://docs.astropy.org/en/stable/units/quantity.html
date_ingested: 2026-10-18
---
# Quantity

A `Quantity` is a number or array with a unit, in `astropy.units`. It subclasses `numpy.ndarray`, so NumPy functions return quantities with units. Units come from [[Unit]]; the page list is in [[astropy_units_overview]].

## Signature

The page gives no full signature (see the astropy API reference). Forms it shows:

```python
from astropy import units as u

15 * u.m / u.s  # number times unit
1.25 / u.s
[1, 2, 3] * u.m  # list or ndarray times unit
u.Quantity(15, u.m / u.s)  # constructor: value, unit
u.Quantity(qlst, u.minute)  # merge quantities with equivalent units
u.Quantity("15 m/s")  # parse a string
u.Quantity(1.0, dtype=np.float32)  # explicit dtype
```

## Parameters / attributes

| Name | Description |
|---|---|
| `.value` | the number or array without its unit |
| `.unit` | the unit, a [[Unit]] |
| `dtype` | pass it explicitly; integers are upcast to float by default, `dtype=None` keeps NumPy's integer inference |
| `copy` | multiplying by a unit copies the data; `copy=False` avoids it |

## Conversion

| Call | Result |
|---|---|
| `q.to(unit)` | converted copy, always copies; `(2.3 * u.m / u.s).to(u.km / u.h)` is `8.28 km / h` |
| `q.to_value(unit)` | only the number; `(2.5 * u.m).to_value(u.cm)` is `250.0`; does not copy when the unit already matches |
| `q.si`, `q.cgs` | convert to base SI or CGS units |
| `q.decompose()` | simplify a result; `(20. * u.cm / (1. * u.m)).decompose()` is `0.2` |
| `a << u.m` | view of an array with units, no copy; `q <<= u.cm` converts in place |

## Arithmetic

- Addition and subtraction need equivalent units; if they differ, the result takes the unit of the left operand.
- Adding a plain number to a non-dimensionless quantity raises `UnitConversionError` (zero, infinity and NaN are exceptions).
- Multiplication and division work with any units and make composite units; use `.to(u.m**2)` to choose the output unit.
- Comparison: `1*u.m < 50*u.cm` is `False`; use `u.allclose()` and `u.isclose()`.

## Dimensionless quantities and Python scalars

- A dimensionless quantity is simplified when added to a plain number or passed to NumPy functions such as `np.exp`: `1. + 1. * u.m / u.km` is `1.001`. Taking `.value` first gives another result.
- `float()` and `int()` work only on dimensionless quantities; `float(3. * u.m)` raises `TypeError`.
- Angle functions accept angle units: `np.sin(30. * u.deg)` is `0.5`.

## Functions that take quantities

`u.quantity_input` checks that arguments have compatible units or physical types. It does not convert them. Expectations are keyword arguments (`@u.quantity_input(myarg=u.deg)`) or annotations (`myarg: u.Quantity[u.arcsec]`, a physical-type string such as `'angle'`, a list of accepted types, `None` defaults, return annotations such as `-> u.deg`).

## Notes

- SciPy support is incomplete.
- Subclassing: override `__new__()` not `__init__()`, use `__array_finalize__()`, see `SpecificTypeQuantity` for physical-type-bound subclasses.
- `Quantity` objects can be table columns (`QTable`).
- Performance: `array << u.m / u.s` took 47 microseconds against 92.5 ms for `array * u.m / u.s / u.kg / u.sr` on an array of 10 million floats (page benchmark), see [[astropy_units_overview]].
- Related: [[PhysicalType]], [[astropy_unit_equivalencies]], [[astropy_unit_formats]], [[astropy_constants]].
