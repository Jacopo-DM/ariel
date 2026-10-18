---
type: api_reference
tags: [astropy, units, physical-type]
source: https://docs.astropy.org/en/stable/units/physical_types.html
date_ingested: 2026-10-18
---
# PhysicalType

Physical types group quantities whose units are dimensionally compatible, for example *mass* covers anything convertible to kilograms. Each is a `PhysicalType` instance.

## Signature

```python
unit.physical_type            # attribute of a Unit
u.get_physical_type(obj)      # obj: a name, unit, Quantity, number or PhysicalType
```

## Examples

```python
>>> u.coulomb.physical_type
PhysicalType('electrical charge')
>>> u.get_physical_type('speed')
PhysicalType({'speed', 'velocity'})
>>> u.get_physical_type(42)
PhysicalType('dimensionless')
>>> acceleration = u.get_physical_type(u.m / u.s ** 2)
>>> acceleration == 'acceleration'
True
>>> for name in u.J.physical_type: print(name)
energy
torque
work
>>> length = u.get_physical_type('length')
>>> 1 / u.get_physical_type('time')
PhysicalType('frequency')
>>> length / 'time'
PhysicalType({'speed', 'velocity'})
```

## Notes

- A unit can have several physical types (a joule is energy, torque and work), so compare with `==` against a name, iterate, or test `'work' in u.J.physical_type`.
- Multiplication, division and exponentiation give dimensional analysis on types, units or names.
- Use it to check the dimension of a seed value (see [[Quantity]], where `u.quantity_input` also accepts physical-type strings).
