"""Physical quantities in pydantic records: validation and unit conversion.

A field typed with ``quantity_of(unit)`` takes an astropy quantity of the
same dimension as ``unit``, refuses a plain number or another dimension, and
stores the value converted to ``unit``, so every later formula reads one
canonical unit.
"""

# Third-party libraries
import astropy.units as u
import numpy as np
from pydantic import PlainValidator

# Global constants
STANDARD_GRAVITY = 9.80665  # m / s^2, exact by definition (ISO 80000-3)

# Radians count as dimensionless, so rad / s and 1 / s convert to each other
ANGLES = u.dimensionless_angles()

# Kilogram-force, the "kg" of servo datasheets ("13.5 kg*cm" is a torque);
# astropy has no such unit, and kg * cm would be a mass times a length
KILOGRAM_FORCE = u.def_unit("kgf", STANDARD_GRAVITY * u.kg * u.m / u.s**2)


def to_unit(value: u.Quantity, unit: u.UnitBase) -> u.Quantity:
    """Convert a quantity to a unit, treating radians as dimensionless.

    The one place that calls astropy's untyped ``Quantity.to``.

    Parameters
    ----------
    value : astropy.units.Quantity
        The quantity.
    unit : astropy.units.UnitBase
        The target unit.

    Returns
    -------
    astropy.units.Quantity
        The quantity in ``unit``.
    """
    converted: u.Quantity = value.to(unit, equivalencies=ANGLES)  # type: ignore[no-untyped-call]  # astropy ships no type hints
    return converted


def quantity_of(
    unit: u.UnitBase,
) -> PlainValidator:
    """Build a validator for a positive, finite, scalar quantity.

    Parameters
    ----------
    unit : astropy.units.UnitBase
        The canonical unit; the value is converted to it.

    Returns
    -------
    PlainValidator
        A pydantic validator that returns the value in ``unit``.
    """

    def check(value: object) -> u.Quantity:
        if not isinstance(value, u.Quantity):
            msg = f"needs an astropy quantity in {unit}, got {value!r}"
            raise ValueError(msg)  # ruff: ignore[type-check-without-type-error]  # pydantic only reports ValueError
        # A Quantity always has a unit; the stub allows None
        if value.unit is None or not value.unit.is_equivalent(
            unit, equivalencies=ANGLES
        ):
            msg = f"needs the dimension of {unit}, got {value.unit}"
            raise ValueError(msg)
        if not value.isscalar:
            msg = "needs a scalar quantity"
            raise ValueError(msg)
        converted = to_unit(value, unit)
        number = float(converted.value)
        if not (np.isfinite(number) and number > 0.0):
            msg = f"needs a positive and finite value, got {value}"
            raise ValueError(msg)
        return converted

    return PlainValidator(check)
