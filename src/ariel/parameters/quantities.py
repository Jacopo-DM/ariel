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


def _read(text: str, unit: u.UnitBase) -> u.Quantity:
    """Read a quantity written as text, such as ``"59.5 g"``.

    Parameters
    ----------
    text : str
        A number followed by its unit (an environment variable, for example).
    unit : astropy.units.UnitBase
        The unit the field needs, named in the error message.

    Returns
    -------
    astropy.units.Quantity
        The quantity the text describes.

    Raises
    ------
    ValueError
        If the text is not a number with a unit astropy knows.
    """
    try:
        return u.Quantity(text)
    except (TypeError, ValueError):
        msg = f"needs a number and a unit such as '1 {unit}', got {text!r}"
        raise ValueError(msg) from None


def quantity_of(
    unit: u.UnitBase,
    *,
    allow_negative: bool = False,
) -> PlainValidator:
    """Build a validator for a finite, scalar quantity (positive by default).

    A plain number is refused, and so is a value without a unit where the
    field has one: radians count as dimensionless in conversions, but an
    angle still has to be written with its unit.

    Parameters
    ----------
    unit : astropy.units.UnitBase
        The canonical unit; the value is converted to it.
    allow_negative : bool
        Accept zero and negative values (an angle), not only positive ones.

    Returns
    -------
    PlainValidator
        A pydantic validator that returns the value in ``unit``. It also reads
        text with a unit (``"59.5 g"``), as environment variables give it.
    """
    needs_unit = unit.physical_type != "dimensionless"

    def check(value: object) -> u.Quantity:
        if isinstance(value, str):
            value = _read(value, unit)
        if not isinstance(value, u.Quantity):
            msg = f"needs an astropy quantity in {unit}, got {value!r}"
            raise ValueError(msg)  # ruff: ignore[type-check-without-type-error]  # pydantic only reports ValueError
        # A Quantity always has a unit; the stub allows None
        given = value.unit
        if (
            given is None
            or not given.is_equivalent(unit, equivalencies=ANGLES)
            or (needs_unit and given.physical_type == "dimensionless")
        ):
            shown = str(given) or "no unit"
            msg = f"needs the dimension of {unit}, got {shown}"
            raise ValueError(msg)
        if not value.isscalar:
            msg = "needs a scalar quantity"
            raise ValueError(msg)
        converted = to_unit(value, unit)
        number = float(converted.value)
        if not np.isfinite(number) or (not allow_negative and number <= 0.0):
            kind = "finite" if allow_negative else "positive and finite"
            msg = f"needs a {kind} value, got {value}"
            raise ValueError(msg)
        return converted

    return PlainValidator(check)


def metres(length: u.Quantity) -> float:
    """Return a length as a plain number of metres, MuJoCo's unit.

    Parameters
    ----------
    length : astropy.units.Quantity
        A length in any unit.

    Returns
    -------
    float
        The length in metres.
    """
    return float(to_unit(length, u.m).value)


def kilograms(mass: u.Quantity) -> float:
    """Return a mass as a plain number of kilograms, MuJoCo's unit.

    Parameters
    ----------
    mass : astropy.units.Quantity
        A mass in any unit.

    Returns
    -------
    float
        The mass in kilograms.
    """
    return float(to_unit(mass, u.kg).value)


def radians(angle: u.Quantity) -> float:
    """Return an angle as a plain number of radians, MuJoCo's unit.

    Parameters
    ----------
    angle : astropy.units.Quantity
        An angle in any unit.

    Returns
    -------
    float
        The angle in radians.
    """
    return float(to_unit(angle, u.rad).value)
