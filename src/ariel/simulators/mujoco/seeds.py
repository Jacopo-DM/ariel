"""Physical-scale seeds from which the MuJoCo settings are derived.

A seed is a physical quantity (an astropy quantity) or a plain positive
number. The model is frozen: ``replace`` returns a new object and validates
the change, which ``model_copy(update=...)`` does not.

Notes
-----
    * Quantities are stored in one canonical unit, so every later formula
        reads plain SI numbers.
    * The tuning seeds ``alpha``, ``kappa`` and ``gamma`` have no primary
        source; see ``.claude/wiki/mujoco/mujoco_parameter_research.md``.
"""

# Standard library
from collections.abc import Callable
from typing import Annotated, Literal, Self

# Third-party libraries
import astropy.units as u
import numpy as np
from pydantic import Field, PlainValidator, computed_field

# Local libraries
from ariel.parameters import FrozenModel

# Global constants
STANDARD_GRAVITY = 9.80665  # m / s^2, exact by definition (ISO 80000-3)
AIR_DENSITY_20C = 1.204  # kg / m^3, consensus of the research reports
AIR_VISCOSITY_20C = 1.81e-5  # Pa s, consensus of the research reports
DEFAULT_LENGTH_SCALE = 0.05  # m, one brick
DEFAULT_TOP_SPEED = 1.0  # m / s
DEFAULT_JOINT_FREQUENCY = 100.0  # rad / s, about 16 Hz for a stiff servo
DEFAULT_FRICTION = 1.0  # MuJoCo's own default
DEFAULT_ALPHA = 0.05  # advection limit
DEFAULT_KAPPA = 0.28  # contact time constant, 14 ms for the default scales
DEFAULT_GAMMA = 0.5  # joint-frequency limit


_ANGLES = u.dimensionless_angles()


def _to(value: u.Quantity, unit: u.UnitBase) -> u.Quantity:
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
    converted: u.Quantity = value.to(unit, equivalencies=_ANGLES)  # type: ignore[no-untyped-call]  # astropy ships no type hints
    return converted


def _quantity_of(
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
        if not value.unit.is_equivalent(unit, equivalencies=_ANGLES):
            msg = f"needs the dimension of {unit}, got {value.unit}"
            raise ValueError(msg)
        if not value.isscalar:
            msg = "needs a scalar quantity"
            raise ValueError(msg)
        converted = _to(value, unit)
        number = float(converted.value)
        if not (np.isfinite(number) and number > 0.0):
            msg = f"needs a positive and finite value, got {value}"
            raise ValueError(msg)
        return converted

    return PlainValidator(check)


def _positive(value: float) -> float:
    """Check that a plain seed is positive and finite.

    Parameters
    ----------
    value : float
        The seed.

    Returns
    -------
    float
        The same value.

    Raises
    ------
    ValueError
        If the value is not positive and finite.
    """
    if not (np.isfinite(value) and value > 0.0):
        msg = f"needs a positive and finite value, got {value}"
        raise ValueError(msg)
    return value


type Length = Annotated[u.Quantity, _quantity_of(u.m)]
type Speed = Annotated[u.Quantity, _quantity_of(u.m / u.s)]
type AngularFrequency = Annotated[u.Quantity, _quantity_of(u.rad / u.s)]
type Acceleration = Annotated[u.Quantity, _quantity_of(u.m / u.s**2)]
type Density = Annotated[u.Quantity, _quantity_of(u.kg / u.m**3)]
type Viscosity = Annotated[u.Quantity, _quantity_of(u.Pa * u.s)]
type TimestepLimit = Literal["advection", "contact", "joint"]
type Positive = Annotated[float, PlainValidator(_positive)]


def _default(number: float, unit: u.UnitBase) -> Callable[[], u.Quantity]:
    """Return a factory for a default quantity (quantities are mutable).

    Parameters
    ----------
    number : float
        The value.
    unit : astropy.units.UnitBase
        Its unit.

    Returns
    -------
    Callable[[], astropy.units.Quantity]
        A function that builds a fresh quantity on every call.
    """
    return lambda: u.Quantity(number, unit)


class Seeds(FrozenModel):
    """The physical scales of a simulated world."""

    length_scale: Length = Field(
        default_factory=_default(DEFAULT_LENGTH_SCALE, u.m),
        description="size of one brick",
    )
    top_speed: Speed = Field(
        default_factory=_default(DEFAULT_TOP_SPEED, u.m / u.s),
        description="fastest speed a body part is expected to reach",
    )
    joint_frequency: AngularFrequency = Field(
        default_factory=_default(DEFAULT_JOINT_FREQUENCY, u.rad / u.s),
        description="natural frequency of the stiffest joint",
    )
    gravity: Acceleration = Field(
        default_factory=_default(STANDARD_GRAVITY, u.m / u.s**2),
        description="standard gravity",
    )
    air_density: Density = Field(
        default_factory=_default(AIR_DENSITY_20C, u.kg / u.m**3),
    )
    air_viscosity: Viscosity = Field(
        default_factory=_default(AIR_VISCOSITY_20C, u.Pa * u.s),
    )
    friction: Positive = DEFAULT_FRICTION
    alpha: Positive = DEFAULT_ALPHA
    kappa: Positive = DEFAULT_KAPPA
    gamma: Positive = DEFAULT_GAMMA

    def replace(self, **changes: object) -> Self:
        """Return a copy with some seeds changed, validated again.

        Parameters
        ----------
        **changes : object
            New values by seed name.

        Returns
        -------
        Seeds
            A new object; derived values recompute from it.
        """
        seeds = self.model_dump(exclude=set(type(self).model_computed_fields))
        return type(self).model_validate(seeds | changes)

    @computed_field  # type: ignore[prop-decorator]  # pydantic documents this stacking
    @property
    def contact_time_constant(self) -> float:
        """Contact softness time constant ``tc`` in seconds.

        ``tc = kappa * L / v_max``. Judgement: ``kappa`` 0.28 reproduces the
        14 ms that ARIEL calibrated for robot-to-robot contact.
        """
        return self.kappa * self._length_over_speed

    @computed_field  # type: ignore[prop-decorator]  # pydantic documents this stacking
    @property
    def advection_limit(self) -> float:
        """Largest step in seconds before a part crosses ``alpha`` lengths.

        ``alpha * L / v_max``. Judgement, see the research page.
        """
        return self.alpha * self._length_over_speed

    @computed_field  # type: ignore[prop-decorator]  # pydantic documents this stacking
    @property
    def joint_limit(self) -> float:
        """Largest step in seconds for the stiffest joint.

        ``gamma / omega_max``. Judgement, see the research page.
        """
        return self.gamma / float(self.joint_frequency.value)

    @computed_field  # type: ignore[prop-decorator]  # pydantic documents this stacking
    @property
    def timestep(self) -> float:
        """Integration step ``h`` in seconds.

        ``min(alpha * L / v_max, tc / 2, gamma / omega_max)``. The ``tc / 2``
        term is documented: MuJoCo clamps ``tc`` to at least ``2 h``
        (``refsafe``).
        """
        return min(
            self.advection_limit,
            self.contact_time_constant / 2,
            self.joint_limit,
        )

    @computed_field  # type: ignore[prop-decorator]  # pydantic documents this stacking
    @property
    def timestep_limit(self) -> TimestepLimit:
        """Name of the term that sets the timestep."""
        limits: dict[TimestepLimit, float] = {
            "advection": self.advection_limit,
            "contact": self.contact_time_constant / 2,
            "joint": self.joint_limit,
        }
        return min(limits, key=limits.__getitem__)

    @computed_field  # type: ignore[prop-decorator]  # pydantic documents this stacking
    @property
    def timestep_over_contact(self) -> float:
        """Ratio ``h / tc``; the documented limit is 0.5 (``refsafe``)."""
        return self.timestep / self.contact_time_constant

    @computed_field  # type: ignore[prop-decorator]  # pydantic documents this stacking
    @property
    def courant(self) -> float:
        """Fraction of ``L`` that a part at ``v_max`` moves in one step."""
        return self.timestep / self._length_over_speed

    @property
    def _length_over_speed(self) -> float:
        """``L / v_max`` in seconds."""
        return float(self.length_scale.value) / float(self.top_speed.value)
