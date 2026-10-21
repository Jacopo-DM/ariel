"""The servo that drives every hinge: DFRobot DSS-M15 (SKU SER0019).

Every value is an astropy quantity written in the datasheet's own unit and
stored in SI, so a torque in kilogram-force centimetres can never be read as
newton metres.

References
----------
    [1] DFRobot, "DF Metal Geared 15Kg Standard Servo 180 (DSS-M15)",
        SKU SER0019, https://www.dfrobot.com/product-120.html (the datasheet
        PDF in the user's uploads, 2026-10-21).
"""

# Standard library
from collections.abc import Callable
from typing import Annotated

# Third-party libraries
import astropy.units as u
from pydantic import ConfigDict, Field

# Local libraries
from ariel.parameters.base_model import FrozenModel
from ariel.parameters.quantities import KILOGRAM_FORCE, quantity_of

type Angle = Annotated[u.Quantity, quantity_of(u.rad)]
type AngularSpeed = Annotated[u.Quantity, quantity_of(u.rad / u.s)]
type Current = Annotated[u.Quantity, quantity_of(u.A)]
type Duration = Annotated[u.Quantity, quantity_of(u.s)]
type Frequency = Annotated[u.Quantity, quantity_of(u.Hz)]
type Length = Annotated[u.Quantity, quantity_of(u.m)]
type Torque = Annotated[u.Quantity, quantity_of(u.N * u.m)]
type Voltage = Annotated[u.Quantity, quantity_of(u.V)]


def _quantity(value: float, unit: u.UnitBase) -> Callable[[], u.Quantity]:
    """Return a factory for a default quantity (quantities are mutable).

    Parameters
    ----------
    value : float
        The number, in ``unit``.
    unit : astropy.units.UnitBase
        The unit the datasheet gives.

    Returns
    -------
    Callable[[], astropy.units.Quantity]
        A function that builds a fresh quantity on every call.
    """
    return lambda: u.Quantity(value, unit)


class ServoRating(FrozenModel):
    """What the servo does at one supply voltage (datasheet, no payload)."""

    voltage: Voltage
    speed: AngularSpeed = Field(
        description="rotation speed without payload (datasheet: s per 60 deg)"
    )
    stall_torque: Torque
    stall_current: Current
    quiescent_current: Current
    standby_current: Current


def _rating_at_6_volts() -> ServoRating:
    """Return the datasheet's 6 V column.

    Returns
    -------
    ServoRating
        The rating at 6 V.
    """
    return ServoRating(
        voltage=6.0 * u.V,
        speed=(60 * u.deg) / (0.18 * u.s),
        stall_torque=13.5 * KILOGRAM_FORCE * u.cm,
        stall_current=1.8 * u.A,
        quiescent_current=80 * u.mA,
        standby_current=4 * u.mA,
    )


def _rating_at_7_2_volts() -> ServoRating:
    """Return the datasheet's 7.2 V column.

    Returns
    -------
    ServoRating
        The rating at 7.2 V.
    """
    return ServoRating(
        voltage=7.2 * u.V,
        speed=(60 * u.deg) / (0.16 * u.s),
        stall_torque=15 * KILOGRAM_FORCE * u.cm,
        stall_current=2 * u.A,
        quiescent_current=100 * u.mA,
        standby_current=5 * u.mA,
    )


class DSSM15(FrozenModel):
    """DFRobot DSS-M15 metal-gear servo, 180 degrees (SKU SER0019).

    The defaults are the datasheet's values. The datasheet gives no mass; the
    measured masses of ARIEL's servos (with their longer cables) belong to
    the hinge module.
    """

    # The defaults go through the same unit check and SI conversion
    model_config = ConfigDict(
        frozen=True, extra="forbid", validate_default=True
    )

    min_voltage: Voltage = Field(default_factory=_quantity(4.8, u.V))
    max_voltage: Voltage = Field(default_factory=_quantity(7.2, u.V))
    at_6_volts: ServoRating = Field(default_factory=_rating_at_6_volts)
    at_7_2_volts: ServoRating = Field(default_factory=_rating_at_7_2_volts)
    angle_range: Angle = Field(
        default_factory=_quantity(180, u.deg),
        description="limited angle, 'about 180 deg' in the datasheet",
    )
    gear_ratio: float = 310.0
    size: tuple[Length, Length, Length] = Field(
        default_factory=lambda: (54.5 * u.mm, 20 * u.mm, 47.5 * u.mm),
        description="outer size in the datasheet's order (54.5 x 20 x 47.5)",
    )
    wire_length: Length = Field(default_factory=_quantity(320, u.mm))
    horn_spline_teeth: int = 25
    horn_spline_diameter: Length = Field(default_factory=_quantity(5.80, u.mm))
    min_pulse_width: Duration = Field(default_factory=_quantity(500, u.us))
    max_pulse_width: Duration = Field(default_factory=_quantity(2500, u.us))
    neutral_pulse_width: Duration = Field(
        default_factory=_quantity(1500, u.us),
        description="centre position; a shorter pulse turns clockwise",
    )
    min_pwm_frequency: Frequency = Field(default_factory=_quantity(50, u.Hz))
    max_pwm_frequency: Frequency = Field(default_factory=_quantity(330, u.Hz))
