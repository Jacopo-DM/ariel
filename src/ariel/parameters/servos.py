"""The servo that drives every hinge: DFRobot DSS-M15 (SKU SER0019).

Every value is an astropy quantity written in the datasheet's own unit and
stored in SI, so a torque in kilogram-force centimetres can never be read as
newton metres.

References
----------
    [1] DFRobot, "DF Metal Geared 15Kg Standard Servo 180 (DSS-M15)",
        SKU SER0019, https://www.dfrobot.com/product-120.html (the datasheet
        PDF in the user's uploads, 2026-10-21).
    [2] maxon, "A-max 16, 16 mm, graphite brushes, 2 W", part 110079,
        https://www.maxongroup.com/maxon/view/product/110079 (read
        2026-10-22): rotor inertia 0.816 g cm2, stall torque 4.49 mN m.
"""

# Standard library
import math
from collections.abc import Callable
from typing import Annotated

# Third-party libraries
import astropy.units as u
from pydantic import ConfigDict, Field

# Local libraries
from ariel.parameters.base_model import FrozenModel
from ariel.parameters.quantities import KILOGRAM_FORCE, quantity_of, to_unit

type Angle = Annotated[u.Quantity, quantity_of(u.rad)]
type AngularSpeed = Annotated[u.Quantity, quantity_of(u.rad / u.s)]
type Current = Annotated[u.Quantity, quantity_of(u.A)]
type Duration = Annotated[u.Quantity, quantity_of(u.s)]
type Frequency = Annotated[u.Quantity, quantity_of(u.Hz)]
type Length = Annotated[u.Quantity, quantity_of(u.m)]
type MomentOfInertia = Annotated[u.Quantity, quantity_of(u.kg * u.m**2)]
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


# Units of the derived values: per radian, so a rotation is never a number
_DAMPING = u.N * u.m * u.s / u.rad
_STIFFNESS = u.N * u.m / u.rad


class ServoSettings(FrozenModel):
    """How the simulated hinge uses its DSS-M15 (goal servo-dynamics).

    The datasheet says what the servo can do; these settings choose the
    supply voltage, how firmly it holds its angle, and the gearbox inertia
    the datasheet leaves out. The methods derive MuJoCo's joint and
    actuator values from both.
    """

    # The defaults go through the same unit check and SI conversion
    model_config = ConfigDict(
        frozen=True, extra="forbid", validate_default=True
    )

    voltage: Voltage = Field(
        default_factory=_quantity(6.0, u.V),
        description="supply voltage; the datasheet rates 6 V and 7.2 V",
    )
    holding_error: Angle = Field(
        default_factory=_quantity(5, u.deg),
        description="angle error at which the servo pushes its stall torque",
    )
    rotor_inertia: MomentOfInertia = Field(
        default_factory=_quantity(0.816, u.g * u.cm**2),
        description="motor rotor inertia, an estimate (see the source)",
    )
    rotor_inertia_source: str = (
        "estimate: maxon A-max 16, part 110079 (0.816 g cm2; stall torque "
        "4.49 mN m, 1.39 N m after the 310:1 gear, close to the DSS-M15's "
        "1.32 N m); the datasheet gives none"
    )

    def rating(self, servo: DSSM15) -> ServoRating:
        """Return the datasheet column for this supply voltage.

        Parameters
        ----------
        servo : DSSM15
            The servo's datasheet record.

        Returns
        -------
        ServoRating
            The rating at ``voltage``.

        Raises
        ------
        ValueError
            If the datasheet has no rating at ``voltage``.
        """
        for rating in (servo.at_6_volts, servo.at_7_2_volts):
            # Both are stored in volts (SI)
            if math.isclose(rating.voltage.value, self.voltage.value):
                return rating
        msg = f"the DSS-M15 has no rating at {self.voltage:g}: 6 V or 7.2 V"
        raise ValueError(msg)

    def stall_torque(self, servo: DSSM15) -> u.Quantity:
        """Return the largest torque the servo exerts (its torque cap).

        Parameters
        ----------
        servo : DSSM15
            The servo's datasheet record.

        Returns
        -------
        astropy.units.Quantity
            The stall torque at ``voltage``, in N m.
        """
        return self.rating(servo).stall_torque

    def no_load_speed(self, servo: DSSM15) -> u.Quantity:
        """Return the top speed under the servo's own power.

        Parameters
        ----------
        servo : DSSM15
            The servo's datasheet record.

        Returns
        -------
        astropy.units.Quantity
            The no-load speed at ``voltage``, in rad / s.
        """
        return self.rating(servo).speed

    def damping(self, servo: DSSM15) -> u.Quantity:
        """Return the joint damping that gives the motor's torque-speed line.

        A DC motor's torque falls in a straight line from the stall torque
        at rest to zero at the no-load speed; a damping of their ratio
        gives that line.

        Parameters
        ----------
        servo : DSSM15
            The servo's datasheet record.

        Returns
        -------
        astropy.units.Quantity
            Stall torque over no-load speed, in N m s / rad.
        """
        return to_unit(
            self.stall_torque(servo) / self.no_load_speed(servo), _DAMPING
        )

    def stiffness(self, servo: DSSM15) -> u.Quantity:
        """Return the position gain: full torque at the holding error.

        Parameters
        ----------
        servo : DSSM15
            The servo's datasheet record.

        Returns
        -------
        astropy.units.Quantity
            Stall torque over holding error, in N m / rad.
        """
        return to_unit(
            self.stall_torque(servo) / self.holding_error, _STIFFNESS
        )

    def armature(self, servo: DSSM15) -> u.Quantity:
        """Return the motor's inertia as the joint feels it through the gear.

        Parameters
        ----------
        servo : DSSM15
            The servo's datasheet record.

        Returns
        -------
        astropy.units.Quantity
            Rotor inertia times the gear ratio squared, in kg m2.
        """
        return to_unit(self.rotor_inertia * servo.gear_ratio**2, u.kg * u.m**2)
