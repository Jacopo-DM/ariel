"""Measured sizes and masses of the RoboGen-lite modules.

Every value is an astropy quantity written in the unit it was measured in,
with its source, and stored in SI. The modules read these records and turn
them into MuJoCo's numbers in one place (``ariel.body_phenotypes``).

References
----------
    [1] The user's workbook ``Ariel_modules_measurements.xlsx`` (masses in
        grams; sheet "Weights"; 2026-10-21).
    [2] The printed parts, ``github.com/ci-group/ariel-models``,
        ``v2/stl`` (millimetres; read 2026-10-21).
    [3] Photos of assembled robots (the user, 2026-10-21): the hinge's fork
        sits on the parent, the cradle with the servo on the child.
"""

# Standard library
from collections.abc import Callable
from typing import Annotated, cast

# Third-party libraries
import astropy.units as u
from pydantic import ConfigDict, Field

# Local libraries
from ariel.parameters.base_model import FrozenModel
from ariel.parameters.quantities import quantity_of, to_unit

type Length = Annotated[u.Quantity, quantity_of(u.m)]
type Mass = Annotated[u.Quantity, quantity_of(u.kg)]
type MassPerLength = Annotated[u.Quantity, quantity_of(u.kg / u.m)]
type Size = tuple[Length, Length, Length]

# The defaults go through the same unit check and SI conversion
_CHECK_DEFAULTS = ConfigDict(frozen=True, extra="forbid", validate_default=True)


def _quantity(value: float, unit: u.UnitBase) -> Callable[[], u.Quantity]:
    """Return a factory for a default quantity (quantities are mutable).

    Parameters
    ----------
    value : float
        The number, in ``unit``.
    unit : astropy.units.UnitBase
        The unit it was measured in.

    Returns
    -------
    Callable[[], astropy.units.Quantity]
        A function that builds a fresh quantity on every call.
    """
    return lambda: u.Quantity(value, unit)


def _cube(edge: float, unit: u.UnitBase) -> Callable[[], Size]:
    """Return a factory for the size of a cube.

    Parameters
    ----------
    edge : float
        The edge, in ``unit``.
    unit : astropy.units.UnitBase
        The unit it was measured in.

    Returns
    -------
    Callable[[], Size]
        A function that builds the three equal sides.
    """
    return lambda: (edge * unit, edge * unit, edge * unit)


class BrickSpec(FrozenModel):
    """A brick: a printed cube (red, the current standard)."""

    model_config = _CHECK_DEFAULTS

    mass: Mass = Field(
        default_factory=_quantity(59.528, u.g),
        description="mean of 100 red bricks [1]",
    )
    size: Size = Field(
        default_factory=_cube(75, u.mm),
        description="outer edges, STL 0002B_brick [2]",
    )


class CoreSpec(FrozenModel):
    """The core: the box with the controller, battery and camera."""

    model_config = _CHECK_DEFAULTS

    mass: Mass = Field(
        default_factory=_quantity(1386.121, u.g),
        description="core 966.85 g + battery 396.497 g + camera 22.774 g [1]",
    )
    size: Size = Field(
        default_factory=_cube(150, u.mm),
        description="outer edges, STL 0001A to 0001D [2]",
    )


class HingeSpec(FrozenModel):
    """A hinge: a fork on the parent, a cradle holding the servo on the child.

    The servo turns the cradle (and the child module) relative to the fork,
    about an axis across the hinge. Its cable runs back to the core, so its
    mass grows with the hinge's distance from the core.
    """

    model_config = _CHECK_DEFAULTS

    length: Length = Field(
        default_factory=_quantity(75.0, u.mm),
        description="parent face to child face, STL 0003A and 0003B [2]",
    )
    width: Length = Field(
        default_factory=_quantity(52, u.mm),
        description="square plates, STL [2]",
    )
    axis_from_parent: Length = Field(
        default_factory=_quantity(31.74, u.mm),
        description="rotation axis from the parent face (the fork's hole) [2]",
    )
    fork_mass: Mass = Field(
        default_factory=_quantity(11.484, u.g),
        description="'Plastic (round hole)', STL 0003A [1]",
    )
    cradle_mass: Mass = Field(
        default_factory=_quantity(16.488, u.g),
        description="'Plastic (round knob)', STL 0003B [1]",
    )
    plate_screws_mass: Mass = Field(
        default_factory=_quantity(8.33 / 2, u.g),
        description="4 of the 8 big screws, one plate's worth [1]",
    )
    horn_screws_mass: Mass = Field(
        default_factory=_quantity(4 * 0.14, u.g),
        description="4 small screws, servo horn to fork [1]",
    )
    connector_mass: Mass = Field(
        default_factory=_quantity(2.21, u.g),
        description="joins the two halves at the pivot [1]",
    )
    bare_servo_mass: Mass = Field(
        default_factory=_quantity(57.236, u.g),
        description="servo without cable: the line through 3 cable lengths",
    )
    cable_mass_per_length: MassPerLength = Field(
        default_factory=_quantity(0.11353, u.g / u.cm),
        description="slope of that line (servos with 55, 70, 95 cm) [1]",
    )
    route_per_step: Length = Field(
        default_factory=_quantity(75, u.mm),
        description="cable route per step up the body's tree to the core",
    )

    @property
    def parent_mass(self) -> u.Quantity:
        """Mass on the parent side: fork, its plate screws, horn screws.

        Returns
        -------
        astropy.units.Quantity
            The mass in kg.
        """
        # Quantity arithmetic is typed as a plain array in the stubs
        total = cast(
            "u.Quantity",
            self.fork_mass + self.plate_screws_mass + self.horn_screws_mass,
        )
        return to_unit(total, u.kg)

    def child_mass(self, steps_from_core: int) -> u.Quantity:
        """Mass on the child side, with the servo and its cable to the core.

        Parameters
        ----------
        steps_from_core : int
            Edges between the core and this hinge in the body's tree (1 for
            a hinge on the core).

        Returns
        -------
        astropy.units.Quantity
            The mass in kg.

        Raises
        ------
        ValueError
            If ``steps_from_core`` is below 1.
        """
        if steps_from_core < 1:
            msg = (
                "a hinge is at least 1 step from the core, "
                f"got {steps_from_core}"
            )
            raise ValueError(msg)
        route = self.route_per_step * steps_from_core
        total = cast(
            "u.Quantity",
            self.cradle_mass
            + self.plate_screws_mass
            + self.connector_mass
            + self.bare_servo_mass
            + self.cable_mass_per_length * route,
        )
        return to_unit(total, u.kg)
