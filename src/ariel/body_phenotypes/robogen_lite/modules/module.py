"""Abstract base class for RoboGen-lite body modules.

Todo:
----
    [ ] Autoincrement of index in subclasses.
"""

# Standard library
import typing
from abc import ABC, abstractmethod

# Third-party libraries
import astropy.units as u
import mujoco

# Local libraries
from ariel.body_phenotypes.robogen_lite.config import ModuleFaces
from ariel.parameters import to_unit


# The one place where a module's quantities become MuJoCo's numbers
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


def half_extents(
    size: tuple[u.Quantity, u.Quantity, u.Quantity],
) -> tuple[float, float, float]:
    """Return MuJoCo's box size (half of each side) from the outer size.

    Parameters
    ----------
    size : tuple[astropy.units.Quantity, ...]
        The full length of each side.

    Returns
    -------
    tuple[float, float, float]
        Half of each side, in metres.
    """
    x, y, z = (metres(side) / 2 for side in size)
    return x, y, z


class Module(ABC):
    """Base class for all modules."""

    required_attributes: typing.ClassVar[list[str]] = ["index", "module_type"]

    # Set by every subclass in __init__
    index: int | None
    sites: dict[ModuleFaces, mujoco.MjsSite]
    spec: mujoco.MjSpec

    def __init_subclass__(cls) -> None:
        """
        Ensure that subclasses define required attributes.

        Raises
        ------
        NotImplementedError
            If a required attribute is not defined in the subclass.
        """
        super().__init_subclass__()
        for attr in cls.required_attributes:
            if not hasattr(cls, attr):
                msg = f"Class '{cls.__name__}' must define attribute '{attr}'"
                raise NotImplementedError(msg)

    @abstractmethod
    def rotate(self, angle: float) -> None:
        """
        Rotate the module by a certain angle.

        Parameters
        ----------
        angle : float
            The angle to rotate the module by, in degrees.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.
        """
        msg = f"{self.__class__.__name__} does not implement 'rotate' method."
        raise NotImplementedError(msg)
