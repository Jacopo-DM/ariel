"""Write the settings as MJCF and list what differs from MuJoCo's defaults.

Notes
-----
    * ``MjSpec.to_xml`` writes only the values that differ from the engine
      defaults, so the file is the list of differences and every other value
      stays the installed engine's default. It also never writes
      ``compiler.balanceinertia``, so ``generate_xml`` adds that attribute
      itself; ``build_spec`` keeps it as it is. The full table, defaults
      included, is in ``derivation_table`` and ``settings``.
"""

# Standard library
from itertools import starmap

# Third-party libraries
import mujoco

# Local libraries
from ariel.simulators.mujoco.seeds import Seeds
from ariel.simulators.mujoco.settings import (
    Setting,
    Value,
    engine_defaults,
    settings,
)

ABSOLUTE_TOLERANCE = 0.0
RELATIVE_TOLERANCE = 1e-6  # the engine stores some values as float32


def _assign(spec: mujoco.MjSpec, name: str, value: Value) -> None:
    """Set one setting on a spec.

    Parameters
    ----------
    spec : mujoco.MjSpec
        The spec to change.
    name : str
        Setting path, for example ``option.timestep`` or ``size.nconmax``.
    value : Value
        The new value; tuples become lists and integers become engine enums
        where the attribute is one.
    """
    *path, leaf = name.split(".")
    target = spec
    for part in path:
        if part != "size":
            target = getattr(target, part)
    current = getattr(target, leaf)
    if isinstance(value, tuple):
        setattr(target, leaf, list(value))
    elif hasattr(current, "value") and hasattr(current, "name"):
        setattr(target, leaf, type(current)(value))
    else:
        setattr(target, leaf, value)


def build_spec(seeds: Seeds) -> mujoco.MjSpec:
    """Build a spec that holds every setting.

    Parameters
    ----------
    seeds : Seeds
        The physical scales.

    Returns
    -------
    mujoco.MjSpec
        A new spec; worlds add their bodies to it.
    """
    spec = mujoco.MjSpec()
    for row in settings(seeds):
        _assign(spec, row.name, row.value)
    return spec


def generate_xml(seeds: Seeds) -> str:
    """Write the settings as MJCF.

    Parameters
    ----------
    seeds : Seeds
        The physical scales.

    Returns
    -------
    str
        The MJCF text.
    """
    spec = build_spec(seeds)
    flag = "true" if spec.compiler.balanceinertia else "false"
    return str(spec.to_xml()).replace(
        "<compiler", f'<compiler balanceinertia="{flag}"', 1
    )


def _equal(left: Value, right: Value) -> bool:
    """Compare two settings, numbers up to rounding.

    Parameters
    ----------
    left : Value
        One value.
    right : Value
        The other.

    Returns
    -------
    bool
        True when they are the same setting value.
    """
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(
            starmap(_close, zip(left, right, strict=True))
        )
    if isinstance(left, float) and isinstance(right, float):
        return _close(left, right)
    return left == right


def _close(left: float, right: float) -> bool:
    """Tell whether two floats agree to a relative 1e-6.

    Parameters
    ----------
    left : float
        One number.
    right : float
        The other.

    Returns
    -------
    bool
        True when they agree.
    """
    scale = max(abs(left), abs(right))
    return abs(left - right) <= RELATIVE_TOLERANCE * scale + ABSOLUTE_TOLERANCE


def differences(seeds: Seeds) -> list[Setting]:
    """List the settings that differ from the installed engine's defaults.

    Parameters
    ----------
    seeds : Seeds
        The physical scales.

    Returns
    -------
    list[Setting]
        The differing settings, in section order.
    """
    defaults = engine_defaults()
    return [
        row
        for row in settings(seeds)
        if not _equal(row.value, defaults[row.name])
    ]


def summary(seeds: Seeds) -> str:
    """Count the differences.

    Parameters
    ----------
    seeds : Seeds
        The physical scales.

    Returns
    -------
    str
        For example ``12 of 157 differ``.
    """
    return f"{len(differences(seeds))} of {len(settings(seeds))} differ"
