"""Plausible bands: a value outside its band raises a ``UserWarning``.

A band never blocks anything. The limits are the user's approved judgement
of 2026-10-19 (wiki page ``mujoco_parameter_research``), except the upper
limit of ``h / tc``, which MuJoCo documents (``refsafe``).
"""

# Standard library
import warnings

# Third-party libraries
import numpy as np

# Local libraries
from ariel import console
from ariel.parameters import FrozenModel
from ariel.simulators.mujoco.seeds import Seeds
from ariel.simulators.mujoco.settings import Setting, settings

SOLREF_DAMPING_INDEX = 1
SOLIMP_D0_INDEX = 0
EDGE_TOLERANCE = 1e-9


class Band(FrozenModel):
    """The plausible range of one quantity."""

    name: str
    low: float
    high: float
    unit: str
    basis: str


class BandResult(FrozenModel):
    """A quantity checked against its band."""

    name: str
    value: float
    low: float
    high: float
    unit: str
    basis: str
    inside: bool


BANDS = (
    Band(
        name="timestep",
        low=0.0005,
        high=0.005,
        unit="s",
        basis="judgement (reports agree, DeepMind Control uses 2 to 5 ms)",
    ),
    Band(
        name="timestep_over_contact",
        low=0.01,
        high=0.5,
        unit="",
        basis="upper limit documented (refsafe), lower limit judgement",
    ),
    Band(
        name="courant",
        low=0.0,
        high=0.05,
        unit="",
        basis="judgement (ChatGPT report)",
    ),
    Band(
        name="gravity",
        low=9.0,
        high=10.0,
        unit="m / s2",
        basis="judgement, Earth",
    ),
    Band(
        name="air_density",
        low=0.8,
        high=1.4,
        unit="kg / m3",
        basis="judgement around the reference values",
    ),
    Band(
        name="air_viscosity",
        low=1.5e-5,
        high=2.2e-5,
        unit="Pa s",
        basis="judgement around the reference values",
    ),
    Band(
        name="friction",
        low=0.2,
        high=1.5,
        unit="",
        basis="judgement; no primary tribology source for ABS or PLA",
    ),
    Band(
        name="solref_damping_ratio",
        low=0.7,
        high=1.5,
        unit="",
        basis="judgement",
    ),
    Band(
        name="solimp_d0",
        low=0.8,
        high=0.98,
        unit="",
        basis="judgement (documented hard range 0.0001 to 0.9999)",
    ),
)


def _tuple_value(rows: dict[str, Setting], name: str, index: int) -> float:
    """Read one element of an array setting.

    Parameters
    ----------
    rows : dict[str, Setting]
        Settings by name.
    name : str
        The setting.
    index : int
        The element.

    Returns
    -------
    float
        The element.
    """
    value = rows[name].value
    return float(value[index]) if isinstance(value, tuple) else float("nan")


def _inside(value: float, band: Band) -> bool:
    """Tell whether a value is within a band, edges included.

    The edges carry a relative tolerance of 1e-9: with the default seeds the
    timestep is advection-limited, so ``courant`` equals ``alpha`` (0.05),
    the upper limit, up to rounding.

    Parameters
    ----------
    value : float
        The quantity.
    band : Band
        Its band.

    Returns
    -------
    bool
        True when the value is finite and within the band.
    """
    return bool(
        np.isfinite(value)
        and (
            band.low <= value <= band.high
            or np.isclose(value, band.low, rtol=EDGE_TOLERANCE, atol=0.0)
            or np.isclose(value, band.high, rtol=EDGE_TOLERANCE, atol=0.0)
        )
    )


def evaluate_bands(seeds: Seeds) -> list[BandResult]:
    """Check every quantity against its plausible band.

    Parameters
    ----------
    seeds : Seeds
        The physical scales.

    Returns
    -------
    list[BandResult]
        One result per band, in the order of ``BANDS``.
    """
    rows = {row.name: row for row in settings(seeds)}
    values = {
        "timestep": seeds.timestep,
        "timestep_over_contact": seeds.timestep_over_contact,
        "courant": seeds.courant,
        "gravity": float(seeds.gravity.value),
        "air_density": float(seeds.air_density.value),
        "air_viscosity": float(seeds.air_viscosity.value),
        "friction": seeds.friction,
        "solref_damping_ratio": _tuple_value(
            rows, "default.geom.solref", SOLREF_DAMPING_INDEX
        ),
        "solimp_d0": _tuple_value(rows, "default.geom.solimp", SOLIMP_D0_INDEX),
    }
    return [
        BandResult(
            name=band.name,
            value=values[band.name],
            low=band.low,
            high=band.high,
            unit=band.unit,
            basis=band.basis,
            inside=_inside(values[band.name], band),
        )
        for band in BANDS
    ]


def warn_outside(results: list[BandResult]) -> int:
    """Warn once for every value outside its band.

    Parameters
    ----------
    results : list[BandResult]
        Output of ``evaluate_bands``.

    Returns
    -------
    int
        The number of values outside their band.
    """
    outside = [result for result in results if not result.inside]
    for result in outside:
        message = (
            f"{result.name} = {result.value:g} {result.unit} is outside its "
            f"plausible band {result.low:g} to {result.high:g} "
            f"({result.basis})"
        )
        console.print(f"[yellow]{message}[/yellow]")
        warnings.warn(message, UserWarning, stacklevel=2)
    return len(outside)


def summary(results: list[BandResult]) -> str:
    """Count the values outside their band.

    Parameters
    ----------
    results : list[BandResult]
        Output of ``evaluate_bands``.

    Returns
    -------
    str
        For example ``1 of 9 values outside their plausible band``.
    """
    outside = sum(not result.inside for result in results)
    return f"{outside} of {len(results)} values outside their plausible band"
