"""The derivation table: every setting with its kind and its source."""

# Standard library
from typing import Literal

# Third-party libraries
import astropy.units as u

# Local libraries
from ariel.parameters import FrozenModel
from ariel.simulators.mujoco.seeds import Seeds

SEED_SOURCE = "seed, set by the user"
CONTACT_SOURCE = (
    "formula tc = kappa * L / v_max; judgement, kappa 0.28 gives the 14 ms "
    "calibrated for robot-to-robot contact"
)
ADVECTION_SOURCE = "formula alpha * L / v_max; judgement (research page)"
JOINT_SOURCE = "formula gamma / omega_max; judgement (research page)"
TIMESTEP_SOURCE = (
    "formula min(advection, tc / 2, joint), bound by the {limit} limit; "
    "tc >= 2 h is documented (refsafe)"
)


class Row(FrozenModel):
    """One line of the derivation table."""

    name: str
    value: float
    unit: str
    kind: Literal["seed", "formula", "fixed"]
    source: str


def derivation_table(seeds: Seeds) -> list[Row]:
    """List every seed and every derived value with its source.

    Parameters
    ----------
    seeds : Seeds
        The physical scales.

    Returns
    -------
    list[Row]
        Seeds first, then formulas, values in SI units.
    """
    rows: list[Row] = []
    for name in type(seeds).model_fields:
        value = getattr(seeds, name)
        if isinstance(value, u.Quantity):
            number, unit = float(value.value), str(value.unit)
        else:
            number, unit = float(value), ""
        rows.append(
            Row(
                name=name,
                value=number,
                unit=unit,
                kind="seed",
                source=SEED_SOURCE,
            ),
        )
    formulas = (
        ("contact_time_constant", seeds.contact_time_constant, CONTACT_SOURCE),
        ("advection_limit", seeds.advection_limit, ADVECTION_SOURCE),
        ("joint_limit", seeds.joint_limit, JOINT_SOURCE),
        (
            "timestep",
            seeds.timestep,
            TIMESTEP_SOURCE.format(limit=seeds.timestep_limit),
        ),
    )
    rows.extend(
        Row(name=name, value=value, unit="s", kind="formula", source=source)
        for name, value, source in formulas
    )
    return rows
