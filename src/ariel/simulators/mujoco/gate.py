"""The one place where ARIEL compiles a MuJoCo model, and the checks it runs.

``compile_model`` compiles a spec, checks the compiled model for physically
implausible sizes and masses, warns once for each finding (a ``UserWarning``,
never an error) and returns the model. Every compile in ARIEL goes through it;
an architecture test pins that (goal physical-checks).
"""

# Standard library
import math
import warnings
from typing import cast

# Third-party libraries
import mujoco
import numpy as np

# Local libraries
from ariel import console
from ariel.parameters import FloatArray, FrozenModel
from ariel.simulators.mujoco.bands import BANDS, Band

# Geom types whose volume follows from their size
_SPHERE = int(mujoco.mjtGeom.mjGEOM_SPHERE)
_CAPSULE = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
_ELLIPSOID = int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)
_CYLINDER = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
_BOX = int(mujoco.mjtGeom.mjGEOM_BOX)
_SIZES_USED = {_SPHERE: 1, _CAPSULE: 2, _ELLIPSOID: 3, _CYLINDER: 2, _BOX: 3}
_FREE = int(mujoco.mjtJoint.mjJNT_FREE)

HALF_SIZE = Band(
    name="geom_half_size",
    low=1e-3,
    high=10.0,
    unit="m",
    basis="judgement: a millimetre screw to a 20 m arena",
)
BODY_MASS = Band(
    name="body_mass",
    low=1e-3,
    high=20.0,
    unit="kg",
    basis="judgement: a gram to a 20 kg robot part",
)
DENSITY = Band(
    name="density",
    low=50.0,
    high=8000.0,
    unit="kg / m3",
    basis="judgement: foam to steel",
)
MASS_RATIO = Band(
    name="joint_mass_ratio",
    low=1.0,
    high=100.0,
    unit="",
    basis="MuJoCo documentation: large mass ratios hurt the solver",
)
TIMESTEP_OVER_CONTACT = next(
    band for band in BANDS if band.name == "timestep_over_contact"
)


class CheckResult(FrozenModel):
    """A value of a compiled model outside its plausible band."""

    check: str
    subject: str
    value: float
    low: float
    high: float
    unit: str
    basis: str


def _outside(band: Band, subject: str, value: float) -> list[CheckResult]:
    """Return the finding for ``value``, or nothing when it is inside.

    Parameters
    ----------
    band : Band
        The plausible range.
    subject : str
        What the value belongs to, for example ``geom 'brick'``.
    value : float
        The value, in the band's unit.

    Returns
    -------
    list[CheckResult]
        One result if the value is outside the band, otherwise none.
    """
    if band.low <= value <= band.high:
        return []
    return [
        CheckResult(
            check=band.name,
            subject=subject,
            value=value,
            low=band.low,
            high=band.high,
            unit=band.unit,
            basis=band.basis,
        )
    ]


def _name(model: mujoco.MjModel, kind: mujoco.mjtObj, index: int) -> str:
    """Return an element's name, or its number when it has none.

    Parameters
    ----------
    model : mujoco.MjModel
        The compiled model.
    kind : mujoco.mjtObj
        The element type.
    index : int
        The element's id.

    Returns
    -------
    str
        The quoted name, or ``#index``.
    """
    name = mujoco.mj_id2name(model, kind, index)
    return f"'{name}'" if name else f"#{index}"


def _volume(kind: int, size: FloatArray) -> float:
    """Return the volume of a primitive geom from its MuJoCo size.

    Parameters
    ----------
    kind : int
        The geom type (sphere, capsule, ellipsoid, cylinder or box).
    size : FloatArray
        MuJoCo's size: radii and half-lengths.

    Returns
    -------
    float
        The volume in cubic metres.
    """
    a, b, c = (float(value) for value in size)
    if kind == _SPHERE:
        return 4 / 3 * math.pi * a**3
    if kind == _ELLIPSOID:
        return 4 / 3 * math.pi * a * b * c
    if kind == _CYLINDER:
        return math.pi * a**2 * 2 * b
    if kind == _CAPSULE:
        return math.pi * a**2 * 2 * b + 4 / 3 * math.pi * a**3
    return 8 * a * b * c


def _half_sizes(model: mujoco.MjModel) -> list[CheckResult]:
    """Check every primitive geom's radii and half-lengths (C1).

    Parameters
    ----------
    model : mujoco.MjModel
        The compiled model.

    Returns
    -------
    list[CheckResult]
        The sizes outside the band.
    """
    found: list[CheckResult] = []
    for geom, kind in enumerate(model.geom_type):
        used = _SIZES_USED.get(int(kind), 0)
        subject = f"geom {_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)}"
        for value in model.geom_size[geom][:used]:
            found += _outside(HALF_SIZE, subject, float(value))
    return found


def _moving_bodies(model: mujoco.MjModel) -> list[int]:
    """Return the bodies that move: not welded to the world, with mass.

    Parameters
    ----------
    model : mujoco.MjModel
        The compiled model.

    Returns
    -------
    list[int]
        Body ids; frames without mass are left out.
    """
    return [
        body
        for body in range(1, model.nbody)
        if model.body_weldid[body] != 0 and model.body_mass[body] > 0
    ]


def _masses_and_densities(model: mujoco.MjModel) -> list[CheckResult]:
    """Check each moving body's mass (C2) and density (C3).

    The density is the body's mass over the volume of its geoms; a body
    with a geom of another type (a mesh) has no density check.

    Parameters
    ----------
    model : mujoco.MjModel
        The compiled model.

    Returns
    -------
    list[CheckResult]
        The masses and densities outside their bands.
    """
    found: list[CheckResult] = []
    for body in _moving_bodies(model):
        mass = float(model.body_mass[body])
        subject = f"body {_name(model, mujoco.mjtObj.mjOBJ_BODY, body)}"
        found += _outside(BODY_MASS, subject, mass)
        geoms = range(
            model.body_geomadr[body],
            model.body_geomadr[body] + model.body_geomnum[body],
        )
        kinds = [int(model.geom_type[geom]) for geom in geoms]
        if kinds and all(kind in _SIZES_USED for kind in kinds):
            volume = sum(
                _volume(kind, model.geom_size[geom])
                for geom, kind in zip(geoms, kinds, strict=True)
            )
            found += _outside(DENSITY, subject, mass / volume)
    return found


def _mass_ratios(model: mujoco.MjModel) -> list[CheckResult]:
    """Check the masses of the two rigid parts each joint connects (C4).

    A rigid part is every body welded together; its mass is their sum.

    Parameters
    ----------
    model : mujoco.MjModel
        The compiled model.

    Returns
    -------
    list[CheckResult]
        The joints whose ratio is outside the band.
    """
    weld = model.body_weldid
    part_mass = np.bincount(
        weld, weights=model.body_mass, minlength=model.nbody
    )
    found: list[CheckResult] = []
    for joint in range(model.njnt):
        if int(model.jnt_type[joint]) == _FREE:
            continue
        child = int(model.jnt_bodyid[joint])
        parent = int(weld[model.body_parentid[child]])
        if parent == 0:  # a joint to the world itself
            continue
        # MuJoCo refuses a moving part without mass, so neither is zero
        masses = (float(part_mass[weld[child]]), float(part_mass[parent]))
        subject = f"joint {_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)}"
        found += _outside(MASS_RATIO, subject, max(masses) / min(masses))
    return found


def _timestep(model: mujoco.MjModel) -> list[CheckResult]:
    """Check the timestep against the fastest contact time constant (C5).

    Parameters
    ----------
    model : mujoco.MjModel
        The compiled model.

    Returns
    -------
    list[CheckResult]
        The ratio when it is outside its band.
    """
    # A positive first solref value is a time constant (negative: stiffness)
    constants = model.geom_solref[:, 0]
    constants = constants[constants > 0]
    if constants.size == 0:
        return []
    ratio = float(model.opt.timestep) / float(constants.min())
    return _outside(TIMESTEP_OVER_CONTACT, "option timestep", ratio)


def check_model(model: mujoco.MjModel) -> list[CheckResult]:
    """Check a compiled model and return what is implausible.

    Parameters
    ----------
    model : mujoco.MjModel
        The compiled model.

    Returns
    -------
    list[CheckResult]
        Every value outside its band (empty for a plausible model).
    """
    return (
        _half_sizes(model)
        + _masses_and_densities(model)
        + _mass_ratios(model)
        + _timestep(model)
    )


def compile_model(spec: mujoco.MjSpec) -> mujoco.MjModel:
    """Compile a spec, warn about implausible values, and return the model.

    Parameters
    ----------
    spec : mujoco.MjSpec
        The spec to compile (``mujoco.MjSpec.from_string(xml)`` for XML).

    Returns
    -------
    mujoco.MjModel
        The compiled model; the checks never block it.
    """
    model = cast("mujoco.MjModel", spec.compile())
    for result in check_model(model):
        message = (
            f"{result.subject}: {result.check} = {result.value:g} "
            f"{result.unit} is outside {result.low:g} to {result.high:g} "
            f"({result.basis})"
        )
        console.print(f"[yellow]{message}[/yellow]")
        warnings.warn(message, UserWarning, stacklevel=2)
    return model
