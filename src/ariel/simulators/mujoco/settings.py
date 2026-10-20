"""Every MuJoCo setting the generator holds, with its kind, reason and source.

The five sections are ``option``, ``compiler``, ``size``, ``visual`` and the
geom contact fields of the main ``default`` class. A setting is a seed-derived
value (``seed`` or ``formula``) or a ``fixed`` one: ARIEL's own choice, or the
default of the installed engine. Names are paths into ``mujoco.MjSpec``, for
example ``option.timestep`` or ``default.geom.solref``.

Notes
-----
    * Engine defaults are read from the installed engine, never typed in, so
        a MuJoCo upgrade shows up as a changed default.
    * Left out: ``compiler.eulerseq`` (a character vector, not a number or
        text), ``compiler.authored`` (a bit mask the XML parser sets to record
        which attributes the file wrote) and ``stat`` (computed by the
        compiler, not set).
"""

# Standard library
from typing import Literal

# Third-party libraries
import mujoco
import numpy as np

# Local libraries
from ariel.parameters import FrozenModel
from ariel.simulators.mujoco.seeds import Seeds

type Value = bool | int | float | str | tuple[float, ...]

SECTIONS = ("option", "compiler", "size", "visual", "default")
SIZE_FIELDS = (
    "memory",
    "nconmax",
    "njmax",
    "nstack",
    "nuserdata",
    "nemax",
    "nuser_body",
    "nuser_jnt",
    "nuser_geom",
    "nuser_site",
    "nuser_cam",
    "nuser_tendon",
    "nuser_actuator",
    "nuser_sensor",
)
GEOM_CONTACT_FIELDS = (
    "friction",
    "solref",
    "solimp",
    "condim",
    "margin",
    "gap",
    "priority",
)
ALIASES = frozenset({"global"})  # visual.global is the same object as global_
SKIPPED = frozenset({"eulerseq", "authored"})
DAMPING_RATIO = 1.0  # critical damping, the MuJoCo default for solref

DOCS = "https://mujoco.readthedocs.io/en/stable/XMLreference.html"
KEPT_REASON = {
    "option": "numerical setting, MuJoCo default kept",
    "compiler": "compiler choice, MuJoCo default kept",
    "size": "memory sizing, MuJoCo default kept",
    "visual": "rendering only, MuJoCo default kept",
    "default": "contact default, MuJoCo default kept",
}
KEPT_SOURCE = {
    "option": f"{DOCS}#option; installed engine default; wiki mjcf_option",
    "compiler": (
        f"{DOCS}#compiler; installed engine default; wiki mjcf_compiler"
    ),
    "size": f"{DOCS}#size; installed engine default; wiki mjcf_size_statistic",
    "visual": f"{DOCS}#visual; installed engine default; wiki mjcf_visual",
    "default": (
        f"{DOCS}#default; installed engine default; "
        "wiki mjcf_geom_joint_defaults"
    ),
}
JUDGEMENT = "judgement (wiki mujoco_parameter_research)"


class Setting(FrozenModel):
    """One MuJoCo setting with where its value comes from."""

    name: str
    value: Value
    kind: Literal["seed", "formula", "fixed"]
    reason: str
    source: str


class Choice(FrozenModel):
    """A value ARIEL fixes on purpose, with the reason."""

    value: Value
    reason: str
    source: str


OLD_CONFIG = "what ARIEL already used (old MujocoConfig)"
OLD_WORLD = "what ARIEL already used (old BaseWorld._init_spec)"
LOOK = "ARIEL's own look of the viewer"
CHOICES = {
    "option.integrator": Choice(
        value=int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST),
        reason=(
            "implicitfast is the documented general recommendation and "
            f"{OLD_CONFIG}"
        ),
        source=f"{DOCS}#option-integrator; wiki mujoco_modeling_guidance",
    ),
    "compiler.balanceinertia": Choice(
        value=True,
        reason=(
            "accept slightly inconsistent inertias of generated bodies; "
            f"{OLD_CONFIG}"
        ),
        source=f"{DOCS}#compiler-balanceinertia",
    ),
    "compiler.degree": Choice(
        value=False,
        reason=f"ARIEL code uses radians; {OLD_CONFIG}",
        source=f"{DOCS}#compiler-angle",
    ),
    "visual.global_.offheight": Choice(
        value=960,
        reason=f"offscreen render height; {OLD_CONFIG}",
        source=f"{DOCS}#visual-global",
    ),
    "visual.global_.offwidth": Choice(
        value=1280,
        reason=f"offscreen render width; {OLD_CONFIG}",
        source=f"{DOCS}#visual-global",
    ),
    "visual.global_.azimuth": Choice(
        value=140.0,
        reason=f"{LOOK}; {OLD_WORLD}",
        source=f"{DOCS}#visual-global",
    ),
    "visual.global_.elevation": Choice(
        value=-20.0,
        reason=f"{LOOK}; {OLD_WORLD}",
        source=f"{DOCS}#visual-global",
    ),
    "visual.headlight.diffuse": Choice(
        value=(0.6, 0.6, 0.6),
        reason=f"{LOOK}; {OLD_WORLD}",
        source=f"{DOCS}#visual-headlight",
    ),
    "visual.headlight.ambient": Choice(
        value=(0.1, 0.1, 0.1),
        reason=f"{LOOK}; {OLD_WORLD}",
        source=f"{DOCS}#visual-headlight",
    ),
    "visual.headlight.specular": Choice(
        value=(0.9, 0.9, 0.9),
        reason=f"{LOOK}; {OLD_WORLD}",
        source=f"{DOCS}#visual-headlight",
    ),
    "visual.rgba.force": Choice(
        value=(1.0, 0.0, 0.0, 1.0),
        reason=f"{LOOK}; {OLD_WORLD}",
        source=f"{DOCS}#visual-rgba",
    ),
    "visual.rgba.haze": Choice(
        value=(0.15, 0.25, 0.35, 1.0),
        reason=f"{LOOK}; {OLD_WORLD}",
        source=f"{DOCS}#visual-rgba",
    ),
    "visual.map.force": Choice(
        value=0.01,
        reason=f"{LOOK}; {OLD_WORLD}",
        source=f"{DOCS}#visual-map",
    ),
    "visual.scale.forcewidth": Choice(
        value=0.3,
        reason=f"{LOOK}; {OLD_WORLD}",
        source=f"{DOCS}#visual-scale",
    ),
    "visual.scale.contactwidth": Choice(
        value=0.5,
        reason=f"{LOOK}; {OLD_WORLD}",
        source=f"{DOCS}#visual-scale",
    ),
    "visual.scale.contactheight": Choice(
        value=0.2,
        reason=f"{LOOK}; {OLD_WORLD}",
        source=f"{DOCS}#visual-scale",
    ),
    "visual.quality.shadowsize": Choice(
        value=8192,
        reason=f"{LOOK}; {OLD_WORLD}",
        source=f"{DOCS}#visual-quality",
    ),
}


def _plain(value: object) -> Value | None:
    """Turn an engine attribute into a plain value.

    Parameters
    ----------
    value : object
        An attribute of an engine object.

    Returns
    -------
    Value | None
        The plain value, or None when the attribute is not a setting.
    """
    result: Value | None = None
    if isinstance(value, bool | int | float | str):
        result = value
    elif isinstance(value, np.ndarray):
        result = tuple(float(item) for item in value.ravel())
    elif hasattr(value, "value") and hasattr(value, "name"):
        result = int(getattr(value, "value"))  # ruff: ignore[get-attr-with-constant]  # mujoco enums are untyped
    return result


def read_object(
    obj: object,
    prefix: str,
    names: tuple[str, ...] | None = None,
) -> dict[str, Value]:
    """Read the settings of an engine object and its sub-objects.

    Private names, aliases, skipped names, methods and anything that is
    neither a plain value nor an engine object are left out.

    Parameters
    ----------
    obj : object
        An engine object, for example ``spec.option``.
    prefix : str
        The path of ``obj``.
    names : tuple[str, ...] | None
        The attributes to read; all public ones when None.

    Returns
    -------
    dict[str, Value]
        Path to plain value.
    """
    found: dict[str, Value] = {}
    for name in dir(obj) if names is None else names:
        if name.startswith("_") or name in ALIASES or name in SKIPPED:
            continue
        attribute = getattr(obj, name)
        path = f"{prefix}.{name}"
        plain = _plain(attribute)
        if plain is not None:
            found[path] = plain
        elif type(attribute).__module__.startswith("mujoco"):
            found |= read_object(attribute, path)
    return found


def read_spec(spec: mujoco.MjSpec) -> dict[str, Value]:
    """Read every held setting from a spec.

    Parameters
    ----------
    spec : mujoco.MjSpec
        A fresh spec (the engine defaults) or a loaded one.

    Returns
    -------
    dict[str, Value]
        Setting path to plain value, in section order. Arrays are tuples of
        floats and engine enums are integers.
    """
    return (
        read_object(spec.option, "option")
        | read_object(spec.compiler, "compiler")
        | read_object(spec, "size", SIZE_FIELDS)
        | read_object(spec.visual, "visual")
        | read_object(spec.default.geom, "default.geom", GEOM_CONTACT_FIELDS)
    )


def engine_defaults() -> dict[str, Value]:
    """Read the defaults of the installed engine.

    Returns
    -------
    dict[str, Value]
        Setting path to plain value, see ``read_spec``.
    """
    return read_spec(mujoco.MjSpec())


def settings(seeds: Seeds) -> list[Setting]:
    """List every setting with its value, kind, reason and source.

    Parameters
    ----------
    seeds : Seeds
        The physical scales.

    Returns
    -------
    list[Setting]
        One entry per engine setting, in section order.
    """
    defaults = engine_defaults()
    friction_tail = tuple(
        float(term) for term in mujoco.MjSpec().default.geom.friction[1:]
    )
    overrides = {
        "option.timestep": Setting(
            name="option.timestep",
            value=seeds.timestep,
            kind="formula",
            reason="derived from the seeds, see timestep in the table",
            source=JUDGEMENT,
        ),
        "option.gravity": Setting(
            name="option.gravity",
            value=(0.0, 0.0, -float(seeds.gravity.value)),
            kind="seed",
            reason="gravity seed, pointing down the z axis",
            source="standard gravity, ISO 80000-3",
        ),
        "option.density": Setting(
            name="option.density",
            value=float(seeds.air_density.value),
            kind="seed",
            reason="air density seed (medium for fluid forces)",
            source="consensus of the research reports, wiki mjcf_option",
        ),
        "option.viscosity": Setting(
            name="option.viscosity",
            value=float(seeds.air_viscosity.value),
            kind="seed",
            reason="air viscosity seed (medium for fluid forces)",
            source="consensus of the research reports, wiki mjcf_option",
        ),
        "default.geom.solref": Setting(
            name="default.geom.solref",
            value=(seeds.contact_time_constant, DAMPING_RATIO),
            kind="formula",
            reason="contact time constant tc and critical damping ratio 1",
            source="wiki mujoco_solver_parameters; tc is judgement",
        ),
        "default.geom.friction": Setting(
            name="default.geom.friction",
            value=(seeds.friction, *friction_tail),
            kind="seed",
            reason="sliding friction seed; torsional and rolling terms kept",
            source=f"{DOCS}#default-geom-friction",
        ),
    }
    overrides |= {
        name: Setting(
            name=name,
            value=choice.value,
            kind="fixed",
            reason=choice.reason,
            source=choice.source,
        )
        for name, choice in CHOICES.items()
    }
    rows: list[Setting] = []
    for name, value in defaults.items():
        section = name.split(".")[0]
        rows.append(
            overrides.get(
                name,
                Setting(
                    name=name,
                    value=value,
                    kind="fixed",
                    reason=KEPT_REASON[section],
                    source=KEPT_SOURCE[section],
                ),
            ),
        )
    return rows
