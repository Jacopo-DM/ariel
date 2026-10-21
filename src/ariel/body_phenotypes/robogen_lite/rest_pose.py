"""Read the boxes of a built body at its rest pose.

The overlap repair and the morphological measures both need, for every geom
of a body, the module that owns it, its half sizes and its pose. The body is
compiled on its own and put at joint angle zero.

Notes
-----
    * A module owns the geoms whose names start with its ``prefix`` in
        ``CoreModule.graph``; with several matches the longest prefix wins
        (``m10`` before ``m1``), and a geom no prefix matches is the core's.
"""

# Third-party libraries
from typing import cast

import mujoco
import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict

# Local libraries
from ariel.body_phenotypes.robogen_lite.config import IDX_OF_CORE
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.parameters import FrozenModel

type Floats = NDArray[np.float64]

CORE_BODY = "core"  # name of the core's body in every ARIEL robot


class RestBoxes(FrozenModel):
    """Every geom of a body: owner, half sizes and pose."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    owners: list[int]  # node id per geom
    halves: Floats  # half sizes, one row of three per geom
    poses: Floats  # 4 x 4 transform per geom, in the core's frame


def owner_by_prefix(name: str, prefixes: list[tuple[str, int]]) -> int:
    """Find the module whose prefix starts a geom name.

    Parameters
    ----------
    name : str
        The geom name.
    prefixes : list[tuple[str, int]]
        Non-empty prefixes with their node ids, longest first.

    Returns
    -------
    int
        The node id; the core's when no prefix matches.
    """
    for prefix, node in prefixes:
        if name.startswith(prefix):
            return node
    return IDX_OF_CORE


def rest_boxes(core: CoreModule) -> RestBoxes:
    """Compile a body and read its boxes at joint angle zero.

    Parameters
    ----------
    core : CoreModule
        A body that carries its graph (``CoreModule.graph``).

    Returns
    -------
    RestBoxes
        One entry per geom, in MuJoCo's geom order.

    Raises
    ------
    ValueError
        If the body has no graph or a geom is not a box.
    """
    if core.graph is None:
        msg = "The body has no graph; build it with a builder of ARIEL."
        raise ValueError(msg)
    prefixes = sorted(
        (
            (prefix, node)
            for node, prefix in core.graph.nodes(data="prefix")
            if prefix
        ),
        key=lambda pair: -len(pair[0]),
    )
    model = cast("mujoco.MjModel", core.spec.compile())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    if any(kind != mujoco.mjtGeom.mjGEOM_BOX for kind in model.geom_type):
        msg = "Only box geoms can be measured."
        raise ValueError(msg)
    owners = [
        owner_by_prefix(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or "",
            prefixes,
        )
        for geom in range(model.ngeom)
    ]
    poses = np.tile(np.eye(4), (model.ngeom, 1, 1))
    poses[:, :3, :3] = data.geom_xmat.reshape(-1, 3, 3)
    poses[:, :3, 3] = data.geom_xpos
    root = np.eye(4)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CORE_BODY)
    root[:3, :3] = data.xmat[body].reshape(3, 3)
    root[:3, 3] = data.xpos[body]
    return RestBoxes(
        owners=owners,
        halves=np.array(model.geom_size),
        poses=np.linalg.inv(root) @ poses,
    )
