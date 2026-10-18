"""Shared building blocks of the unified Lynx evolution pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import mujoco
import numpy as np

from ariel.body_phenotypes.lynx_mjspec.lynx_arm import LynxArm
from ariel.body_phenotypes.lynx_mjspec.table import TableWorld

NUM_JOINTS = 6
NUM_TUBES = 5

TUBE_MIN = 0.1
TUBE_MAX = 1.0

DEFAULT_TARGET = np.array([0.20, 0.00, 1.20], dtype=np.float64)
DEFAULT_SIM_STEPS = 3500
DEFAULT_CTRL_FREQ = 20
DEFAULT_TOUCH_THRESHOLD = 0.01

DEFAULT_ACTION_SCALE = 0.25
DEFAULT_MAX_DELTA = 0.12


@dataclass(frozen=True)
class PolicySpec:
    """Sizes and action limits of the policy network."""

    input_size: int = NUM_JOINTS + NUM_JOINTS + 3
    hidden_size: int = 32
    output_size: int = NUM_JOINTS
    action_scale: float = DEFAULT_ACTION_SCALE
    max_delta: float = DEFAULT_MAX_DELTA


class FastNumpyNetwork:
    """Three-layer MLP with tanh activations for deterministic replay."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        weights: np.ndarray,
    ) -> None:
        """Create a three-layer tanh network with zero-initialised weights.

        Raises
        ------
        ValueError
            If the weight vector has the wrong size.
        """
        w1_end = hidden_size * input_size
        b1_end = w1_end + hidden_size

        w2_end = b1_end + (hidden_size * hidden_size)
        b2_end = w2_end + hidden_size

        w3_end = b2_end + (output_size * hidden_size)
        b3_end = w3_end + output_size

        if len(weights) != b3_end:
            msg = f"Invalid weight size {len(weights)}, expected {b3_end}"
            raise ValueError(msg)

        self.w1 = weights[0:w1_end].reshape(hidden_size, input_size)
        self.b1 = weights[w1_end:b1_end]

        self.w2 = weights[b1_end:w2_end].reshape(hidden_size, hidden_size)
        self.b2 = weights[w2_end:b2_end]

        self.w3 = weights[b2_end:w3_end].reshape(output_size, hidden_size)
        self.b3 = weights[w3_end:b3_end]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Map an observation to an action in [-1, 1]^output_size.

        Returns
        -------
        np.ndarray
            The action, one value per output unit, in [-1, 1].
        """
        x = np.tanh(self.w1 @ x + self.b1)
        x = np.tanh(self.w2 @ x + self.b2)
        return cast("np.ndarray", np.tanh(self.w3 @ x + self.b3))


def num_network_params(spec: PolicySpec) -> int:
    """Return the number of weights and biases of the policy network.

    Returns
    -------
    int
        The number of weights and biases.
    """
    h = spec.hidden_size
    i = spec.input_size
    o = spec.output_size
    return (h * i + h) + (h * h + h) + (o * h + o)


def resolve_site_id(model: mujoco.MjModel, base_name: str) -> int:
    """Return the id of a site by name or name suffix, or -1.

    Returns
    -------
    int
        The site id, or -1 if no site matches.
    """
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, base_name)
    if sid != -1:
        return cast("int", sid)

    for i in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) or ""
        if name.endswith(base_name):
            return i

    return -1


def get_actuated_joint_ids(
    model: mujoco.MjModel, count: int = NUM_JOINTS
) -> list[int]:
    """Return the joint ids driven by the first `count` actuators.

    Returns
    -------
    list[int]
        The ids of the driven joints.
    """
    ids: list[int] = []
    for i in range(min(count, model.nu)):
        jid = int(model.actuator_trnid[i, 0])
        if jid >= 0:
            ids.append(jid)
    return ids


def get_joint_state(
    model: mujoco.MjModel, data: mujoco.MjData, joint_ids: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Return the positions and velocities of the given joints.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The joint positions and the joint velocities.
    """
    q = np.zeros(NUM_JOINTS, dtype=np.float64)
    qd = np.zeros(NUM_JOINTS, dtype=np.float64)

    for i, jid in enumerate(joint_ids[:NUM_JOINTS]):
        qaddr = int(model.jnt_qposadr[jid])
        daddr = int(model.jnt_dofadr[jid])
        q[i] = data.qpos[qaddr]
        qd[i] = data.qvel[daddr]

    return q, qd


def set_target_position(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_site_id: int,
    target: np.ndarray,
) -> None:
    """Move the target site to the world position `target`."""
    body_id = int(model.site_bodyid[target_site_id])

    # If the site is attached to worldbody, site_pos is already in world
    # coordinates.
    if body_id == 0:
        model.site_pos[target_site_id] = target
        mujoco.mj_forward(model, data)
        return

    # For body-attached sites, update the local site offset relative to parent
    # body.
    parent_body_id = int(model.body_parentid[body_id])
    parent_xpos = data.xpos[parent_body_id]
    model.site_pos[target_site_id] = target - parent_xpos
    mujoco.mj_forward(model, data)


def build_observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_ids: list[int],
    tcp_sid: int,
    tgt_sid: int,
) -> np.ndarray:
    """Return joint positions, velocities and target relative to the TCP.

    Returns
    -------
    np.ndarray
        The observation vector for the policy.
    """
    q, qd = get_joint_state(model, data, joint_ids)
    rel_target = data.site_xpos[tgt_sid] - data.site_xpos[tcp_sid]
    return cast(
        "np.ndarray", np.concatenate([q, qd, rel_target]).astype(np.float64)
    )


def apply_position_delta_control(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_ids: list[int],
    action: np.ndarray,
    action_scale: float,
    max_delta: float,
) -> None:
    """Set position targets from a clipped, scaled action delta."""
    q, _ = get_joint_state(model, data, joint_ids)
    delta = np.clip(action, -1.0, 1.0) * action_scale
    delta = np.clip(delta, -max_delta, max_delta)
    desired = q + delta

    for i, jid in enumerate(joint_ids[:NUM_JOINTS]):
        lo, hi = model.jnt_range[jid]
        desired[i] = np.clip(desired[i], lo, hi)

    data.ctrl[:NUM_JOINTS] = desired


def build_model(
    tube_lengths: np.ndarray,
) -> tuple[mujoco.MjModel, mujoco.MjData, int, int, list[int]]:
    """Compile the arm on a table; return model, data and ids.

    Returns
    -------
    tuple[mujoco.MjModel, mujoco.MjData, int, int, list[int]]
        The model, the data, the tcp site id, the target site id and the
        actuated joint ids.

    Raises
    ------
    RuntimeError
        If a site id cannot be resolved or too few joints are actuated.
    """
    config: dict[str, Any] = {
        "num_joints": NUM_JOINTS,
        "genotype_tube": [1] * NUM_TUBES,
        "genotype_joints": NUM_JOINTS,
        "tube_lengths": np.clip(tube_lengths, TUBE_MIN, TUBE_MAX).tolist(),
        "rotation_angles": [0.0, -1.57, 0.0, 0.0, 0.0, 0.0],
        "task": "reach",
    }

    arm = LynxArm(config=config)
    world = TableWorld()
    world.spawn(arm.spec)

    model = cast("mujoco.MjModel", world.spec.compile())
    data = mujoco.MjData(model)

    tcp_sid = resolve_site_id(model, "tcp")
    tgt_sid = resolve_site_id(model, "target")
    if tcp_sid == -1 or tgt_sid == -1:
        msg = "Could not resolve tcp/target site IDs"
        raise RuntimeError(msg)

    joint_ids = get_actuated_joint_ids(model)
    if len(joint_ids) < NUM_JOINTS:
        msg = (
            f"Expected at least {NUM_JOINTS} actuated joints, "
            f"got {len(joint_ids)}"
        )
        raise RuntimeError(msg)

    return model, data, tcp_sid, tgt_sid, joint_ids
