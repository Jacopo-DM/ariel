"""Test: the Lynx arm, its table world and the shared pipeline helpers."""

from __future__ import annotations

import runpy
from pathlib import Path

import mujoco
import numpy as np
import pytest

from ariel.body_phenotypes.lynx_mjspec import lynx_arm as lynx_arm_module
from ariel.body_phenotypes.lynx_mjspec.lynx_arm import ArmConfig, LynxArm
from ariel.body_phenotypes.lynx_mjspec.table import TableWorld
from ariel.body_phenotypes.lynx_mjspec.unified_pipeline import common
from ariel.body_phenotypes.lynx_mjspec.unified_pipeline.common import (
    NUM_JOINTS,
    NUM_TUBES,
    FastNumpyNetwork,
    PolicySpec,
)

TUBE_LENGTH = 0.4
HIDDEN = 3
SPEC_DEFAULT_PARAMS = 1766
YAW = np.pi / 2
TARGET = np.array([0.2, 0.0, 1.2], dtype=np.float64)
TWO_LINKS = 2


def _compiled_arm(config: ArmConfig | None = None) -> mujoco.MjModel:
    arm = LynxArm(config=config)
    world = TableWorld()
    world.spawn(arm.spec)
    return world.spec.compile()


# --- ArmConfig and LynxArm -------------------------------------------------- #


def test_arm_config_defaults_are_independent() -> None:
    """Every ArmConfig gets its own lists."""
    first = ArmConfig()
    second = ArmConfig()
    assert first.tube_lengths == second.tube_lengths
    assert first.tube_lengths is not second.tube_lengths
    assert first.num_joints == NUM_JOINTS


def test_default_arm_compiles_with_one_joint_per_link() -> None:
    """The default arm has six joints and six actuators."""
    model = _compiled_arm()
    assert model.njnt == NUM_JOINTS
    assert model.nu == NUM_JOINTS
    del model


def test_arm_uses_the_smaller_of_joints_and_genotype() -> None:
    """Fewer genotype joints than num_joints limits the chain."""
    config = ArmConfig(num_joints=NUM_JOINTS, genotype_joints=TWO_LINKS)
    model = _compiled_arm(config)
    assert model.njnt == TWO_LINKS
    del model


def test_tube_geoms_only_for_enabled_tubes() -> None:
    """A tube flag of 0 gives a link without a tube geom."""
    with_tubes = _compiled_arm(
        ArmConfig(
            genotype_tube=[1] * NUM_TUBES,
            tube_lengths=[TUBE_LENGTH] * NUM_TUBES,
        )
    )
    without = _compiled_arm(ArmConfig(genotype_tube=[0] * NUM_TUBES))
    assert with_tubes.ngeom > without.ngeom
    del with_tubes, without


def test_arm_without_rotation_angles_still_builds() -> None:
    """Missing yaw angles default to zero."""
    model = _compiled_arm(ArmConfig(rotation_angles=[]))
    assert model.njnt == NUM_JOINTS
    del model


def test_main_block_launches_the_viewer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running the module as a script builds the arm and opens a viewer."""
    launched: list[int] = []
    monkeypatch.setattr(
        mujoco.viewer,
        "launch",
        lambda model, _data: launched.append(model.njnt),
    )
    runpy.run_path(str(Path(lynx_arm_module.__file__)), run_name="__main__")
    assert launched == [NUM_JOINTS]


# --- table world ------------------------------------------------------------ #


def test_table_world_has_a_target_site() -> None:
    """The table world compiles and carries the target site."""
    world = TableWorld()
    model = world.spec.compile()
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target") >= 0
    del model


# --- the policy network ----------------------------------------------------- #


def test_policy_spec_defaults_match_the_arm() -> None:
    """The default policy maps 15 inputs to six actions."""
    spec = PolicySpec()
    assert spec.output_size == NUM_JOINTS
    assert spec.input_size == NUM_JOINTS + NUM_JOINTS + 3


def test_num_network_params_counts_weights_and_biases() -> None:
    """The formula matches the layer shapes."""
    spec = PolicySpec(hidden_size=HIDDEN)
    expected = (
        (HIDDEN * spec.input_size + HIDDEN)
        + (HIDDEN * HIDDEN + HIDDEN)
        + (spec.output_size * HIDDEN + spec.output_size)
    )
    assert common.num_network_params(spec) == expected


def test_network_forward_is_bounded() -> None:
    """The output of the network lies in [-1, 1]."""
    spec = PolicySpec(hidden_size=HIDDEN)
    rng = np.random.default_rng(0)
    weights = rng.normal(size=common.num_network_params(spec))
    net = FastNumpyNetwork(
        spec.input_size, spec.hidden_size, spec.output_size, weights
    )
    out = net.forward(rng.normal(size=spec.input_size))
    assert out.shape == (spec.output_size,)
    assert np.all(np.abs(out) <= 1.0)


def test_network_rejects_wrong_weight_count() -> None:
    """A weight vector of the wrong size is an error."""
    with pytest.raises(ValueError, match="Invalid weight size"):
        FastNumpyNetwork(2, HIDDEN, 1, np.zeros(1))


# --- model helpers ---------------------------------------------------------- #


def test_build_model_returns_ids_that_resolve() -> None:
    """build_model compiles the arm on a table and finds its ids."""
    tubes = np.full(NUM_TUBES, TUBE_LENGTH, dtype=np.float64)
    model, data, tcp, target, joints = common.build_model(tubes)
    assert tcp >= 0
    assert target >= 0
    assert len(joints) == NUM_JOINTS
    assert model.nu == NUM_JOINTS
    del model, data


def test_build_model_clips_tube_lengths() -> None:
    """Lengths outside the bounds are clipped before building."""
    tubes = np.full(NUM_TUBES, common.TUBE_MAX * 5, dtype=np.float64)
    model, data, *_ = common.build_model(tubes)
    assert model.nbody > NUM_JOINTS
    del model, data


def test_build_model_reports_missing_sites(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the sites cannot be found the model is rejected."""
    monkeypatch.setattr(common, "resolve_site_id", lambda _model, _name: -1)
    with pytest.raises(RuntimeError, match="tcp/target"):
        common.build_model(np.full(NUM_TUBES, TUBE_LENGTH))


def test_build_model_reports_too_few_joints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fewer actuated joints than expected is an error."""
    monkeypatch.setattr(common, "get_actuated_joint_ids", lambda _model: [0])
    with pytest.raises(RuntimeError, match="actuated joints"):
        common.build_model(np.full(NUM_TUBES, TUBE_LENGTH))


def test_resolve_site_id_by_suffix_and_missing() -> None:
    """A site is found by exact name, by suffix, or reported as -1."""
    model, data, *_ = common.build_model(np.full(NUM_TUBES, TUBE_LENGTH))
    assert common.resolve_site_id(model, "target") >= 0
    assert common.resolve_site_id(model, "tcp") >= 0
    assert common.resolve_site_id(model, "no-such-site") == -1
    del model, data


def test_joint_state_and_observation_shapes() -> None:
    """Joint state has six entries each; the observation adds the offset."""
    model, data, tcp, target, joints = common.build_model(
        np.full(NUM_TUBES, TUBE_LENGTH)
    )
    mujoco.mj_forward(model, data)
    q, qd = common.get_joint_state(model, data, joints)
    assert q.shape == qd.shape == (NUM_JOINTS,)
    obs = common.build_observation(model, data, joints, tcp, target)
    assert obs.shape == (PolicySpec().input_size,)
    del model, data


def test_set_target_position_world_site() -> None:
    """A target site on the world body is moved by its own position."""
    model, data, _, target, _ = common.build_model(
        np.full(NUM_TUBES, TUBE_LENGTH)
    )
    common.set_target_position(model, data, target, TARGET)
    assert np.allclose(data.site_xpos[target], TARGET)
    del model, data


def test_set_target_position_body_site() -> None:
    """A site on a body is set relative to the parent body of that body."""
    xml = """
    <mujoco><worldbody>
      <body name="holder" pos="1 0 0"><joint type="slide" axis="1 0 0"/>
        <geom size=".1"/>
        <body name="inner"><joint type="hinge" axis="0 0 1"/>
          <geom size=".1"/><site name="s"/>
        </body>
      </body>
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "s")
    parent = int(model.body_parentid[model.site_bodyid[site]])
    common.set_target_position(model, data, site, TARGET)
    assert np.allclose(model.site_pos[site], TARGET - data.xpos[parent])
    del model, data


def test_position_delta_control_respects_limits() -> None:
    """The commanded targets stay inside the joint ranges."""
    model, data, _, _, joints = common.build_model(
        np.full(NUM_TUBES, TUBE_LENGTH)
    )
    mujoco.mj_forward(model, data)
    big = np.full(NUM_JOINTS, 10.0)
    common.apply_position_delta_control(
        model, data, joints, big, action_scale=100.0, max_delta=100.0
    )
    for i, jid in enumerate(joints):
        low, high = model.jnt_range[jid]
        assert low <= data.ctrl[i] <= high
    del model, data


def test_actuated_joint_ids_skip_actuators_without_a_joint() -> None:
    """An actuator whose transmission target is unset is left out."""
    model, data, *_ = common.build_model(np.full(NUM_TUBES, TUBE_LENGTH))
    model.actuator_trnid[0, 0] = -1
    ids = common.get_actuated_joint_ids(model)
    assert len(ids) == NUM_JOINTS - 1
    del model, data
