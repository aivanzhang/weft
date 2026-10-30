"""Shared environment-capabilities-v1 contract checks across all supported Weft profiles."""

from copy import deepcopy
import json
import subprocess
import sys

import gymnasium as gym
import numpy as np
import pytest

from weft.capabilities import (
    CAPABILITY_PROFILE,
    describe_capabilities,
    describe_run_capabilities,
    validate_capabilities,
)
from weft.env import RollingEnv
from weft.foot_reference import build_foot_reference
from weft.foot_sandbox import FootEnv
from weft.human import HumanEnv, JOINT_NAMES, build_human
from weft.io import digest, write_json
from weft.lab import Lab
from weft.scene import SCENE_UNITS, SceneEnv
from weft.spatial_lab import authored_spatial_scene, prepare_spatial_lab, write_sealed_spatial_fixture
from weft.spatial_rigid import SpatialRigidEnv
from weft.whole_body import WholeBodyEnv
from weft.world import World
from test_foot_reference import fixture as foot_fixture
from test_mobile_integration import evidence as mobile_evidence


def make_planar_scene(profile="planar-rigid-v1", multi_object=False):
    v2 = profile == "planar-rigid-v2"
    v2_common = {"angle": 0.0, "angular_velocity": 0.0, "friction": 0.5} if v2 else {}
    objects = [
        {
            "id": "actor-a",
            "label": "Actor A",
            "role": "active",
            "shape": "circle",
            "position": [0.2, 0.3],
            "velocity": [0.0, 0.0],
            "radius": 0.03,
            "color": "#DDCC00",
            "mass": 1.0,
            "provenance": {"geometry": "synthetic"},
            **v2_common,
        }
    ]
    if multi_object:
        objects.extend(
            [
                {
                    "id": "actor-b",
                    "label": "Actor B",
                    "role": "passive",
                    "shape": "box" if v2 else "circle",
                    **(
                        {"size": [0.06, 0.05], "angle": 0.1, "angular_velocity": 0.0, "friction": 0.5}
                        if v2
                        else {"radius": 0.03}
                    ),
                    "position": [0.45, 0.3],
                    "velocity": [0.0, 0.0],
                    "color": "#965644",
                    "mass": 1.2,
                    "provenance": {"geometry": "synthetic"},
                },
                {
                    "id": "wall-0",
                    "label": "Divider",
                    "role": "static",
                    "shape": "box",
                    "size": [0.12, 0.03],
                    "position": [0.5, 0.52],
                    "velocity": [0.0, 0.0],
                    "color": "#888888",
                    "mass": 1.0,
                    "provenance": {"geometry": "synthetic"},
                    **v2_common,
                },
            ]
        )
    return {
        "schema_version": 2 if v2 else 1,
        "profile": profile,
        "width": 1.0,
        "height": 0.6,
        "dt": 1 / 30,
        "gravity": [0.0, 0.0],
        "damping": 0.9,
        "restitution": 0.65,
        "objects": objects,
        "controlled_id": "actor-a",
        "target": [0.8, 0.3],
        "task": {"position_tolerance": 0.06, "speed_tolerance": 0.15, "max_steps": 300, "acceleration": 2.0},
        "relationships": [],
        "assumptions": ["Synthetic planar capability fixture"],
        **({"units": deepcopy(SCENE_UNITS), "boundary_friction": 0.5} if v2 else {}),
    }


def assert_matches_gym_env(cap, env):
    assert cap["profile"] == CAPABILITY_PROFILE
    assert cap["evidence_separation"]["physical_validation"] == "unverified"
    assert cap["entities"]["mixed_rigid_articulated"] is False
    assert cap["task_contract"]["authored_addition"] is True
    assert len(cap["unsupported_capabilities"]) >= 2
    assert cap["observation_contract"]["dimension"] == env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Discrete):
        assert cap["action_contract"]["space_type"] == "discrete"
        assert cap["action_contract"]["dimension"] == 1
        assert len(cap["action_contract"]["discrete_actions"]) == env.action_space.n
    else:
        assert cap["action_contract"]["space_type"] == "continuous_box"
        assert cap["action_contract"]["dimension"] == env.action_space.shape[0]
        assert cap["action_contract"]["continuous_bounds"] == [
            float(env.action_space.low[0]),
            float(env.action_space.high[0]),
        ]


def test_capabilities_match_live_gym_contracts_across_all_rigid_profiles():
    world = World(
        x0=0.2,
        v0=0.0,
        drag=0.5,
        radius=0.04,
        support_slope=0.0,
        support_intercept=0.75,
        width_px=640,
        height_px=360,
        source_frame=0,
    )
    rolling_env = RollingEnv(world)
    rolling_scene = {
        "schema_version": 1,
        "profile": "rolling",
        "task": rolling_env.task.to_dict(),
        "objects": [{"id": "ball-0", "label": "Ball", "role": "active", "shape": "circle", "radius": world.radius}],
    }
    rolling_cap = describe_capabilities(rolling_scene, lab_profile="rolling", learned_policy_available=True)
    assert_matches_gym_env(rolling_cap, rolling_env)
    assert rolling_cap["representation_family"] == "rigid_1d_support"
    assert rolling_cap["entities"]["multi_target_selection_endpoint"] is None

    for profile in ("planar-rigid-v1", "planar-rigid-v2"):
        scene = make_planar_scene(profile=profile, multi_object=True)
        env = SceneEnv(scene)
        try:
            cap = describe_capabilities(
                env.scene,
                lab_profile="planar",
                learned_policy_available=True,
                trained_target_id="actor-a",
                source_video_present=True,
            )
            assert_matches_gym_env(cap, env)
            assert cap["representation_family"] == "rigid_planar_2d"
            assert [a["id"] for a in cap["entities"]["controllable_actors"]] == ["actor-a", "actor-b"]
            assert cap["entities"]["multi_target_selection_endpoint"] == "/api/actor"
            assert cap["evidence_separation"]["muscle_display"] == "not_applicable"
        finally:
            env.close()

    spatial_env = SpatialRigidEnv(authored_spatial_scene())
    try:
        spatial_cap = describe_capabilities(
            spatial_env.scene,
            lab_profile="spatial-rigid",
            learned_policy_available=False,
        )
        assert_matches_gym_env(spatial_cap, spatial_env)
        assert spatial_cap["representation_family"] == "rigid_spatial_3d"
        assert spatial_cap["coordinate_frame"]["dimensions"] == 3
        assert [a["id"] for a in spatial_cap["entities"]["controllable_actors"]] == ["block", "companion"]
        assert spatial_cap["entities"]["multi_target_selection_endpoint"] == "/api/actor"
        assert spatial_cap["controllers"]["learned_policy_available"] is False
        assert spatial_cap["controllers"]["trained_target_id"] is None
    finally:
        spatial_env.close()


def test_capabilities_match_live_gym_contracts_across_all_articulated_human_profiles(tmp_path):
    hinge_ev = {
        "source_sha256": "synthetic-fixture-not-a-physical-video",
        "lengths": dict(torso=0.45, upper_arm=0.3, forearm=0.27, thigh=0.43, shin=0.43, shoulder_width=0.38, hip_width=0.28),
        "reference": [
            {"time_s": 0.0, "joints": {name: 0.1 for name in JOINT_NAMES}},
            {"time_s": 0.5, "joints": {name: 0.4 for name in JOINT_NAMES}},
            {"time_s": 1.0, "joints": {name: 0.8 for name in JOINT_NAMES}},
        ],
    }
    ev = mobile_evidence()
    for body_model, src_ev, expected_profile, expected_joints, expected_assistance in (
        ("hinge", hinge_ev, "exercise-v1", 8, False),
        ("spatial", ev, "exercise-v2", 19, False),
        ("mobile", ev, "exercise-v3", 19, True),
    ):
        spec = build_human(src_ev, tmp_path / body_model, body_model=body_model)
        env = HumanEnv(spec)
        try:
            env.scene["joint_names"] = list(env.joint_names)
            cap = describe_capabilities(
                env.scene,
                lab_profile="exercise",
                learned_policy_available=True,
                trained_target_id=env.selected_joint,
                reference_available=True,
            )
            assert cap["scene_profile"] == expected_profile
            assert_matches_gym_env(cap, env)
            assert len(cap["entities"]["controllable_actors"]) == expected_joints
            assert cap["action_contract"]["hidden_assistance"] is expected_assistance
            assert cap["evidence_separation"]["muscle_display"] == "illustrative_motor_command_not_measured_activation"
        finally:
            env.close()

    mobile_spec = build_human(ev, tmp_path / "wb_body", body_model="mobile")
    wb_env = WholeBodyEnv(mobile_spec, horizon_steps=6)
    try:
        wb_cap = describe_capabilities(
            wb_env.scene,
            lab_profile="whole_body",
            learned_policy_available=True,
            reference_available=True,
        )
        assert_matches_gym_env(wb_cap, wb_env)
        assert wb_cap["action_contract"]["dimension"] == 22
        assert wb_cap["observation_contract"]["dimension"] == 155
        assert wb_cap["action_contract"]["hidden_assistance"] is False
        assert wb_cap["controllers"]["trained_target_id"] == "all_direct_motors"
    finally:
        wb_env.close()

    _, raw_feet, foot_meta = foot_fixture()
    foot_ref = build_foot_reference(mobile_spec, raw_feet, foot_meta)
    foot_env = FootEnv(mobile_spec, foot_ref, horizon_steps=6)
    try:
        foot_cap = describe_capabilities(foot_env.scene, lab_profile="standalone", reference_available=True)
        assert_matches_gym_env(foot_cap, foot_env)
        assert foot_cap["action_contract"]["dimension"] == 28
        assert foot_cap["observation_contract"]["dimension"] == 85
        assert foot_cap["controllers"]["learned_policy_available"] is False
    finally:
        foot_env.close()


def test_validate_capabilities_rejects_dishonest_physical_or_mixed_world_claims():
    valid = describe_capabilities(authored_spatial_scene())
    tampered_val = deepcopy(valid)
    tampered_val["evidence_separation"]["physical_validation"] = "verified"
    with pytest.raises(ValueError, match="physical_validation='unverified'"):
        validate_capabilities(tampered_val)

    tampered_mixed = deepcopy(valid)
    tampered_mixed["entities"]["mixed_rigid_articulated"] = True
    with pytest.raises(ValueError, match="mixed_rigid_articulated"):
        validate_capabilities(tampered_mixed)

    tampered_unsupported = deepcopy(valid)
    tampered_unsupported["unsupported_capabilities"] = []
    with pytest.raises(ValueError, match="unsupported_capabilities"):
        validate_capabilities(tampered_unsupported)


def test_describe_run_capabilities_and_cli_keep_inspected_runs_immutable(tmp_path):
    source_run = tmp_path / "spatial-source"
    write_sealed_spatial_fixture(source_run)
    manifest_before = digest(source_run / "manifest.json")

    standalone_cap = describe_run_capabilities(source_run)
    assert standalone_cap["scene_profile"] == "spatial-rigid-v1"
    assert standalone_cap["lab_profile"] == "standalone"
    assert digest(source_run / "manifest.json") == manifest_before

    with pytest.raises(ValueError, match="outside the inspected directory"):
        describe_run_capabilities(source_run, out=source_run / "cap.json")

    lab_dir = tmp_path / "spatial-lab"
    prepare_spatial_lab(source_run, lab_dir)
    out_json = tmp_path / "lab-capabilities.json"
    completed = subprocess.run(
        [sys.executable, "-m", "weft", "capabilities", "--run", str(lab_dir), "--out", str(out_json)],
        capture_output=True,
        text=True,
        check=True,
    )
    cli_doc = json.loads(completed.stdout)
    saved_doc = json.loads(out_json.read_text())
    assert cli_doc == saved_doc
    assert saved_doc["lab_profile"] == "spatial_rigid"
    assert saved_doc["controllers"]["learned_policy_available"] is False

    lab = Lab(lab_dir)
    try:
        summary = lab.summary()
        assert summary["capabilities"] == saved_doc
    finally:
        lab.close()
