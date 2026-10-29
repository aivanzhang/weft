"""Native geometry, explicit control, missing evidence and recording regressions."""

from copy import deepcopy
import json

import mujoco
import numpy as np
import pytest

from weft.foot_body import ANKLE_NAMES, FootBody, audit_foot_body
from weft.foot_reference import build_foot_reference
from weft.foot_sandbox import FootEnv, command_action, console
from weft.human import HumanEnv, build_human
from weft.io import write_json
from weft.recording import Recorder, export_recording, load_recording
from test_foot_reference import fixture as foot_fixture
from test_mobile_integration import evidence


@pytest.fixture
def inputs(tmp_path):
    spec = build_human(evidence(), tmp_path / "body", body_model="mobile")
    _, raw, metadata = foot_fixture()
    return spec, build_foot_reference(spec, raw, metadata)


def unavailable(reference, index, side="left"):
    reference["frames"][index]["feet"][side].update(status="unavailable", reference=None, endpoint_geometry=None)


def test_native_endpoints_match_frozen_geometry_without_moving_body_or_floor(inputs):
    spec, reference = inputs
    before = deepcopy(inputs)
    baseline, candidate = HumanEnv(spec), FootBody(spec, reference)
    assert candidate.model.nu == candidate.model.nq == 28
    np.testing.assert_array_equal(candidate.model.opt.gravity, baseline.model.opt.gravity)
    np.testing.assert_array_equal(candidate.model.geom_pos[candidate.mobility.floor], baseline.model.geom_pos[baseline.mobility.floor])
    for i, row in enumerate(reference["frames"]):
        original = baseline.reference_snapshot(i)
        state = candidate.reference_snapshot(i)
        for name, point in original["human"]["joints"].items():
            np.testing.assert_allclose(state["human"]["joints"][name], point, atol=1e-12, rtol=0)
        for side in ("left", "right"):
            ankle = np.array(state["human"]["joints"][side + "_ankle"])
            for part in ("heel", "foot_index"):
                expected = np.array(row["feet"][side]["endpoint_geometry"][part + "_relative"])
                actual = np.array(state["human"]["joints"][side + "_" + part]) - ankle
                np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
        assert len(state["human"]["angles"]) == 25 and len(state["human"]["muscles"]) == 50
        assert state["human"]["mobility"]["scope"] == "kinematic_overlap_not_dynamic_force"
    assert inputs == before
    np.testing.assert_array_equal(candidate.data.qpos, np.zeros(28))
    with pytest.raises(RuntimeError, match="FootEnv"):
        candidate.reset()
    baseline.close()
    candidate.close()


def test_all_original_poses_retained_and_missing_feet_not_interpolated(inputs):
    spec, reference = inputs
    unavailable(reference, 3)
    result = audit_foot_body(spec, reference)
    assert result["baseline"]["summary"]["source_pose_checks"] == len(spec["reference"])
    assert result["summary"]["body_source_poses"] == 10
    assert result["summary"]["checked"] == 5
    assert result["summary"]["unavailable"] == 5
    assert result["checks"][3]["passes_floor_limit"] is None
    assert result["checks"][3]["max_penetration_model_units"] is None
    assert result["checks"][4]["status"] == "checked"
    assert result["interactive_start_available"] is True
    assert not result["promoted"] and not result["summary"]["foot_control_grid_checked"]
    assert [row["time_s"] for row in result["checks"]] == [row["time_s"] for row in spec["reference"]]


def test_starting_gap_never_selects_a_later_available_episode(inputs):
    spec, reference = inputs
    unavailable(reference, 0)
    result = audit_foot_body(spec, reference)
    assert not result["interactive_start_available"] and result["checks"][1]["status"] == "checked"
    with pytest.raises(ValueError, match="original starting pose"):
        FootEnv(spec, reference)


def test_native_floor_penetration_includes_added_capsule_radius(inputs):
    spec, reference = inputs
    candidate = FootBody(spec, reference)
    audit = audit_foot_body(spec, reference)
    for i in range(6):
        state = candidate.reference_snapshot(i)
        lowest = min(state["human"]["joints"][side + "_" + part][2] - .018
                     for side in ("left", "right") for part in ("heel", "foot_index"))
        assert audit["checks"][i]["max_penetration_model_units"] >= max(0., -lowest) - 1e-12
    assert any(c["geom"].endswith("foot_geom") for row in audit["checks"] for c in row.get("floor_contacts", []))
    assert audit["summary"]["failed_floor_checks"] > 0
    assert audit["contract"]["body_or_floor_refit"] is False
    candidate.close()


def test_external_input_mutation_cannot_change_compiled_reference(inputs):
    candidate = FootBody(*inputs)
    before = candidate.reference_snapshot(0)
    unavailable(inputs[1], 0)
    inputs[0]["reference"][0]["joints"]["left_knee"] = 1.5
    assert candidate.reference_snapshot(0) == before
    candidate.close()


def test_missing_shape_keeps_baseline_checks_and_all_source_slots(inputs):
    spec, reference = inputs
    reference["fixed_shapes"]["left"].update(status="unavailable", fixed_parameters=None)
    for i in range(6):
        unavailable(reference, i)
    result = audit_foot_body(spec, reference)
    assert result["status"] == "unavailable"
    assert result["summary"]["checked"] == 0 and result["summary"]["unavailable"] == 10
    assert result["baseline"]["summary"]["source_pose_checks"] == 10
    with pytest.raises(ValueError, match="both fixed foot shapes"):
        FootEnv(spec, reference)


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(body_model_sha256="b" * 64),
    lambda r: r.update(source_sha256="b" * 64),
    lambda r: r["contract"].update(interpolation="linear"),
    lambda r: r["frames"][1].update(time_s=r["frames"][0]["time_s"]),
    lambda r: r["frames"][0].update(evidence_id="unrelated"),
    lambda r: r["frames"][0]["feet"]["left"]["reference"].update(heel_to_index_axis=[0., 0., 0.]),
    lambda r: r["fixed_shapes"]["left"]["fixed_parameters"].update(length=float("nan")),
])
def test_invalid_candidate_provenance_and_geometry_rejected(inputs, mutation):
    spec, reference = inputs
    mutation(reference)
    with pytest.raises(ValueError):
        FootBody(spec, reference)


def test_direct_motor_step_matches_native_engine_without_assistance(inputs):
    env = FootEnv(*inputs, horizon_steps=3)
    observation, info = env.reset(seed=9)
    assert env.observation_space.contains(observation) and observation.shape == (85,)
    assert info["hidden_assistance"] is False
    native = mujoco.MjData(env.model)
    native.qpos[:] = env.data.qpos
    mujoco.mj_forward(env.model, native)
    action = np.zeros(28)
    action[env.action_names.index("left_ankle_y")] = .1
    native.ctrl[env._actuators] = action * env.limits
    for _ in range(8):
        mujoco.mj_step(env.model, native)
    env.step(action)
    np.testing.assert_allclose(env.data.qpos, native.qpos, atol=1e-12, rtol=0)
    np.testing.assert_allclose(env.data.qvel, native.qvel, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(env.data.ctrl[env._actuators], action * env.limits)
    assert not env.base.mobility.support_enabled
    assert not np.any(env.data.qfrc_applied) and not np.any(env.data.xfrc_applied)
    state = env.snapshot()
    assert state["human"]["motor_units"] == ["Nm"] * 25 + ["N"] * 3
    assert state["human"]["angles"]["left_ankle_y"] != pytest.approx(env.target[20])
    env.close()


@pytest.mark.parametrize("action", [True, [0.] * 22, [0.] * 29, [True] + [0.] * 27,
    [float("nan")] + [0.] * 27, [1.001] + [0.] * 27, ["0"] * 28])
def test_invalid_actions_do_not_advance_physics(inputs, action):
    env = FootEnv(*inputs)
    env.reset()
    before = env.data.qpos.copy()
    with pytest.raises(ValueError):
        env.step(action)
    np.testing.assert_array_equal(env.data.qpos, before)
    assert env.data.time == 0
    env.close()


def test_recording_preserves_all_motor_commands_feet_and_visual_replay(inputs, tmp_path):
    env = FootEnv(*inputs, horizon_steps=6)
    env.reset(seed=9)
    recorder = Recorder(tmp_path / "session", env.scene)
    recorder.start(env.snapshot())
    action, steps = command_action("motor right_ankle_x .1 2", env)
    assert steps == 2
    for _ in range(steps):
        values = action()
        _, reward, terminated, truncated, _ = env.step(values)
        recorder.append({"type": "step", "action": values.tolist(), "reward": reward,
                         "terminated": terminated, "truncated": truncated, "snapshot": env.snapshot()})
    recorder.close()
    loaded = load_recording(recorder.root)
    assert loaded["events"][-1]["snapshot"] == env.snapshot()
    assert len(loaded["events"][-1]["action"]) == 28
    result = export_recording(recorder.root)
    assert result["video_frames"] == 3
    html = (recorder.root / "replay.html").read_text()
    assert "UNVERIFIED FEET" in html and "STARTING-POSE TASK" in html
    assert "Close-up of feet" in html and "footFocus.addEventListener('change',draw)" in html
    env.close()


@pytest.mark.parametrize("mutation", [
    lambda s: s["human"].pop("foot_candidate"),
    lambda s: s["human"]["angles"].pop("left_ankle_x"),
    lambda s: s["human"]["muscles"].pop(),
    lambda s: s["human"]["foot_candidate"].update(physical_validation="verified"),
    lambda s: s["human"]["joints"].pop("right_heel"),
    lambda s: s["human"]["segments"].pop(),
])
def test_recording_rejects_incomplete_or_mislabelled_foot_state(inputs, tmp_path, mutation):
    env = FootEnv(*inputs)
    env.reset()
    state = env.snapshot()
    mutation(state)
    with pytest.raises(ValueError):
        Recorder(tmp_path / "invalid-session", env.scene).start(state)
    assert not (tmp_path / "invalid-session").exists()
    env.close()


def test_console_records_hold_release_reset_and_rejects_unknown_motor(inputs, tmp_path):
    body, feet = tmp_path / "source" / "human.json", tmp_path / "source" / "feet.json"
    body.parent.mkdir()
    write_json(body, inputs[0])
    write_json(feet, inputs[1])
    commands = iter(["motor imaginary 1", "hold 2", "reset", "release", "quit"])
    messages = []
    result = console(body, feet, tmp_path / "session", input_fn=lambda _: next(commands), output_fn=messages.append)
    assert result["video_frames"] == 5 and result["steps"] == 3 and result["resets"] == 1
    assert any("Unknown motor" in message for message in messages)
    assert (tmp_path / "session/current-frame.png").is_file()
    assert json.loads((tmp_path / "session/candidate-audit.json").read_text())["promoted"] is False
    assert json.loads(body.read_text()) == inputs[0]
