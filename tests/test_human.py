from copy import deepcopy
import json

import numpy as np
import pytest

pytest.importorskip("mujoco")

from gymnasium.utils.env_checker import check_env

from weft.human import HumanEnv, JOINT_LIMITS, JOINT_NAMES, build_human, evaluate_reference, validate_human
from weft.human_learning import policy_action, train_human


@pytest.fixture
def human_evidence():
    return {"source_sha256": "synthetic-fixture-not-a-physical-video",
            "lengths": dict(torso=0.45, upper_arm=0.3, forearm=0.27, thigh=0.43, shin=0.43,
                            shoulder_width=0.38, hip_width=0.28),
            "reference": [{"time_s": 0.0, "joints": {name: 0.1 for name in JOINT_NAMES}},
                          {"time_s": 0.5, "joints": {name: 0.4 for name in JOINT_NAMES}},
                          {"time_s": 1.0, "joints": {name: 0.8 for name in JOINT_NAMES}}]}


@pytest.fixture
def human_spec(human_evidence, tmp_path):
    return build_human(human_evidence, tmp_path / "model")


def test_human_contract_clamps_explicitly_and_preserves_inputs(human_evidence, tmp_path):
    human_evidence["reference"][1]["joints"]["left_knee"] = -0.3
    spec = build_human(human_evidence, tmp_path / "clamped")
    assert spec["reference"][1]["joints"]["left_knee"] == 0
    assert human_evidence["reference"][1]["joints"]["left_knee"] == -0.3
    assert spec["clamped_reference_samples"][0]["observed_radians"] == -0.3
    with pytest.raises(FileExistsError):
        build_human(human_evidence, tmp_path / "clamped")
    for field, value in [("dt", float("nan")), ("schema_version", True), ("profile", "unknown")]:
        candidate = deepcopy(spec)
        candidate[field] = value
        with pytest.raises(ValueError):
            validate_human(candidate)
    candidate = deepcopy(spec)
    candidate["reference"][1]["time_s"] = 0
    with pytest.raises(ValueError, match="increase"):
        validate_human(candidate)


def test_human_gym_contract_determinism_and_reference_does_not_mutate(human_spec):
    env = HumanEnv(human_spec)
    check_env(env, skip_render_check=True)
    env.reset(seed=12, options={"jitter": False})
    original = env.snapshot()
    reference = env.reference_snapshot(2)
    assert reference["human"]["angles"]["left_knee"] == 0.8
    assert reference["human"]["reference"]
    assert env.snapshot() == original
    first = env.step(1)
    state = env.snapshot()
    env.reset(seed=12, options={"jitter": False})
    second = env.step(1)
    assert np.array_equal(first[0], second[0])
    assert state == env.snapshot()
    env.close()
    with pytest.raises(RuntimeError):
        env.step(0)


def test_motors_change_dynamics_and_muscles_are_only_proxies(human_spec):
    coast, flex = HumanEnv(human_spec), HumanEnv(human_spec)
    for env in (coast, flex):
        env.reset(seed=0, options={"jitter": False})
    for _ in range(8):
        coast.advance_physics(0)
        flex.advance_physics(1)
    a, b = coast.snapshot(), flex.snapshot()
    assert b["human"]["angles"]["left_knee"] > a["human"]["angles"]["left_knee"] + 0.1
    assert all(m["status"] == "illustrative" and 0 <= m["activation"] <= 1 for m in b["human"]["muscles"])
    assert any(m["activation"] > 0 for m in b["human"]["muscles"] if m["name"].startswith("left_knee"))
    assert any(np.linalg.norm(obj["velocity"]) > 0 for obj in b["objects"])
    assert b["human"]["joints"]["pelvis"] == a["human"]["joints"]["pelvis"]


def test_authored_joint_limits_remain_stable_under_sustained_action(human_spec):
    env = HumanEnv(human_spec)
    for joint in JOINT_NAMES:
        env.reset(seed=1, options={"jitter": False, "selected_joint": joint})
        for _ in range(100):
            env.advance_physics(1)
        snapshot = env.snapshot()
        low, high = JOINT_LIMITS[joint]
        assert low - 0.1 <= snapshot["human"]["angles"][joint] <= high + 0.1
        assert np.isfinite(snapshot["human"]["angles"][joint])
    with pytest.raises(ValueError):
        env.select_joint("neck")
    with pytest.raises(ValueError):
        env.advance_physics(True)


def test_human_reference_evaluation_preserves_dynamic_trace(human_spec, tmp_path):
    result = evaluate_reference(human_spec, tmp_path)
    artifact = json.loads((tmp_path / "human-reference-rollout.json").read_text())
    assert result["frames"] == len(artifact["states"]) == 31
    assert np.isfinite(result["reference_servo_rmse_radians"])
    assert artifact["states"][1]["time_s"] > 0


def test_human_rl_records_actual_learning_and_matched_trials(human_spec, tmp_path):
    output = tmp_path / "learning"
    result = train_human(human_spec, output, episodes=4, eval_episodes=2, seed=8)
    policy = json.loads((output / "policy.json").read_text())
    evaluation = json.loads((output / "rl-evaluation.json").read_text())
    assert result["training"]["completed_episodes"] == 4
    assert np.count_nonzero(np.asarray(policy["q_values"])) > 0
    assert 0 <= policy_action(policy, [0.2, 0]) <= 3
    for value in evaluation["policies"].values():
        assert [trial["seed"] for trial in value["trials"]] == evaluation["matched_episode_seeds"]
    with pytest.raises(FileExistsError):
        train_human(human_spec, output, episodes=4, eval_episodes=2)


def test_reference_evaluation_covers_beyond_the_old_fifteen_second_cutoff(human_evidence, tmp_path):
    human_evidence['reference'][-1]['time_s'] = 16
    spec = build_human(human_evidence, tmp_path / 'long')
    result = evaluate_reference(spec, tmp_path)
    assert result['frames'] == 481
    assert result['evaluated_duration_s'] == pytest.approx(16)
    assert result['duration_coverage_fraction'] == pytest.approx(1)
    assert set(result['per_joint_rmse_radians']) == set(JOINT_NAMES)
