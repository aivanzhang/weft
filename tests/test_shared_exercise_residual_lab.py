"""Residual policies retain task meaning, selected intent and exact replay in the lab."""

import json
import shutil

import pytest

from weft.io import digest, write_json
from weft.lab import Lab
from weft.recording import load_recording
from weft.shared_exercise_lab import load_shared_exercise_lab, prepare_shared_exercise_lab
from weft.shared_exercise_residual_learning import (
    evaluate_shared_exercise_residual_policy, train_shared_exercise_residual_policy,
)
from test_shared_exercise_policy_lab import seal
from test_shared_exercise_run import source_fixture


@pytest.fixture(scope="module")
def residual_bundles(tmp_path_factory):
    cache = {}

    def get(root_assistance=False):
        if root_assistance not in cache:
            root = tmp_path_factory.mktemp("residual_lab_bundle")
            source, scene = source_fixture(root)
            learning, evaluation = root / "learning", root / "evaluation"
            train_shared_exercise_residual_policy(source, learning, root_assistance=root_assistance,
                seed=7091, generations=1, population=4, max_seconds=20)
            evaluate_shared_exercise_residual_policy(learning, evaluation, seed=7092, trials=1, max_seconds=20)
            cache[root_assistance] = {"source": source, "scene": scene, "learning": learning, "evaluation": evaluation}
        return cache[root_assistance]

    return get


@pytest.mark.parametrize("root_assistance", [False, True])
def test_residual_lab_controls_record_and_resume_the_exact_policy(tmp_path, residual_bundles, root_assistance):
    bundle = residual_bundles(root_assistance)
    lab_root = tmp_path / "lab"
    before = {path: digest(path) for directory in (bundle["source"], bundle["learning"], bundle["evaluation"])
              for path in directory.rglob("*") if path.is_file()}
    prepared = prepare_shared_exercise_lab(bundle["source"], lab_root, root_assistance=root_assistance,
        policy_run=bundle["learning"], evaluation_run=bundle["evaluation"])
    policy_hash = digest(bundle["learning"] / "policy.json")
    assert prepared["policy_sha256"] == policy_hash
    assert (lab_root / "run/policy.json").read_bytes() == (bundle["learning"] / "policy.json").read_bytes()
    lab = Lab(lab_root)
    try:
        summary = lab.summary()
        assert summary["exercise_policy"]["profile"] == "shared-exercise-policy-v2"
        assert summary["exercise_policy"]["evaluation_status"] == "evaluated"
        assert summary["exercise_policy"]["selected_feedback_kind"] in {"authored_unchanged", "searched_joint_residuals"}
        count = 22 if root_assistance else 19
        assert len(summary["action_contract"]["names"]) == count
        assert summary["exercise_task"]["control"]["object_forces"] == "forbidden"
        lab.control("/api/record/start", {})
        for payload in ({"action": [0.] * count}, {"controller": "reference"}, {"policy": True}):
            lab.control("/api/step", payload)
        lab.control("/api/reset", {"seed": 7193})
        lab.control("/api/step", {"policy": True, "steps": 3})
        stopped = lab.control("/api/record/stop", {})
        exported = lab.export_future.result(timeout=30)
        assert exported["action_replay"]["exact_state_replay"]
        assert exported["action_replay"]["policy_sha256"] == policy_hash
        assert exported["action_replay"]["learned_actions_verified"] == 4
        assert exported["events"] == exported["video_frames"] == 8
        saved = load_recording(lab_root / "sessions" / stopped["directory"])
        assert saved["scene"]["learned_policy"]["document_utf8"].encode() == (bundle["learning"] / "policy.json").read_bytes()
        steps = [event for event in saved["events"] if event["type"] == "step"]
        assert [event["controller"] for event in steps] == ["manual", "authored"] + ["learned"] * 4
        assert all(not any(event["snapshot"]["engine_state"]["normalized_action"][count:]) for event in saved["events"])
    finally:
        lab.close()
    resumed = Lab(lab_root)
    try:
        summary = resumed.summary()
        assert summary["exercise_policy"]["policy_sha256"] == policy_hash
        assert any(row["name"] == stopped["directory"] and row["complete"] for row in summary["sessions"])
    finally:
        resumed.close()
    assert {path: digest(path) for path in before} == before


def test_residual_lab_rejects_a_false_unchanged_baseline_label(tmp_path, residual_bundles):
    bundle = residual_bundles()
    changed = tmp_path / "changed"
    shutil.copytree(bundle["learning"], changed)
    policy = json.loads((changed / "policy.json").read_bytes())
    policy["residual_coefficients"][1][0] = .01
    write_json(changed / "policy.json", policy)
    training = json.loads((changed / "training.json").read_bytes())
    training.update(policy_sha256=digest(changed / "policy.json"), selected_feedback_kind="authored_unchanged",
                    selected_authored_baseline=True, selected_coast_baseline=False)
    write_json(changed / "training.json", training)
    seal(changed)
    with pytest.raises(ValueError, match="selection metadata"):
        prepare_shared_exercise_lab(bundle["source"], tmp_path / "unused", policy_run=changed)
    assert not (tmp_path / "unused").exists()


def test_residual_package_without_evaluation_stays_unavailable_not_passed(tmp_path, residual_bundles):
    bundle = residual_bundles()
    prepare_shared_exercise_lab(bundle["source"], tmp_path / "lab", policy_run=bundle["learning"])
    loaded = load_shared_exercise_lab(tmp_path / "lab/run")
    assert loaded["contract"]["exercise_policy"]["evaluation_status"] == "not_attached"
    assert loaded["contract"]["exercise_policy"]["evaluation_successes"] is None
