"""Saved action hashes, protected tasks and explicit original-runtime gates precede replay."""

from copy import deepcopy
from hashlib import sha256
import json
import time

import pytest

from weft import shared_exercise_trial_replay as module
from weft.recording import load_recording
from weft.shared_exercise_learning import _trial, _runtime_versions
from weft.shared_exercise_task import SharedExerciseTaskEnv
from test_shared_exercise_recording import short_task


@pytest.fixture(scope="module")
def source():
    scene, task = short_task()
    env = SharedExerciseTaskEnv(scene, task)
    try:
        trial = _trial(env, "authored", 1729, {"jitter": False, "jitter_scale": 0.}, None,
                       time.monotonic() + 10, trace=True)
    finally:
        env.close()
    return scene, task, trial


def run(source, out, *, trial=None, **options):
    scene, task, original = source
    raw = json.dumps(original if trial is None else trial, allow_nan=False).encode()
    args = {"expected_trial_sha256": sha256(raw).hexdigest(), "expected_runtime_sha256": module.replay_runtime_hashes(scene),
            "expected_runtime_versions": _runtime_versions(), "controller": "authored", **options}
    return module.replay_saved_trial(scene, task, raw, out, **args)


def test_actual_saved_actions_export_and_exact_substeps_without_second_rollout(source, tmp_path):
    original = deepcopy(source)
    result = run(source, tmp_path / "replay")
    assert result["exact_state_replay"] and result["control_steps"] == 3 and result["substeps"] == 48
    loaded = load_recording(tmp_path / "replay/recording")
    assert len(loaded["events"]) == 4
    assert loaded["events"][-1]["snapshot"] == source[2]["final_snapshot"]
    assert all(row["controller"] == "authored" for row in loaded["events"][1:])
    assert (tmp_path / "replay/recording/replay.html").is_file()
    assert source == original


@pytest.mark.parametrize("change", ["trace", "count", "action", "termination", "reset", "initial", "final", "status"])
def test_incomplete_or_changed_contract_rejects_before_creating_output(source, tmp_path, change):
    trial = deepcopy(source[2])
    if change == "trace": trial["actions"].pop()
    if change == "count": trial["control_steps"] = 2
    if change == "action": trial["actions"][0]["action"][0] = 2.
    if change == "termination": trial["actions"][0]["terminated"] = True
    if change == "reset": trial["reset_options"]["jitter"] = True
    if change == "initial": trial["initial_state_sha256"] = "0" * 64
    if change == "final": trial["actions"][-1]["snapshot_sha256"] = "0" * 64
    if change == "status": trial["status"] = "interrupted"
    with pytest.raises(ValueError): run(source, tmp_path / "reject", trial=trial)
    assert not (tmp_path / "reject").exists()


@pytest.mark.parametrize("option", ["expected_trial_sha256", "expected_runtime_sha256", "expected_runtime_versions"])
def test_original_hash_and_execution_identity_are_mandatory(source, tmp_path, option):
    with pytest.raises(ValueError): run(source, tmp_path / "reject", **{option: "wrong"})
    assert not (tmp_path / "reject").exists()


def test_divergence_stops_after_first_action_and_preserves_failed_attempt(source, tmp_path):
    trial = deepcopy(source[2]); trial["actions"][0]["snapshot_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="boundary differs at action 0"):
        run(source, tmp_path / "failed", trial=trial)
    root = tmp_path / "failed"
    assert json.loads((root / "manifest.json").read_text())["status"] == "failed"
    assert json.loads((root / "failure.json").read_text())["partial_evidence_preserved"]
    assert len((root / "recording/events.jsonl").read_text().splitlines()) == 1
    assert len((root / "diagnostic/substeps.jsonl").read_text().splitlines()) == 16
    assert not (root / "recording/video.mp4").exists()


def test_authored_label_requires_exact_authored_actions(source, tmp_path):
    trial = deepcopy(source[2]); trial["actions"][0]["action"][0] = .125
    with pytest.raises(ValueError, match="claimed authored actions differ"):
        run(source, tmp_path / "failed", trial=trial)
    assert not (tmp_path / "failed/diagnostic/substeps.jsonl").exists()


def test_existing_output_is_not_overwritten(source, tmp_path):
    with pytest.raises(ValueError, match="new output"):
        run(source, tmp_path)
