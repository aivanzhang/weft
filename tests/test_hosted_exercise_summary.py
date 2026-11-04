"""Pin the artifact names and keys the hosted exercise summary reads.

The summary shipped empty baselines for a release because it looked for them in
human-evaluation.json, which holds the reference servo's tracking error instead. These
tests keep the two evaluations distinct and keep absent stages absent.
"""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hosting"))
from exercise_job import RESULT_FILES, summarize  # noqa: E402

RL_EVALUATION = {
    "selected_joint": "left_knee", "target_angle": 1.9432,
    "success_definition": "Angle error ≤0.08 radians AND angular speed ≤0.15 radians/second",
    "policies": {
        "q_learning": {"episodes": 100, "successes": 100, "success_rate": 1.0,
                       "mean_return": 12.515212, "trials": [{"seed": 1, "success": True}]},
        "random": {"episodes": 100, "successes": 1, "success_rate": 0.01, "mean_return": -6.76464},
        "coast": {"episodes": 100, "successes": 0, "success_rate": 0.0, "mean_return": -5.04414},
    },
}
# Reference-servo tracking error: a different measurement, with no policy successes in it.
HUMAN_EVALUATION = {
    "reference_servo_rmse_radians": 0.040126, "frames": 246,
    "duration_coverage_fraction": 1.0, "maximum_floor_penetration_model_units": 0.0022548,
    "per_joint_rmse_radians": {"left_knee": 0.0098}, "scope": "authored reference servo",
}


def write_run(directory, **artifacts):
    directory.mkdir(parents=True, exist_ok=True)
    for name, payload in artifacts.items():
        (directory / f"{name.replace('_', '-')}.json").write_text(json.dumps(payload))
    return directory


def test_baselines_come_from_rl_evaluation_not_the_servo_tracking_artifact(tmp_path):
    run = write_run(tmp_path / "run", rl_evaluation=RL_EVALUATION,
                    human_evaluation=HUMAN_EVALUATION,
                    task={"selected_joint": "left_knee", "angle_tolerance_radians": 0.08},
                    training={"algorithm": "human_tabular_q_learning",
                              "completed_episodes": 1500, "seconds": 9.6134},
                    pose_quality={"sampled_frames": 103, "tracked_frames": 103,
                                  "tracked_fraction": 1.0, "passes": True,
                                  "warnings": ["out-of-plane motion"]})
    summary = summarize(run)
    assert summary["baselines"] == {
        "coast": {"successes": 0, "episodes": 100, "success_rate": 0.0, "mean_return": -5.044},
        "q_learning": {"successes": 100, "episodes": 100, "success_rate": 1.0,
                       "mean_return": 12.515},
        "random": {"successes": 1, "episodes": 100, "success_rate": 0.01, "mean_return": -6.765},
    }
    # Per-episode trials stay out; the summary must remain small enough for one response header.
    assert "trials" not in json.dumps(summary)
    assert summary["task"]["joint"] == "left_knee"
    assert summary["task"]["angle_tolerance_radians"] == 0.08
    assert summary["reference_tracking"]["servo_rmse_radians"] == 0.0401
    assert summary["reference_tracking"]["frames"] == 246
    assert summary["training"] == {"algorithm": "human_tabular_q_learning", "episodes": 1500,
                                   "seconds": 9.61}
    assert summary["pose"] == {"sampled_frames": 103, "tracked_frames": 103,
                               "tracked_fraction": 1.0, "passes": True, "warnings": 1}
    assert len(json.dumps(summary)) < 16 * 1024


def test_report_supplies_the_summary_when_standalone_artifacts_are_absent(tmp_path):
    run = write_run(tmp_path / "run", report={
        "seconds": 29.216548, "limits": ["Pose is estimated from one camera"],
        "learning": {"evaluation": RL_EVALUATION}, "reference_tracking": HUMAN_EVALUATION,
        "pose_quality": {"sampled_frames": 103, "tracked_frames": 100,
                         "tracked_fraction": 0.9709, "passes": True, "warnings": []}})
    summary = summarize(run)
    assert summary["baselines"]["q_learning"]["successes"] == 100
    assert summary["pose"]["tracked_frames"] == 100
    assert summary["seconds"] == 29.2
    assert summary["limits"] == ["Pose is estimated from one camera"]


def test_absent_stages_stay_absent_and_unreadable_artifacts_do_not_fail(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "rl-evaluation.json").write_text("{not json")
    summary = summarize(run)
    for key in ("baselines", "training", "reference_tracking", "pose", "manifest_sha256",
                "manifest_files"):
        assert key not in summary
    # The artifact list reports what the archive carries, so an unparseable file still counts.
    assert summary["artifacts"] == ["rl-evaluation.json"]
    assert summary["limits"] == []


def test_manifest_digest_pins_the_run_and_non_finite_values_become_null(tmp_path):
    run = write_run(tmp_path / "run",
                    manifest={"schema": 1, "files": {"human.xml": "ab", "policy.json": "cd"}},
                    rl_evaluation={"policies": {"q_learning": {
                        "episodes": 1, "successes": 1, "success_rate": 1.0,
                        "mean_return": float("inf")}}})
    summary = summarize(run)
    assert summary["manifest_files"] == 2
    assert len(summary["manifest_sha256"]) == 64
    assert summary["baselines"]["q_learning"]["mean_return"] is None
    # The job writes result.json with allow_nan=False, so a non-finite leak would raise here.
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("name", RESULT_FILES)
def test_every_allowlisted_artifact_is_reported_when_present(tmp_path, name):
    run = tmp_path / "run"
    run.mkdir()
    (run / name).write_text("{}")
    assert summarize(run)["artifacts"] == [name]
