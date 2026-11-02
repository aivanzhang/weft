"""Complete model comparison retains regressions, ambiguity and independent gates."""

from copy import deepcopy
import json

import pytest

from test_exercise_body_source import pose_run
from test_exercise_reconstruction_readiness import replace_sample
from weft.exercise_vision import MODEL_VARIANTS
from weft.pose_model_comparison import compare_pose_models


def raw(value):
    return json.dumps(value, sort_keys=True).encode()


@pytest.fixture
def branches(tmp_path):
    source = pose_run(tmp_path / "poses", mode="image")
    baseline = json.loads((source / "evidence.json").read_bytes())
    candidate = deepcopy(baseline)
    candidate["model"].update(MODEL_VARIANTS["heavy"], name="Analytical Heavy model fixture")
    return baseline, candidate


def test_complete_comparison_counts_recovery_and_regression_without_merging(branches):
    baseline, candidate = branches
    replace_sample(candidate, 0, 1)
    replace_sample(candidate, 10, 0)
    saved = deepcopy(branches)
    report = compare_pose_models(raw(baseline), raw(candidate))
    assert branches == saved
    assert report["transitions"]["recovered_usable"] == report["transitions"]["lost_usable"] == 1
    assert report["net_usable_samples"] == 0 and report["merged_frames"] == 0
    assert len(report["rows"]) == len(baseline["frames"])
    assert all(row["selected_for_merged_output"] is False for row in report["rows"])
    assert report["body_admitted"] is False and report["contact_state"] == "unknown"


def test_competing_candidate_remains_rejected_even_when_visibility_passes(branches):
    baseline, candidate = branches
    replace_sample(candidate, 10, 2)
    report = compare_pose_models(raw(baseline), raw(candidate))
    assert report["candidate_quality"]["tracked_fraction"] > .65
    assert not report["candidate_pose_quality_passes"]
    assert report["decision"] == "candidate_remains_source_quality_rejected"
    assert report["rows"][10]["candidate_hypothesis_count"] == 2


@pytest.mark.parametrize("changed", ["source", "omit", "decoder", "mode", "capacity", "model", "same", "runtime", "quality", "refined"])
def test_changed_source_or_weakened_branch_contract_rejects(branches, changed):
    baseline, candidate = branches
    if changed == "source": candidate["source_sha256"] = "a" * 64
    elif changed == "omit": candidate["frames"].pop()
    elif changed == "decoder": candidate["frames"][1]["decoder_timestamp_s"] += .001
    elif changed == "mode": candidate["model"]["running_mode"] = "VIDEO"
    elif changed == "capacity": candidate["model"]["maximum_pose_hypotheses"] = 1
    elif changed == "model": candidate["model"]["sha256"] = "a" * 64
    elif changed == "same": candidate["model"] = deepcopy(baseline["model"])
    elif changed == "runtime": candidate["model"]["mediapipe_version"] = "different"
    elif changed == "quality": candidate["quality"]["passes"] = not candidate["quality"]["passes"]
    else: candidate["frames"][0]["image_refinement"] = {}
    with pytest.raises(ValueError): compare_pose_models(raw(baseline), raw(candidate))
