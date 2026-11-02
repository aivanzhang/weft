"""Readiness routes original uncertainty without changing poses or scoring gates."""

from copy import deepcopy
from hashlib import sha256
import json

import pytest

from test_exercise_body_source import pose_run
from test_native_clock_runtime import native_binding
from test_exercise_vision import inference_result
from weft import exercise_reconstruction_readiness as readiness
from weft import exercise_vision
from weft.io import canonical_hash


def raw(document):
    return (json.dumps(document, sort_keys=True) + "\n").encode()


@pytest.fixture
def source(tmp_path):
    root = pose_run(tmp_path / "poses")
    return json.loads((root / "evidence.json").read_bytes())


def replace_sample(source, index, count):
    previous = source["frames"][index]
    replacement = exercise_vision.summarize_result(inference_result(index * .01, count=count))
    replacement.update({key: previous[key] for key in ("frame", "id", "time_s", "decoder_timestamp_s", "inference_seconds")})
    source["frames"][index] = replacement
    source["quality"] = exercise_vision._quality(source)


def test_passed_source_stays_diagnostic_and_preserves_all_unsampled_slots(source):
    before = deepcopy(source)
    result = readiness.diagnose_reconstruction(raw(source))
    assert source == before
    assert result["source_quality"] == source["quality"]
    assert not result["body_admitted"] and not result["support_observability"]["verified_contact"]
    assert not result["selective_crop"]["retry_can_address_all_source_rejections"]
    assert len(result["slots"]) == source["decoded_frames"]
    assert sum(row["state"] == "not_sampled" for row in result["slots"]) == 31
    assert result["identity"]["changes_applied"] == 0
    assert all(row["contact_state"] == "unknown" for row in result["slots"])
    assert result["support_observability"]["ground_height"] is None


def test_competing_hypotheses_survive_and_block_automatic_retry(source):
    replace_sample(source, 10, 2)
    result = readiness.diagnose_reconstruction(raw(source))
    assert result["status"] == "source_quality_rejected"
    assert result["identity"]["competing_samples"] == [{"frame": 20, "evidence_id": source["frames"][10]["id"],
                                                        "retained_hypotheses": 2, "selected_hypothesis": None}]
    assert result["slots"][20]["state"] == "ambiguous_identity"
    assert 20 not in result["selective_crop"]["recoverable_unusable_frames"]
    assert not result["selective_crop"]["retry_can_address_all_source_rejections"]
    assert all(not point["available"] for point in result["slots"][20]["support_points"].values())


def test_recovery_upper_bound_is_from_existing_planner_not_manual_selection(source):
    for index in range(5, 22): replace_sample(source, index, 0)
    result = readiness.diagnose_reconstruction(raw(source))
    from weft.pose_crop_selection import plan_selective_crop
    expected = [plan_selective_crop(source, row) for row in source["frames"]]
    assert result["selective_crop"]["plans"] == expected
    assert result["selective_crop"]["optimistic_usable_ceiling"] < result["selective_crop"]["minimum_required_usable_samples"]
    assert not result["selective_crop"]["could_clear_visibility_gate"]
    assert any(row["kind"] == "obtain_new_supported_pose_context" for row in result["next_actions"])


def test_native_clock_changes_only_diagnostic_time_and_keeps_gap_endpoints(source):
    for index in (10, 11, 12): replace_sample(source, index, 0)
    binding = native_binding(source, [60000 + i * 1000 + (3000 if i >= 20 else 0) for i in range(63)],
                             time_base="1/30000", offset=-1.997)
    for row in source["frames"]:
        if row["decoder_timestamp_s"] is not None:
            row["decoder_timestamp_s"] = binding["rows"][row["frame"]]["decoder_timestamp_s"]
    binding.update(evidence_sha256=sha256(raw(source)).hexdigest(), evidence_canonical_sha256=canonical_hash(source))
    result = readiness.diagnose_reconstruction(raw(source), native_clock=binding)
    gap = next(row for row in result["unusable_spans"] if row["first_frame"] == 20)
    assert gap["last_frame"] == 24
    assert gap["first_time_s"] == pytest.approx(20 / 30 + .1)
    assert gap["following_usable_evidence_id"] == source["frames"][13]["id"]
    assert gap["interpolated_samples"] == 0
    assert result["slots"][20]["nominal_time_s"] == 20 / 30
    assert source["frames"][10]["time_s"] == 20 / 30


@pytest.mark.parametrize("change", ["quality", "order", "identity", "nominal", "derived", "hypothesis_count", "missing_hypothesis", "oversize", "duplicate_json"])
def test_invalid_source_contract_cannot_make_a_readiness_report(source, change):
    if change == "quality": source["quality"]["passes"] = False
    elif change == "order": source["frames"].reverse()
    elif change == "identity": source["frames"][0]["id"] = "invented"
    elif change == "nominal": source["frames"][1]["time_s"] += .001
    elif change == "derived": source["source_clock"] = {}
    elif change == "hypothesis_count": source["frames"][0]["pose_count"] = True
    elif change == "missing_hypothesis": source["frames"][0]["pose_count"] = 2
    data = raw(source)
    if change == "oversize": data = b" " * (readiness.MAX_INPUT_BYTES + 1)
    elif change == "duplicate_json": data = b'{"profile":1,"profile":2}'
    with pytest.raises(ValueError): readiness.diagnose_reconstruction(data)


def test_exact_evidence_bytes_bind_native_clock(source):
    binding = native_binding(source, [60000 + i * 1000 for i in range(63)], time_base="1/30000", offset=-1.997)
    binding["evidence_sha256"] = sha256(raw(source)).hexdigest()
    with pytest.raises(ValueError, match="byte hash"):
        readiness.diagnose_reconstruction(raw(source) + b" ", native_clock=binding)


def test_diagnostics_never_calls_perception_or_dynamics(source, monkeypatch):
    import mujoco
    def forbidden(*args, **kwargs): pytest.fail("No inference, decode or physics in source diagnostic")
    monkeypatch.setattr(exercise_vision, "inspect_exercise", forbidden)
    monkeypatch.setattr(exercise_vision.cv2, "VideoCapture", forbidden)
    monkeypatch.setattr(mujoco, "mj_step", forbidden)
    result = readiness.diagnose_reconstruction(raw(source))
    assert all(value == 0 for value in result["work"].values())
