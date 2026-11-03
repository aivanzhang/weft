"""Ambiguous detections remain visible barriers, never a permission to join people."""

from copy import deepcopy
from hashlib import sha256
import json

import pytest

from weft.pose_identity_windows import screen_identity_windows


def encoded(evidence, raw):
    raw_bytes = json.dumps(raw, sort_keys=True).encode()
    evidence = deepcopy(evidence)
    evidence["pose_identity"]["raw_artifact_sha256"] = sha256(raw_bytes).hexdigest()
    return json.dumps(evidence, sort_keys=True).encode(), raw_bytes


def fixture(count=24, barriers=(9, 13)):
    frames = []
    for i in range(count):
        competing = i in barriers
        frames.append({"frame": i, "id": f"{'a'*64}:frame:{i}", "time_s": i/10,
                       "decoder_timestamp_s": i/10+.003, "status": "multiple_people" if competing else "tracked",
                       "pose_count": 2 if competing else 1,
                       "identity_status": "unresolved_multiple_pose_hypotheses" if competing else "single_model_hypothesis_not_verified_person_count",
                       "landmarks": [] if competing else [{"x": .5}], "world_landmarks": [],
                       **({"poses": [{"landmarks": [{"x": .4}]}, {"landmarks": [{"x": .6}]}]} if competing else {})})
    evidence = {"schema_version": 1, "profile": "exercise-pose-v1", "source_sha256": "a"*64,
                "model": {"name": "fixture-not-inference"}, "fps": 10., "inference_stride": 1,
                "decoded_frames": count, "frames": frames,
                "pose_identity": {"mode": "raw", "repaired_frames": 0},
                "quality": {"passed": False, "errors": ["original global identity veto"]}, "reference": []}
    return evidence, {"frames": deepcopy(frames), "model": deepcopy(evidence["model"])}


def test_barriers_retain_full_source_without_choosing_or_joining_hypotheses():
    evidence, raw = fixture(); source = deepcopy((evidence, raw)); inputs = encoded(evidence, raw)
    report = screen_identity_windows(*inputs)
    assert (evidence, raw) == source
    assert report["retained_inputs"]["evidence"] == json.loads(inputs[0])
    assert report["retained_inputs"]["raw_poses"] == raw
    assert report["retained_inputs"]["evidence"]["quality"] == evidence["quality"]
    assert len(report["sample_slots"]) == 24
    assert [b["frame"] for b in report["identity_barriers"]] == [9, 13]
    assert [w["frame_interval"] for w in report["windows"]] == [[0, 8], [10, 12], [14, 23]]
    assert [w["screen_passed"] for w in report["windows"]] == [True, False, True]
    assert set(report["windows"][1]["screen_failures"]) == {"insufficient_usable_samples", "insufficient_usable_span"}
    assigned = [i for w in report["windows"] for i in w["sample_indices"]]
    assert sorted(assigned + [b["sample_index"] for b in report["identity_barriers"]]) == list(range(24))
    assert all(b["selected_hypothesis"] is None and not b["interpolation_across_barrier_allowed"] for b in report["identity_barriers"])
    assert not report["simulation_ready"] and not report["full_clip_body_qualified"]
    assert not report["complete_native_clock_verified"] and report["selected_window"] is None
    assert all(not w["body_qualified"] and not w["identity_verified"] for w in report["windows"])
    assert report["sample_slots"][14]["source_time_s"] == 1.4
    assert report["sample_slots"][14]["decoder_timestamp_s"] == raw["frames"][14]["decoder_timestamp_s"]


def test_missing_samples_count_against_coverage_and_are_never_filled():
    evidence, raw = fixture(count=20, barriers=())
    for i in range(7):
        for doc in (evidence, raw):
            doc["frames"][i].update(status="no_pose", pose_count=0, landmarks=[])
    result = screen_identity_windows(*encoded(evidence, raw))
    assert result["windows"][0]["usable_fraction"] == .65 and result["windows"][0]["screen_passed"]
    assert result["windows"][0]["usable_sample_indices"] == list(range(7, 20))
    for doc in (evidence, raw):doc["frames"][7].update(status="low_confidence")
    result = screen_identity_windows(*encoded(evidence, raw))
    assert result["windows"][0]["screen_failures"] == ["insufficient_usable_fraction"]


@pytest.mark.parametrize("barriers,expected", [((0, 1, 23), [[2, 22]]), (tuple(range(24)), []), ((), [[0, 23]])])
def test_boundary_adjacent_and_all_ambiguous_frames_do_not_create_empty_or_crossing_windows(barriers, expected):
    report = screen_identity_windows(*encoded(*fixture(barriers=barriers)))
    assert [w["frame_interval"] for w in report["windows"]] == expected
    assert len(report["sample_slots"]) == 24


@pytest.mark.parametrize("change", [
    lambda e: e["frames"].pop(5),
    lambda e: e["frames"][5].update(frame=4),
    lambda e: e["frames"][5].update(time_s=.5001),
    lambda e: e["frames"][5].update(status=[]),
    lambda e: e["frames"][5].update(poses=None),
    lambda e: e["frames"][9].update(status="tracked"),
    lambda e: e["frames"][9]["poses"].pop(),
    lambda e: e["frames"][9].update(landmarks=[{"x": .4}]),
    lambda e: e["frames"][9].update(identity_status="verified"),
    lambda e: e["frames"][0].update(pose_count=0),
    lambda e: e["pose_identity"].update(mode="repaired"),
    lambda e: e.update(source_clock={}),
    lambda e: e.update(fps=float("nan")),
])
def test_incomplete_retimed_selected_and_repaired_evidence_is_rejected(change):
    evidence, raw = fixture(); change(evidence); raw["frames"] = deepcopy(evidence["frames"])
    with pytest.raises(ValueError):screen_identity_windows(*encoded(evidence, raw))


def test_raw_artifact_pin_and_full_row_equality_are_required():
    evidence, raw = fixture(); e, r = encoded(evidence, raw)
    with pytest.raises(ValueError, match="exact raw artifact"):screen_identity_windows(e, r+b" ")
    raw["frames"][9]["poses"][1]["landmarks"][0]["x"] = .7
    with pytest.raises(ValueError, match="Every original pose row"):screen_identity_windows(*encoded(evidence, raw))


def test_bounded_and_duplicate_field_json_is_rejected():
    evidence, raw = encoded(*fixture())
    for value in (b"{}"*(8*1024*1024+1), b'{"frames":[],"frames":[]}', b"[]", evidence.decode()):
        with pytest.raises(ValueError):screen_identity_windows(value, raw)


def test_read_only_output_is_detached_and_policy_cannot_be_modified_between_calls():
    inputs = encoded(*fixture())
    first = screen_identity_windows(*inputs)
    first["screen_policy"]["minimum_usable_samples"] = 0
    first["retained_inputs"]["raw_poses"]["frames"].clear()
    second = screen_identity_windows(*inputs)
    assert second["screen_policy"]["minimum_usable_samples"] == 8
    assert len(second["retained_inputs"]["raw_poses"]["frames"]) == 24


def test_aggregate_byte_limit_precedes_parsing(monkeypatch):
    import weft.pose_identity_windows as module
    inputs = encoded(*fixture())
    monkeypatch.setattr(module, "MAX_INPUT_BYTES", sum(map(len, inputs))-1)
    monkeypatch.setattr(module, "_document", lambda _: pytest.fail("aggregate bound must precede parsing"))
    with pytest.raises(ValueError, match="aggregate byte bound"):screen_identity_windows(*inputs)
