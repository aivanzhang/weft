"""Native clock observations preserve detector history and every decoded slot."""

from copy import deepcopy
from hashlib import sha256
import json

import pytest

from test_observations import exercise
from weft.io import canonical_hash
from weft.observations import adapt_exercise, validate_observations
from weft.source_clock import DETECTOR_CLOCK, PROFILE, derive_evidence


def native_pose(*, mismatch=True, missing_decoder=False):
    evidence = exercise()
    evidence.update(schema_version=1, reference=[{"time_s": 0, "evidence_id": evidence["frames"][0]["id"]}])
    evidence["frame_count"] = 6 if mismatch else 4
    if mismatch:
        evidence["container_frame_audit"] = {"encoded_video_frames": 4, "decoded_frames": 4, "matches": True}
    rows = [{"frame": frame, "pts": tick, "decoder_timestamp_s": tick / 1000 + .007}
            for frame, tick in enumerate([1000, 1033, 1100, 1133])]
    for sample in evidence["frames"]:
        sample["decoder_timestamp_s"] = None if missing_decoder else rows[sample["frame"]]["decoder_timestamp_s"]
    raw = (json.dumps(evidence, sort_keys=True, indent=2) + "\n").encode()
    binding = {"schema_version": 1, "profile": PROFILE, "source_sha256": evidence["source_sha256"],
        "evidence_sha256": sha256(raw).hexdigest(), "evidence_canonical_sha256": canonical_hash(evidence),
        "metadata_frames": evidence["frame_count"], "decoded_frames": 4, "nominal_fps": 30,
        "time_base": "1/1000", "native_origin_pts": 1000, "decoder_minus_native_s": .007,
        "native_to_reconstruction_offset_s": -1., "capture_clock_status": "unknown",
        "detector_clock_status": DETECTOR_CLOCK, "rows": rows}
    return evidence, derive_evidence(evidence, binding)


def test_native_observations_keep_complete_slots_history_and_distinct_counts():
    original, view = native_pose()
    before = deepcopy(view)
    document = adapt_exercise(view)
    assert view == before
    assert document["schema_version"] == 4
    source = document["source"]
    assert source["metadata_frames"] == 6 and source["decoded_frames"] == 4
    assert source["frame_count_audit"] == original["container_frame_audit"]
    assert source["frame_times_s"] == [0., .033, .1, .133]
    assert document["time"]["basis"] == "native_pts_elapsed_time"
    assert [row["time_s"] for row in document["observations"]] == [0., .033, .1, .133]
    assert [row["visibility"] for row in document["observations"]] == ["partial", "not_sampled", "not_detected", "not_sampled"]
    payload = document["evidence"][1]["payload"]
    assert payload["time_s"] == .1
    assert payload["source_time_derivation"]["original_pose_time_s"] == 2 / 30
    assert payload["decoder_timestamp_s"] == original["frames"][1]["decoder_timestamp_s"]
    assert document["provenance"]["source_metadata"]["source_clock"]["original_times"]["frames"] == [0, 2 / 30]
    assert source["native_clock_summary"]["native_origin_pts"] == 1000
    assert source["native_clock_summary"]["native_to_reconstruction_offset_s"] == -1.


def test_missing_sampled_decoder_history_is_not_backfilled():
    _, view = native_pose(missing_decoder=True)
    document = adapt_exercise(view)
    assert all(row["payload"]["decoder_timestamp_s"] is None for row in document["evidence"])
    assert all(row["decoder_timestamp_s"] is not None for row in document["source"]["native_clock"]["rows"])


@pytest.mark.parametrize("change", ["elapsed", "elapsed_bool", "metadata", "decoded", "audit", "summary", "clock", "payload_nominal", "payload_decoder", "missing_slot", "time_basis"])
def test_native_clock_custody_rejects_changed_mapping(change):
    _, view = native_pose()
    document = adapt_exercise(view)
    source = document["source"]
    if change == "elapsed": source["frame_times_s"][2] = 2 / 30
    elif change == "elapsed_bool": source["frame_times_s"][0] = False
    elif change == "metadata": source["metadata_frames"] = 4
    elif change == "decoded": source["decoded_frames"] = 3
    elif change == "audit": source["frame_count_audit"] = None
    elif change == "summary": source["native_clock_summary"]["binding_sha256"] = "f" * 64
    elif change == "clock": source["native_clock"]["rows"][2]["pts"] += 1
    elif change == "payload_nominal": document["evidence"][1]["payload"]["source_time_derivation"]["original_pose_time_s"] = .1
    elif change == "payload_decoder": document["evidence"][1]["payload"]["decoder_timestamp_s"] = .1
    elif change == "missing_slot": document["observations"].pop()
    else: document["time"]["basis"] = "frame_index_divided_by_fps"
    with pytest.raises(ValueError): validate_observations(document)


def test_derived_payload_tamper_cannot_enter_adapter():
    _, view = native_pose()
    view["frames"][0]["landmarks"][0]["x"] = .9
    with pytest.raises(ValueError, match="original evidence identity"):
        adapt_exercise(view)


def test_explicit_construction_selection_preserves_native_times_and_unsampled_slots():
    _, view = native_pose()
    document = adapt_exercise(view, construction_frames=[0])
    assert len(document["observations"]) == 4 and len(document["evidence"]) == 1
    assert document["observations"][2]["time_s"] == .1
    assert document["observations"][2]["visibility"] == "unknown"
    assert not document["observations"][2]["measurements"]
    slots = document["provenance"]["construction_selection"]["source_slots"]
    assert len(slots) == 6 and slots[-1]["source_status"] == "not_decoded"


def test_native_relationship_interval_uses_native_not_nominal_duration():
    _, view = native_pose()
    document = adapt_exercise(view, scene={"profile": "exercise-v1"})
    document["relationships"][0]["interval_s"] = [.1, .133]
    validate_observations(document)
    document["relationships"][0]["interval_s"] = [.1, .15]
    with pytest.raises(ValueError, match="relationship time"):
        validate_observations(document)


@pytest.mark.parametrize("metadata", [3, 5])
def test_one_frame_rounding_keeps_exact_receipt_without_inventing_encoded_audit(metadata):
    original, view = native_pose(mismatch=False)
    original["frame_count"] = metadata
    receipt = {"metadata_frames": metadata, "decoded_frames": 4, "difference": 4 - metadata,
               "note": "One-frame metadata rounding retained without an encoded-frame audit."}
    original["decode_metadata_discrepancy"] = receipt
    binding = deepcopy(view["source_clock"]["binding"])
    binding.update(metadata_frames=metadata, evidence_canonical_sha256=canonical_hash(original),
                   evidence_sha256=sha256(json.dumps(original).encode()).hexdigest())
    document = adapt_exercise(derive_evidence(original, binding))
    assert document["source"]["metadata_frames"] == metadata
    assert document["source"]["decoded_frames"] == 4
    assert document["source"]["frame_count_audit"] == receipt
    assert document["provenance"]["source_metadata"]["decode_metadata_discrepancy"] == receipt
    assert "encoded_video_frames" not in document["source"]["frame_count_audit"]
    for key, bad_value in (("difference", metadata - 4), ("decoded_frames", 3), ("note", ""), ("difference", True)):
        bad = deepcopy(document)
        bad["source"]["frame_count_audit"][key] = bad_value
        with pytest.raises(ValueError):
            validate_observations(bad)
