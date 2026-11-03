"""Unknown axial orientation may change; observed anchors and source gaps may not."""

from copy import deepcopy
import json
import math

import numpy as np
import pytest

from weft.io import canonical_hash
from weft.spatial_pose import LIMB_ENDPOINTS, build_spatial_reference, rotation_zyx
from weft import spatial_twist_continuity as continuity
from test_spatial_pose import pose_frame, anchor


def inputs():
    frames = [pose_frame(i, limb=rotation_zyx([2.6, 0., 0.]), flexion=bend)
              for i, bend in enumerate((.05, .01, .05))]
    for i, frame in enumerate(frames): frame["id"] = "a"*64+f":frame:{i}"
    refs, audit = build_spatial_reference({"source_sha256": "a"*64, "frames": frames}, anchor())
    return refs, audit["frames"], [frame["id"] for frame in frames]


def propose(refs, audit, slots, **options):
    return continuity.propose_twist_continuity(refs, audit, slots, source_sha256="a"*64,
        parent_human_sha256="b"*64, anchor_rotation=np.eye(3).tolist(), **options)


def test_transport_removes_unknown_frame_reset_without_promoting_confidence():
    refs, audit, slots = inputs(); before = deepcopy((refs, audit, slots))
    changed, report = propose(refs, audit, slots)
    assert (refs, audit, slots) == before and report["accepted_spans"] == 4
    assert changed[0] == refs[0] and changed[-1] == refs[-1]
    assert report["original_reference_sha256"] == canonical_hash(refs)
    for limb in LIMB_ENDPOINTS:
        names = [limb+suffix for suffix in ("_twist", "_lateral", "")]
        old, new = (rotation_zyx([row["joints"][n] for n in names]) for row in (refs[1], changed[1]))
        np.testing.assert_allclose(old[:, 2], new[:, 2], atol=1e-12)
        assert audit[1]["joint_confidence"][limb+"_twist"] == 0
        assert changed[1]["joints"][limb+"_twist"] == pytest.approx(2.6)
        distal = limb.rsplit("_", 1)[0]+("_knee" if limb.endswith("hip") else "_elbow")
        assert changed[1]["joints"][distal] == refs[1]["joints"][distal]
    assert report["confidence_promoted"] is False
    assert all(row["after_peak_chart_speed_rad_s"] < 1e-10 for row in report["trials"])


@pytest.mark.parametrize("damage,reason", [("gap", "missing_source_pose"), ("duration", "anchor_span"),
    ("identity", "identity_status"), ("anchor", "anchors_lack"), ("clamped", "model_hinge")])
def test_unsupported_spans_preserve_complete_original(damage, reason):
    refs, audit, slots = inputs()
    if damage == "gap": slots.insert(1, "a"*64+":frame:99")
    if damage == "duration": refs[-1]["time_s"] = audit[-1]["time_s"] = .3
    if damage == "identity": audit[1]["identity_status"] = "different_identity"
    if damage == "anchor":
        for limb in LIMB_ENDPOINTS: audit[0]["joint_confidence"][limb+"_twist"] = 0
    if damage == "clamped":
        for name in refs[1]["joints"]:
            if name.endswith(("_knee", "_elbow")): refs[1]["joints"][name] = 2.6
    changed, report = propose(refs, audit, slots)
    assert changed == refs and report["accepted_spans"] == 0
    assert all(reason in row["reason"] for row in report["trials"])


def test_unknown_prefix_and_suffix_do_not_change_initial_floor_or_final_pose():
    refs, audit, slots = inputs()
    for row in audit:
        row["assumed_twist"] = list(LIMB_ENDPOINTS)
        for limb in LIMB_ENDPOINTS: row["joint_confidence"][limb+"_twist"] = 0.
    changed, report = propose(refs, audit, slots)
    assert changed == refs and report["accepted_spans"] == 0
    assert all(row["reason"] == "two_observable_anchors_required" for row in report["trials"])


def test_already_continuous_unknown_frames_are_not_new_observations():
    refs, audit, slots = inputs()
    for limb in LIMB_ENDPOINTS: refs[1]["joints"][limb+"_twist"] = 2.6
    changed, report = propose(refs, audit, slots)
    assert changed == refs and report["accepted_spans"] == 0


@pytest.mark.parametrize("damage", ["source", "time", "confidence", "shape", "nonfinite", "huge", "deadline"])
def test_invalid_inputs_fail_before_proposal(damage):
    refs, audit, slots = inputs(); options = {}
    if damage == "source": slots[0] = "other-source"
    if damage == "time": audit[0]["time_s"] = .01
    if damage == "confidence": audit[1]["joint_confidence"]["left_shoulder_twist"] = True
    if damage == "shape": refs[0]["joints"]["extra"] = 0
    if damage == "nonfinite": refs[0]["joints"]["torso_twist"] = math.nan
    if damage == "huge": refs *= 601
    if damage == "deadline": options["max_seconds"] = 11
    with pytest.raises(ValueError): propose(refs, audit, slots, **options)


def test_antipodal_axis_transport_refuses_instead_of_choosing_a_hidden_rotation():
    with pytest.raises(ValueError, match="antipodal"):
        continuity._transport(np.eye(3), np.array([0., 0., -1.]))


def test_deadline_is_checked_during_reference_work(monkeypatch):
    refs, audit, slots = inputs(); ticks = iter([0., 2.])
    monkeypatch.setattr(continuity.time, "monotonic", lambda: next(ticks))
    with pytest.raises(TimeoutError): propose(refs, audit, slots, max_seconds=1)


@pytest.mark.parametrize("damage", ["assumed_size", "assumed_type", "assumed_duplicate", "confidence_bool", "confidence_high", "metadata_deep", "metadata_string", "huge_integer"])
def test_audit_input_bounds_precede_transport(damage, monkeypatch):
    refs, audit, slots = inputs()
    if damage == "assumed_size": audit[1]["assumed_twist"] *= 2
    if damage == "assumed_type": audit[1]["assumed_twist"] = [{}]
    if damage == "assumed_duplicate": audit[1]["assumed_twist"] = ["left_shoulder"]*2
    if damage == "confidence_bool": audit[0]["joint_confidence"]["left_shoulder_twist"] = True
    if damage == "confidence_high": audit[0]["joint_confidence"]["left_shoulder_twist"] = 2
    if damage == "metadata_string": audit[0]["extra"] = "x"*4097
    if damage == "huge_integer": audit[0]["extra"] = 10**400
    if damage == "metadata_deep":
        nested = []
        for _ in range(20): nested = [nested]
        audit[0]["extra"] = nested
    monkeypatch.setattr(continuity, "_transport", lambda *_: pytest.fail("Invalid input reached transport"))
    with pytest.raises(ValueError): propose(refs, audit, slots)


@pytest.mark.parametrize("step", [1e-310, 1e-160])
def test_tiny_native_durations_refuse_nonfinite_cost_without_retiming(step):
    refs, audit, slots = inputs()
    for i, (reference, row) in enumerate(zip(refs, audit)):
        reference["time_s"] = row["time_s"] = i*step
    before = deepcopy((refs, audit, slots))
    result, receipt = propose(refs, audit, slots)
    assert result == refs and (refs, audit, slots) == before
    assert receipt["accepted_spans"] == 0
    assert all("nonfinite chart-speed cost" in row["reason"] for row in receipt["trials"])
    json.dumps(receipt, allow_nan=False)


@pytest.mark.parametrize("identity", [[], {}, 1, True, None])
def test_evidence_id_type_rejected_before_membership(identity, monkeypatch):
    refs, audit, slots = inputs(); audit[0]["evidence_id"] = identity
    monkeypatch.setattr(continuity, "_transport", lambda *_: pytest.fail("Invalid identity reached transport"))
    with pytest.raises(ValueError, match="identity/time"):
        propose(refs, audit, slots)
