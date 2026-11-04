"""The supported-window proposal reproduces the frozen constraints, never relaxes them."""

import json

import pytest

from weft.portal_supported_window import (WINDOW_POLICY, propose_supported_window,
                                          supported_window_availability)


def evidence(statuses, *, fps=25.0, stride_s=0.08, review=None):
    frames = [{"decoder_timestamp_s": round(i * stride_s, 4), "status": status,
               "id": f"pin:frame:{i * 2}"} for i, status in enumerate(statuses)]
    value = {"fps": fps, "frames": frames, "quality": {}}
    if review is not None:
        value["quality"]["competing_hypotheses"] = review
    return value


def pattern(length=250):
    """Reproduce the live waiters-squat defect layout on a 20 s, 12.5 Hz sample grid."""
    statuses = ["tracked"] * length
    for t in (0.88,):
        statuses[int(t / 0.08)] = "multiple_people"
    for t in (19.52, 19.60):
        statuses[int(round(t / 0.08))] = "multiple_people"
    for t in (14.56, 14.64, 14.72):
        statuses[int(round(t / 0.08))] = "invalid_geometry"
    for t in (1.52, 1.68, 2.4, 3.28, 3.44, 4.48, 5.52, 6.0):
        statuses[int(round(t / 0.08))] = "low_confidence"
    return statuses


def unresolved_review():
    def gap(ids):
        return {"status": "unresolved", "evidence_ids": ids}
    return {"resolved": False, "gaps": [gap(["pin:frame:22"]), gap(["pin:frame:488", "pin:frame:490"])]}


def test_live_defect_pattern_yields_the_manually_found_window():
    result = propose_supported_window(evidence(pattern(), review=unresolved_review()))
    proposal = result["proposal"]
    assert proposal is not None
    # Settled start lands after the last scattered weak sample at 6.0 s and before the
    # dropout cluster near 14.6 s — the window three manual attempts converged on.
    assert 6.0 <= proposal["start_s"] <= 6.5
    assert 14.2 <= proposal["end_s"] <= 14.6
    assert proposal["settled_start"] is True
    kinds = {item["kind"] for item in result["excluded"]}
    assert kinds == {"unresolved_competing_hypotheses", "reference_gap"}
    assert result["policy"]["maximum_reference_gap_seconds"] == pytest.approx(0.25)


def test_clean_evidence_proposes_the_frame_bounded_prefix():
    result = propose_supported_window(evidence(["tracked"] * 300))
    proposal = result["proposal"]
    assert proposal is not None and proposal["settled_start"] is True
    assert proposal["start_s"] == 0.0
    assert proposal["seconds"] <= WINDOW_POLICY["maximum_window_frames"] / 25.0 + 1e-6
    assert result["excluded"] == []


def test_review_resolved_competing_samples_are_not_excluded():
    statuses = ["tracked"] * 150
    statuses[60] = "multiple_people"
    review = {"resolved": True, "gaps": [{"status": "resolved", "evidence_ids": ["pin:frame:120"]}]}
    result = propose_supported_window(evidence(statuses, review=review))
    assert result["excluded"] == []
    assert result["proposal"]["seconds"] >= 10


def test_pervasive_defects_refuse_with_reasons():
    statuses = ["tracked" if i % 3 == 0 else "invalid_geometry" for i in range(120)]
    result = propose_supported_window(evidence(statuses))
    assert result["proposal"] is None
    assert result["reasons"]


def test_unshaped_evidence_refuses():
    assert propose_supported_window({"fps": 25.0, "frames": "nonsense"})["proposal"] is None
    assert propose_supported_window({"fps": 0, "frames": []})["proposal"] is None


def test_availability_requires_a_recorded_proposal(tmp_path):
    state = supported_window_availability(tmp_path)
    assert state["available"] is False and "recorded window proposal" in state["reason"]
    folder = tmp_path / "perception/supported-window"
    folder.mkdir(parents=True)
    (folder / "proposal.json").write_text(json.dumps(
        {"proposal": None, "reasons": ["no span satisfies the frozen window constraints"]}))
    state = supported_window_availability(tmp_path)
    assert state["available"] is False
    assert state["reasons"] == ["no span satisfies the frozen window constraints"]
