"""Analytic controls only; no real-source labels or trajectories inform these tests."""

import time

import numpy as np
import pytest

from weft.joint_tracking import JOINT_CONTRACT, JointAppearance


SETTINGS = {"search_radius_px": 60, "minimum_score": .7, "minimum_margin": .06, "max_gap_frames": 3}


def patch(seed=4, size=12):
    return np.random.default_rng(seed).integers(30, 240, (size, size), dtype=np.uint8)


def frame(items=(), shape=(100, 180)):
    gray = np.full(shape, 10, dtype=np.uint8)
    for x, y, texture in items:
        h, w = texture.shape
        gray[y:y+h, x:x+w] = texture
    return gray


def colored(items=(), shape=(100, 180)):
    gray = frame([(x, y, p) for x, y, p, channel in items], shape)
    bgr = np.repeat(np.full(shape, 10, dtype=np.uint8)[..., None], 3, axis=2)
    for x, y, p, channel in items:
        h, w = p.shape
        bgr[y:y+h, x:x+w] = 0
        bgr[y:y+h, x:x+w, channel] = p
    return gray, bgr


def step(tracker, image, color=None):
    result = tracker.locate(image, color=color)
    tracker.commit(result)
    return result


def test_grayscale_translation_and_immutable_templates():
    p = patch()
    image = frame([(30, 30, p)])
    tracker = JointAppearance(image, [[30, 30, 12, 12]], SETTINGS)
    signature = tracker.source_signatures
    image[:] = 0
    p[:] = 0
    for x in (33, 37, 43, 40, 34):
        result = step(tracker, frame([(x, 30, patch())]))[0]
        assert result["status"] == "tracked"
        assert result["box"] == [x, 30, 12, 12]
        assert result["score"] >= .99
    assert tracker.source_signatures == signature
    assert isinstance(signature, tuple)
    with pytest.raises(ValueError):
        tracker._tracks[0]["template"].setflags(write=True)
    with pytest.raises(AttributeError):
        tracker.source_signatures = ("forged",)


def test_distinct_color_separates_identical_grayscale_near_contact():
    p = patch()
    gray, color = colored([(30, 30, p, 2), (48, 30, p, 0)])
    tracker = JointAppearance(gray, [[30, 30, 12, 12], [48, 30, 12, 12]], SETTINGS, color=color)
    gray, color = colored([(44, 30, p, 2), (30, 30, p, 0)])
    results = step(tracker, gray, color)
    assert [r["box"] for r in results] == [[44, 30, 12, 12], [30, 30, 12, 12]]
    assert all(r["appearance_evidence"]["passes"] for r in results)


def test_same_appearance_contact_and_crossing_abstain():
    p = patch()
    tracker = JointAppearance(frame([(30, 30, p), (48, 30, p)]),
                              [[30, 30, 12, 12], [48, 30, 12, 12]], SETTINGS)
    for items in ([(34, 30, p), (48, 30, p)], [(42, 30, p)], [(30, 30, p), (48, 30, p)]):
        results = step(tracker, frame(items))
        assert all(r["box"] is None for r in results)
        assert all(r["status"] in {"ambiguous", "ambiguous_overlap"} for r in results)
        assert all(r["association_evidence"]["temporal_disabled_contact"] for r in results)


def test_joint_temporal_association_resolves_distant_identical_peaks():
    p = patch(size=24)
    tracker = JointAppearance(frame([(30, 30, p), (68, 30, p)]),
                              [[30, 30, 24, 24], [68, 30, 24, 24]], SETTINGS)
    # These are still contact-scale and must not be resolved by a motion prior.
    results = step(tracker, frame([(32, 30, p), (66, 30, p)]))
    assert all(r["box"] is None for r in results)
    p = patch()
    tracker = JointAppearance(frame([(30, 30, p), (60, 30, p)]),
                              [[30, 30, 12, 12], [60, 30, 12, 12]], SETTINGS)
    competing_peaks_seen = False
    for locations in ((40, 70), (50, 80), (60, 90)):
        results = step(tracker, frame([(x, 30, p) for x in locations]))
        assert [r["box"][0] for r in results] == list(locations)
        competing_peaks_seen |= any(r["association_evidence"]["candidate_count"] > 1 for r in results)
    assert competing_peaks_seen


def test_abrupt_reversal_retains_measurement_not_prediction():
    p, q = patch(1), patch(2)
    tracker = JointAppearance(frame([(30, 30, p), (80, 30, q)]),
                              [[30, 30, 12, 12], [80, 30, 12, 12]], SETTINGS)
    for x, y in ((42, 68), (52, 64), (36, 80), (20, 90)):
        results = step(tracker, frame([(x, 30, p), (y, 30, q)]))
        assert [r["box"] for r in results] == [[x, 30, 12, 12], [y, 30, 12, 12]]


def test_short_occlusion_recovery_and_permanent_loss():
    p = patch()
    tracker = JointAppearance(frame([(30, 30, p)]), [[30, 30, 12, 12]], SETTINGS)
    assert step(tracker, frame())[0]["box"] is None
    assert step(tracker, frame())[0]["box"] is None
    assert step(tracker, frame([(40, 30, p)]))[0]["box"] == [40, 30, 12, 12]
    for _ in range(4):
        assert step(tracker, frame())[0]["box"] is None
    result = step(tracker, frame([(40, 30, p)]))[0]
    assert result["status"] == "lost"
    assert result["box"] is None


def test_no_duplicate_assignment_when_only_one_object_visible():
    p = patch()
    tracker = JointAppearance(frame([(30, 30, p), (48, 30, p)]),
                              [[30, 30, 12, 12], [48, 30, 12, 12]], SETTINGS)
    results = step(tracker, frame([(40, 30, p)]))
    accepted = [r["box"] for r in results if r["box"] is not None]
    assert len(accepted) <= 1
    assert accepted == []


def test_wrong_color_never_rewrites_identity():
    p = patch()
    gray, color = colored([(30, 30, p, 2)])
    tracker = JointAppearance(gray, [[30, 30, 12, 12]], SETTINGS, color=color)
    signatures = tracker.source_signatures
    gray, color = colored([(34, 30, p, 0)])
    result = step(tracker, gray, color)[0]
    assert result["status"] == "appearance_changed"
    assert result["box"] is None
    gray, color = colored([(38, 30, p, 2)])
    assert step(tracker, gray, color)[0]["box"] == [38, 30, 12, 12]
    assert tracker.source_signatures == signatures


def test_predictions_and_out_of_search_objects_are_never_published():
    p = patch()
    tracker = JointAppearance(frame([(30, 30, p)]), [[30, 30, 12, 12]], SETTINGS)
    assert step(tracker, frame([(40, 30, p)]))[0]["box"] is not None
    result = step(tracker, frame())[0]
    assert result["box"] is None
    assert result["score"] < .7
    assert step(tracker, frame([(150, 30, p)]))[0]["box"] is None


def test_low_configured_score_never_lowers_visual_floor():
    p = patch()
    tracker = JointAppearance(frame([(30, 30, p)]), [[30, 30, 12, 12]], dict(SETTINGS, minimum_score=.5))
    assert tracker._minimum_score == .7
    mixed = np.clip(.55 * p + .85 * patch(99), 0, 255).astype(np.uint8)
    result = step(tracker, frame([(34, 30, mixed)]))[0]
    assert result["score"] < .7
    assert result["box"] is None


def test_locate_commit_contract_rejects_invented_or_reused_results():
    p = patch()
    tracker = JointAppearance(frame([(30, 30, p)]), [[30, 30, 12, 12]], SETTINGS)
    result = tracker.locate(frame([(33, 30, p)]))
    with pytest.raises(ValueError):
        tracker.locate(frame([(35, 30, p)]))
    result[0]["box"][0] = 90
    with pytest.raises(ValueError):
        tracker.commit(result)
    result[0]["box"][0] = 33
    tracker.commit(result)
    with pytest.raises(ValueError):
        tracker.commit(result)


def test_search_and_assignment_work_are_bounded_for_eight_objects():
    positions = [(30 + 100 * (i % 4), 30 + 100 * (i // 4), patch(i)) for i in range(8)]
    tracker = JointAppearance(frame(positions, (250, 450)),
                              [[x, y, 12, 12] for x, y, p in positions], SETTINGS)
    timings = []
    for k in range(50):
        shifted = [(x + k % 3, y, p) for x, y, p in positions]
        started = time.perf_counter()
        results = step(tracker, frame(shifted, (250, 450)))
        timings.append(time.perf_counter() - started)
        assert all(r["status"] == "tracked" for r in results)
        for result in results:
            audit = result["association_evidence"]
            assert audit["radius_px"] <= SETTINGS["search_radius_px"]
            assert audit["search_pixels"] <= (12 + 2 * 60) ** 2
            assert audit["candidate_count"] <= JOINT_CONTRACT["maximum_peaks"]
            assert audit["assignment_nodes"] <= JOINT_CONTRACT["maximum_assignment_nodes"]
    # Structural bounds are assertions; latency is descriptive, not a machine-specific gate.
    print(f"eight-object analytic ms: median={1000*np.median(timings):.3f}, p95={1000*np.quantile(timings,.95):.3f}")


def test_repeated_texture_peak_overflow_abstains():
    p = patch(size=6)
    items = [(x, y, p) for x in (30, 38, 46) for y in (30, 38, 46)]
    tracker = JointAppearance(frame([(38, 38, p)]), [[38, 38, 6, 6]], SETTINGS)
    result = step(tracker, frame(items))[0]
    assert result["status"] == "proposal_limit"
    assert result["box"] is None


@pytest.mark.parametrize("mutation", ["oversize", "dtype", "count", "flat", "overlap"])
def test_invalid_inputs_fail_closed(mutation):
    p = patch()
    image, boxes = frame([(30, 30, p)]), [[30, 30, 12, 12]]
    if mutation == "oversize":
        image = np.zeros((961, 100), np.uint8)
    elif mutation == "dtype":
        image = image.astype(float)
    elif mutation == "count":
        boxes *= 9
    elif mutation == "flat":
        image[:] = 0
    elif mutation == "overlap":
        boxes *= 2
    with pytest.raises(ValueError):
        JointAppearance(image, boxes, SETTINGS)


def test_assignment_budget_exhaustion_abstains(monkeypatch):
    import weft.joint_tracking as tracking
    monkeypatch.setattr(tracking, "JOINT_CONTRACT", dict(JOINT_CONTRACT, maximum_assignment_nodes=1))
    p = patch()
    tracker = JointAppearance(frame([(30, 30, p)]), [[30, 30, 12, 12]], SETTINGS)
    result = step(tracker, frame([(32, 30, p)]))[0]
    assert result["status"] == "association_limit"
    assert result["box"] is None
    assert result["association_evidence"]["assignment_nodes"] == 1
    assert result["association_evidence"]["assignment_complete"] is False


def test_rejected_hypotheses_and_score_meanings_are_explicit():
    p = patch()
    tracker = JointAppearance(frame([(30, 30, p), (48, 30, p)]),
                              [[30, 30, 12, 12], [48, 30, 12, 12]], SETTINGS)
    results = step(tracker, frame([(32, 30, p), (46, 30, p)]))
    for result in results:
        audit = result["association_evidence"]
        assert audit["candidate_hypotheses"]
        assert len(audit["candidate_hypotheses"]) <= 6
        assert all(h["status"] == "rejected_association" for h in audit["candidate_hypotheses"])
        assert all(h["visual_score"] >= .7 for h in audit["candidate_hypotheses"])
        assert audit["assignment_score"] > result["score"]
        assert audit["association_margin"] < SETTINGS["minimum_margin"]
        assert result["box"] is None


def test_repeated_and_jittered_frames_do_not_shrink_observed_hop_search():
    p = patch()
    tracker = JointAppearance(frame([(30, 30, p)]), [[30, 30, 12, 12]], SETTINGS)
    assert step(tracker, frame([(42, 30, p)]))[0]["box"] == [42, 30, 12, 12]
    for x in (42, 42, 43, 42, 42):
        result = step(tracker, frame([(x, 30, p)]))[0]
        audit = result["association_evidence"]
        assert result["box"] == [x, 30, 12, 12]
        assert audit["observed_hop_max_px"] == 12
        assert audit["radius_px"] == 30
    # A 28px reversal was excluded by the former post-repeat 18px local window.
    result = step(tracker, frame([(14, 30, p)]))[0]
    assert result["status"] == "tracked"
    assert result["box"] == [14, 30, 12, 12]
    assert result["association_evidence"]["best_visual_hypothesis"]["visual_score"] >= .99


def test_repeated_frames_before_abrupt_distinct_appearance_crossing():
    p, q = patch(5), patch(6)
    tracker = JointAppearance(frame([(30, 30, p), (90, 30, q)]),
                              [[30, 30, 12, 12], [90, 30, 12, 12]], SETTINGS)
    for _ in range(4):
        results = step(tracker, frame([(42, 30, p), (78, 30, q)]))
        assert all(row["status"] == "tracked" for row in results)
    results = step(tracker, frame([(68, 30, p), (52, 30, q)]))
    assert [row["box"] for row in results] == [[68, 30, 12, 12], [52, 30, 12, 12]]
    assert all(row["association_evidence"]["radius_px"] == 30 for row in results)


def test_confusable_alternative_outside_former_local_window_is_audited():
    p = patch()
    tracker = JointAppearance(frame([(30, 30, p), (70, 30, p)]),
                              [[30, 30, 12, 12], [70, 30, 12, 12]], SETTINGS)
    results = step(tracker, frame([(30, 30, p), (54, 30, p)]))
    assert all(row["box"] is None for row in results)
    for row in results:
        audit = row["association_evidence"]
        assert audit["search_policy"] == "full_configured_confusable_radius"
        assert audit["radius_px"] == 60
        assert audit["candidate_count"] == 2
        assert audit["proposal_status"] == "candidates_ready"
        assert audit["final_status"] == row["status"] == "ambiguous"
        assert "status" not in audit
    assert {tuple(row["box"]) for row in results[0]["association_evidence"]["candidate_hypotheses"]} == {
        (30, 30, 12, 12), (54, 30, 12, 12)}


def test_belowthreshold_best_location_and_exact_clipped_search_rect_are_hypotheses_only():
    p = patch()
    tracker = JointAppearance(frame([(2, 3, p)]), [[2, 3, 12, 12]], SETTINGS)
    result = step(tracker, frame())[0]
    audit = result["association_evidence"]
    assert result["box"] is None
    assert audit["proposal_status"] == audit["final_status"] == "not_detected"
    assert audit["search_rect_xywh"] == [0, 0, 32, 33]
    assert audit["search_pixels"] == 32 * 33
    hypothesis = audit["best_visual_hypothesis"]
    assert hypothesis["box"] == [0, 0, 12, 12]
    assert hypothesis["visual_gate_passes"] is False
    assert hypothesis["status"] == "below_visual_gate"
    assert hypothesis["coordinate_scope"] == "hypothesis_only_not_an_observed_position"
    assert audit["candidate_hypotheses"] == []
    assert tracker._tracks[0]["box"] == (2, 3, 12, 12)
    assert tracker._tracks[0]["observed_hop_max_px"] == 0


def test_dark_light_boundary_does_not_turn_a_rejected_peak_into_a_position():
    p = patch(44)
    source = frame([(45, 30, p)])
    tracker = JointAppearance(source, [[45, 30, 12, 12]], SETTINGS)
    boundary = np.full(source.shape, 10, dtype=np.uint8)
    boundary[:, 60:] = 230
    rejected = step(tracker, boundary)[0]
    assert rejected["box"] is None
    assert rejected["score"] < .7
    audit = rejected["association_evidence"]
    assert audit["best_visual_hypothesis"]["visual_gate_passes"] is False
    assert audit["search_rect_xywh"] == [27, 12, 48, 48]
    # An exact observed patch across the boundary may recover, but the edge itself cannot.
    boundary[30:42, 60:72] = p
    recovered = step(tracker, boundary)[0]
    assert recovered["box"] == [60, 30, 12, 12]
    assert recovered["score"] >= .99


def test_expanded_confusable_search_keeps_all_hard_resource_bounds():
    p = patch()
    positions = [(30 + i*100, 30, p) for i in range(8)]
    source = frame(positions, shape=(100, 900))
    tracker = JointAppearance(source, [[x, y, 12, 12] for x, y, _ in positions], SETTINGS)
    timings = []
    for _ in range(30):
        started = time.perf_counter()
        results = step(tracker, source)
        timings.append(time.perf_counter()-started)
        assert all(row["status"] == "tracked" for row in results)
        for row in results:
            audit = row["association_evidence"]
            assert audit["radius_px"] == 60
            assert audit["search_pixels"] <= (12+2*60)**2
            assert audit["candidate_count"] <= 6
            assert audit["assignment_nodes"] <= 20000
    print(f"eight-confusable-object analytic ms: median={1000*np.median(timings):.3f}, p95={1000*np.quantile(timings,.95):.3f}")


def test_v2_contract_is_explicit_without_relaxing_acceptance_or_loss():
    assert JOINT_CONTRACT["method"] == "experimental-joint-immutable-roi-v2"
    assert JOINT_CONTRACT["minimum_visual_score"] == .7
    assert JOINT_CONTRACT["maximum_peaks"] == 6
    assert JOINT_CONTRACT["maximum_assignment_nodes"] == 20000
    p = patch()
    tracker = JointAppearance(frame([(30, 30, p)]), [[30, 30, 12, 12]], SETTINGS)
    for _ in range(SETTINGS["max_gap_frames"] + 1):
        result = step(tracker, frame())[0]
        assert result["box"] is None
    result = step(tracker, frame([(30, 30, p)]))[0]
    assert result["status"] == "lost" and result["box"] is None
    assert result["association_evidence"]["search_rect_xywh"] is None
    assert result["association_evidence"]["best_visual_hypothesis"] is None
