from copy import deepcopy
from hashlib import sha256
import json

import numpy as np
import pytest

from weft.mask_track_hypotheses import associate_source_masks, track_mask_hypotheses


def slot(frame, name, x, *, status="available", pattern=None):
    return {"id": f"{frame}:{name}", "raw_object_id": name, "status": status,
            "box": [x, 2, x + 6, 8] if status == "available" else None,
            "mask": (np.ones((6, 6), np.uint8) if pattern is None else pattern) if status == "available" else None}


def frame(index, positions=(2, 20), *, statuses=("available", "available"), stamp=None):
    return {"frame": index, "time_s": index / 30 if stamp is None else stamp,
            "slots": [slot(index, name, x, status=status) for name, x, status in
                      zip(("a", "b"), positions, statuses, strict=True)]}


def run(frames):
    return track_mask_hypotheses(iter(frames), width=40, height=20)


def test_raw_id_swap_uses_image_continuity_without_rewriting_source():
    source = [frame(0), frame(1, (20, 2)), frame(2, (20, 3))]
    result = run(source)
    assert [r["anchor_id"] for r in result["frames"][1]["slots"]] == ["0:b", "0:a"]
    assert result["frames"][2]["slots"][1]["anchor_id"] == "0:a"
    assert source[1]["slots"][0]["raw_object_id"] == "a"
    assert not result["raw_identity_reclassified"] and not result["physical_identity_verified"]
    assert result["contact"] == "unknown" and not result["source_custody_verified"]


def test_identical_competitors_abstain_and_never_recover_old_anchor():
    result = run([frame(0), frame(1, (2, 2)), frame(2)])
    assert all(row["anchor_id"] is None for item in result["frames"][1:] for row in item["slots"])
    assert len(result["frames"][1]["pairs"]) == 4
    assert not any(p["accepted"] for p in result["frames"][1]["pairs"])


@pytest.mark.parametrize("status", ["missing", "withheld"])
def test_gap_keeps_all_slots_and_does_not_bridge(status):
    result = run([frame(0), frame(1, statuses=(status, "available")), frame(2)])
    assert len(result["frames"]) == 3
    assert result["frames"][1]["slots"][0]["status"] == status
    assert result["frames"][2]["slots"][0]["anchor_id"] is None
    assert result["frames"][2]["slots"][1]["anchor_id"] == "0:b"


def test_preinitialization_is_distinct_from_recovery():
    result = run([frame(0, statuses=("pre_initialization", "available")), frame(1)])
    assert result["frames"][1]["slots"][0]["status"] == "declared_initialization"
    assert result["frames"][1]["slots"][0]["anchor_id"] == "1:a"


def test_time_gap_drops_anchors_not_observations():
    result = run([frame(0), frame(1, stamp=.4), frame(2, stamp=.43)])
    assert all(row["status"] == "time_gap" for row in result["frames"][1]["slots"])
    assert all(row["anchor_id"] is None for row in result["frames"][2]["slots"])


def test_overlap_counts_exact_cells_without_canvas_or_rescaling():
    result = run([frame(0), frame(1, (3, 20))])
    pair = result["frames"][1]["pairs"][0]
    assert pair["intersection_pixels"] == 30 and pair["union_pixels"] == 42
    assert pair["iou"] == 30 / 42 and pair["accepted"]


def test_disjoint_masks_in_same_box_are_not_a_match():
    a, b = frame(0), frame(1)
    a["slots"][0]["mask"][:, 3:] = 0
    b["slots"][0]["mask"][:, :3] = 0
    result = run([a, b])
    assert result["frames"][1]["pairs"][0]["iou"] == 0
    assert result["frames"][1]["slots"][0]["anchor_id"] is None


def test_empty_mask_is_not_initialized():
    a = frame(0)
    a["slots"][0]["mask"][:] = 0
    result = run([a, frame(1)])
    assert result["frames"][0]["slots"][0]["status"] == "empty_mask"
    assert result["frames"][1]["slots"][0]["anchor_id"] is None


def test_releasing_input_arrays_does_not_mutate_previous_frame():
    a = frame(0)
    def generator():
        yield a
        a["slots"][0]["mask"][:] = 0
        yield frame(1)
    result = run(generator())
    assert result["frames"][1]["slots"][0]["anchor_id"] == "0:a"


@pytest.mark.parametrize("change", [
    lambda rows: rows[1].update(frame=3),
    lambda rows: rows[1].update(time_s=0),
    lambda rows: rows[1].update(time_s=float("nan")),
    lambda rows: rows[1]["slots"].pop(),
    lambda rows: rows[1]["slots"][0].update(id="0:a"),
    lambda rows: rows[1]["slots"][0].update(raw_object_id="b"),
    lambda rows: rows[1]["slots"][0].update(box=[2.1, 2, 8.1, 8]),
    lambda rows: rows[1]["slots"][0].update(mask=np.full((6, 6), 255, np.uint8)),
    lambda rows: rows[1]["slots"][0].update(mask=np.ones((5, 6), np.uint8)),
    lambda rows: rows[1]["slots"][0].update(status="withheld"),
])
def test_invalid_source_slots_rejected(change):
    rows = [frame(0), frame(1)]
    change(rows)
    with pytest.raises(ValueError):
        run(rows)


def test_deadline_and_frame_bounds():
    with pytest.raises(TimeoutError):
        track_mask_hypotheses([frame(0)], width=40, height=20, maximum_seconds=1e-12)
    with pytest.raises(ValueError, match="frame bound"):
        run(frame(i) for i in range(501))


def test_contract_result_is_defensive():
    result = run([frame(0)])
    result["contract"]["minimum_iou"] = 0
    assert run([frame(0)])["contract"]["minimum_iou"] == .5


def source_fixture():
    from test_silhouette_geometry import fixture
    _, inputs = fixture()
    source = next(iter(inputs["source_documents"].values()))
    masks = json.loads(inputs["mask_document"])
    masks["contract"] = {"mapping": "Raster cell edges span the original half-open model box; centroid uses cell centers",
                         "method": "saved-sam-mask-raster-v1"}
    rasters = {}
    for row in masks["observations"]:
        if row["mask_status"] != "decoded":
            continue
        x0, y0, x1, y1 = row["box"]["box_xyxy_halfopen"]
        row["raster_dimensions"] = [y1 - y0, x1 - x0]
        raw = bytes([1]) * ((y1 - y0) * (x1 - x0))
        row["raster_sha256"] = sha256(raw).hexdigest()
        rasters[row["id"]] = raw
    return source, masks, rasters


def encode(document):
    return json.dumps(document, sort_keys=True).encode()


def test_source_adapter_preserves_complete_slots_and_submitted_clock():
    source, masks, rasters = source_fixture()
    result = associate_source_masks(source, encode(masks), raster_loader=lambda row: rasters[row["id"]])
    assert result["source_observation_binding_verified"] and len(result["frames"]) == 3
    assert not result["source_custody_verified"] and not result["encoded_to_raster_derivation_verified"]
    assert len(result["loaded_rasters"]) == 6
    assert len(result["frames"][0]["slots"]) == 3
    assert all(row["status"] == "pre_initialization" for row in result["frames"][0]["slots"])
    assert not result["raw_identity_reclassified"]


def test_withheld_coordinates_never_reach_raster_callback():
    source, masks, rasters = source_fixture()
    document = json.loads(source)
    for row in document["observations"]:
        if row["frame"] == 2:
            row["partition"] = "audit"
    calls = []
    def load(row):
        calls.append(row["frame"])
        return rasters[row["id"]]
    result = associate_source_masks(encode(document), encode(masks), raster_loader=load)
    assert calls == [1, 1, 1] and len(result["withheld_slots"]) == 3
    assert all(row["box"] is None for row in result["frames"][2]["slots"])


@pytest.mark.parametrize("field,value", [("time_s", .123), ("raw_observation_sha256", "0"*64),
                                          ("identity", {"status": "resolved"}), ("encoded_sha256", "0"*64)])
def test_mask_custody_tampering_rejected_before_pixels(field, value):
    source, masks, _ = source_fixture()
    masks["observations"][3][field] = value
    calls = []
    with pytest.raises(ValueError):
        associate_source_masks(source, encode(masks), raster_loader=lambda row: calls.append(row))
    assert not calls


def test_binary_tampering_rejected():
    source, masks, rasters = source_fixture()
    with pytest.raises(ValueError, match="exact original binary"):
        associate_source_masks(source, encode(masks), raster_loader=lambda row: b"\0" * len(rasters[row["id"]]))


@pytest.mark.parametrize("field,value", [("mask_status", "INVALID"), ("phase", "invented"), ("proposal_status", "invented")])
def test_unknown_status_never_silently_becomes_missing(field, value):
    source, masks, _ = source_fixture()
    masks["observations"][3][field] = value
    with pytest.raises(ValueError, match="availability/phase"):
        associate_source_masks(source, encode(masks), raster_loader=lambda row: pytest.fail("No pixels expected"))


def test_postpreparation_deadline_is_enforced(monkeypatch):
    import weft.mask_track_hypotheses as module
    stamps = iter([0, 0, 20])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(stamps))
    with pytest.raises(TimeoutError):
        run([frame(0)])


@pytest.mark.parametrize("raw", [b'{"profile":1,"profile":2}', b'{"x":NaN}', b'{"x":1e999}', b'[]'])
def test_strict_mask_document(raw):
    source, _, _ = source_fixture()
    with pytest.raises(ValueError):
        associate_source_masks(source, raw, raster_loader=lambda row: pytest.fail("No pixels expected"))


def test_hash_agreement_is_not_encoded_pixel_verification():
    source, masks, rasters = source_fixture()
    for row in masks["observations"]:
        if row["id"] in rasters:
            rasters[row["id"]] = b"\0" * len(rasters[row["id"]])
            row["raster_sha256"] = sha256(rasters[row["id"]]).hexdigest()
    result = associate_source_masks(source, encode(masks), raster_loader=lambda row: rasters[row["id"]])
    assert result["decoded_raster_hashes_verified"]
    assert not result["encoded_to_raster_derivation_verified"] and not result["source_custody_verified"]


def test_aggregate_pixels_reserved_before_loading(monkeypatch):
    import weft.mask_track_hypotheses as module
    source, masks, _ = source_fixture()
    monkeypatch.setitem(module.CONTRACT, "maximum_total_pixels", 1)
    with pytest.raises(ValueError, match="aggregate raster reservation"):
        associate_source_masks(source, encode(masks), raster_loader=lambda row: pytest.fail("No pixels expected"))
