"""Temporal shape priors preserve positive pixels and never manufacture contact."""

from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import runpy

import numpy as np
from PIL import Image
import pytest

from weft import equipment_temporal_silhouette as module
from weft import silhouette_geometry
from weft.world_evidence import _freeze, _thaw
from test_silhouette_geometry import OTHERS, TARGET, fixture, png, raw, repin


PRIOR = {"mode": "fixed_camera_static_rigid", "provenance": "Authored test prior; not measured stationarity."}


def two_frames(*, first_occluded=True, second_occluded=False, shift=0, holes=False):
    scene, kwargs = fixture(occlusion=first_occluded, holes=holes)
    seg, masks = json.loads(kwargs["segmentation_evidence"]), json.loads(kwargs["mask_document"])
    seg["time"]["timestamps_s"][2] = masks["frames"][2]["time_s"] = .333
    for row in seg["observations"]:
        if row["frame"] != 2:
            continue
        row["time_s"] = .333
        if row["object_id"] == OTHERS[0]:
            bounds = [360, 350, 410, 420] if second_occluded else [10, 10, 30, 30]
            row["geometry"].update(box_xyxy_halfopen=bounds,
                box_xywh=[bounds[0], bounds[1], bounds[2] - bounds[0], bounds[3] - bounds[1]],
                center_px=[(bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2])
        if row["object_id"] == TARGET and shift:
            geometry = row["geometry"]
            geometry["box_xyxy_halfopen"][0] += shift
            geometry["box_xyxy_halfopen"][2] += shift
            geometry["box_xywh"][0] += shift
            geometry["center_px"][0] += shift
    for row in masks["observations"]:
        if row["frame"] == 2:
            kwargs["raster_documents"][row["raster_path"]] = kwargs["raster_documents"][row["raster_path"].replace("2-", "1-")]
    return repin(scene, kwargs, seg, masks)


def context(**kwargs):
    scene, inputs = two_frames(**kwargs)
    return silhouette_geometry.prepare_silhouette_evidence(scene, **inputs)


def propose(prepared, **kwargs):
    return module.propose_temporal_equipment_geometry(prepared, "marker_box", construction_frames=[1, 2],
        static_prior=PRIOR, **({"conditional_exterior": True} | kwargs))


def test_clear_second_frame_resolves_occlusion_without_losing_foreground():
    prepared = context()
    single = silhouette_geometry.propose_silhouette_geometry(prepared, "marker_box", source_frame=1,
        occluder_ids=OTHERS, conditional_exterior=True)
    assert single["status"] == "no_qualified_improvement"
    result = propose(prepared)
    report = result["report"]
    assert result["status"] == "conditional_proposal"
    assert report["source_fit"]["candidate_conditional_exterior_cells"] < report["source_fit"]["baseline_conditional_exterior_cells"]
    assert all(row["new_foreground_misses"] == 0 for row in report["frames"])
    assert all(row["clear_support_frames"] == [2] for row in report["bands"] if not row["frozen_original_width"])
    assert report["collision_proxy"]["removed_fraction"] > 0
    assert not report["collision_proxy"]["initial_clearance_checked"] and report["contact"] == "unknown"
    assert not report["stationarity"]["fixed_camera_verified"]
    assert len(report["all_source_slots"]) == 9 and report["source_slots_filled"] == 0


@pytest.mark.parametrize("change", [{"conditional_exterior": False}, {"max_parts": 1}])
def test_no_prior_authorization_or_geometric_gain_never_publishes_a_config(change):
    prepared = context(second_occluded=True)
    result = propose(prepared, **change)
    assert result["geometry_config"] is None
    assert result["report"]["collision_proxy"]["removed_fraction"] == pytest.approx(0)


def test_missing_competitor_is_not_an_unoccluded_frame():
    prepared = context()
    rows = _thaw(prepared._rows)
    rows[(2, OTHERS[0])]["proposal_status"] = "missing"
    object.__setattr__(prepared, "_rows", _freeze(rows))
    result = propose(prepared)
    assert result["geometry_config"] is None
    assert result["report"]["frames"][1]["conflicts"][0]["status"] == "missing_competitor_freezes_view"


def test_mask_holes_remain_solid_and_physical_properties_are_not_recovered():
    prepared = context(holes=True)
    result = propose(prepared)
    obj = _thaw(prepared._scene)["objects"][1]
    assert result["status"] == "conditional_proposal"
    assert all(row["filled_hole_cells"] > 0 for row in result["report"]["frames"])
    assert all(part["size"][1] == obj["size"][1] for part in result["candidate_geometry"]["parts"])
    assert sum(part["size"][2] for part in result["candidate_geometry"]["parts"]) == pytest.approx(obj["size"][2])
    dimensions = np.asarray(obj["size"])
    expected = obj["mass"] / 12 * (np.sum(dimensions ** 2) - dimensions ** 2)
    assert result["candidate_geometry"]["inertia"]["diagonal"] == pytest.approx(expected)


def test_static_image_contradiction_retains_original_and_rejection():
    result = propose(context(shift=4))
    assert result["geometry_config"] is None
    assert not result["report"]["stationarity"]["passed"]
    assert "image_stationarity_screen" in result["report"]["rejection_reasons"]
    assert result["report"]["fit_count"] == 0
    assert result["report"]["frames"][1]["baseline_missed_foreground_cells"] == result["report"]["frames"][1]["candidate_missed_foreground_cells"]


def test_second_view_positive_cells_constrain_every_band_even_when_occluded():
    prepared = context(first_occluded=False, second_occluded=True)
    rasters = _thaw(prepared._rasters)
    shape, previous, digest = rasters["masks/2-0.png"]
    changed = np.frombuffer(previous, np.uint8).reshape(shape).copy()
    changed[:, :] = 1
    rasters["masks/2-0.png"] = (shape, changed.tobytes(), digest)
    object.__setattr__(prepared, "_rasters", _freeze(rasters))
    result = propose(prepared)
    assert all(row["new_foreground_misses"] == 0 for row in result["report"]["frames"])
    assert result["geometry_config"] is None


def test_withheld_coordinates_are_never_queried():
    prepared = context()
    slots = _thaw(prepared._construction)
    key = (prepared._document_sha256, TARGET, 2)
    slots[key]["partition"], slots[key]["admission"] = "assessment", "withheld"
    object.__setattr__(prepared, "_construction", _freeze(slots))
    rows = _thaw(prepared._rows)
    rows[(2, TARGET)]["box"] = None
    object.__setattr__(prepared, "_rows", _freeze(rows))
    with pytest.raises(ValueError, match="never assessment/final"):
        propose(prepared)


@pytest.mark.parametrize("frames", [[1, 1], [2, 1], [1], [True, 2], list(range(17)), [1, 9]])
def test_invalid_frame_work_is_rejected(frames):
    with pytest.raises(ValueError):
        module.propose_temporal_equipment_geometry(context(), "marker_box", construction_frames=frames, static_prior=PRIOR)


@pytest.mark.parametrize("prior", [None, {}, {"mode": "fixed_camera_static_rigid", "provenance": " "},
                                     {"mode": "camera_moving", "provenance": "test"}])
def test_explicit_static_prior_is_required(prior):
    with pytest.raises(ValueError, match="prior"):
        module.propose_temporal_equipment_geometry(context(), "marker_box", construction_frames=[1, 2], static_prior=prior)


def test_deadline_and_pixel_bound_fail_without_partial_config(monkeypatch):
    prepared = context()
    monkeypatch.setattr(module, "_deadline", lambda _: (_ for _ in ()).throw(TimeoutError("bound")))
    with pytest.raises(TimeoutError):
        propose(prepared)
    monkeypatch.undo()
    monkeypatch.setitem(module.CONTRACT, "maximum_pixels", 1)
    with pytest.raises(ValueError, match="work bound"):
        propose(prepared)


def test_results_do_not_mutate_source_and_default_png_context_stays_identical():
    prepared = context()
    before = _thaw(prepared._scene), _thaw(prepared._hashes), _thaw(prepared._rasters)
    first = propose(prepared)
    first["candidate_geometry"]["parts"][0]["size"][0] = 20
    second = propose(prepared)
    assert second["candidate_geometry"]["parts"][0]["size"][0] < 1
    assert (_thaw(prepared._scene), _thaw(prepared._hashes), _thaw(prepared._rasters)) == before
    scene, inputs = two_frames()
    one = silhouette_geometry.prepare_silhouette_evidence(scene, **inputs)
    two = silhouette_geometry.prepare_silhouette_evidence(scene, **inputs, binary_raster_documents=None)
    assert _thaw(one._hashes) == _thaw(two._hashes)


def binary_inputs():
    scene, inputs = two_frames()
    cached = {}
    for name, encoded in inputs["raster_documents"].items():
        with Image.open(BytesIO(encoded)) as image:
            cached[name] = np.asarray(image, np.uint8).tobytes()
    return scene, {**inputs, "raster_documents": {}, "binary_raster_documents": cached}


def test_cached_binary_preparation_performs_no_png_decode_and_matches_geometry(monkeypatch):
    original = propose(context())
    scene, inputs = binary_inputs()
    monkeypatch.setattr(silhouette_geometry, "_read_raster", lambda *_: pytest.fail("must reuse cached binary"))
    prepared = silhouette_geometry.prepare_silhouette_evidence(scene, **inputs)
    result = propose(prepared)
    assert result["candidate_geometry"] == original["candidate_geometry"]
    assert set(_thaw(prepared._hashes)["raster_input_kinds"].values()) == {"cached_binary"}


@pytest.mark.parametrize("kind", ["dimensions", "hash", "nonbinary", "foreground", "overlap", "unknown"])
def test_cached_binary_rejects_invalid_custody(kind):
    scene, inputs = binary_inputs()
    name = next(iter(inputs["binary_raster_documents"]))
    masks = json.loads(inputs["mask_document"])
    row = next(row for row in masks["observations"] if row["raster_path"] == name)
    if kind == "dimensions":
        row["raster_dimensions"][0] += 1
    elif kind == "hash":
        row["raster_sha256"] = "0" * 64
    elif kind == "nonbinary":
        inputs["binary_raster_documents"][name] = b"\2" + inputs["binary_raster_documents"][name][1:]
        row["raster_sha256"] = silhouette_geometry._sha(inputs["binary_raster_documents"][name])
    elif kind == "foreground":
        row["silhouette"]["foreground_pixels"] += 1
    elif kind == "overlap":
        inputs["raster_documents"][name] = b"png"
    else:
        inputs["binary_raster_documents"]["unknown"] = b"\0"
    inputs["mask_document"] = raw(masks)
    with pytest.raises(ValueError):
        silhouette_geometry.prepare_silhouette_evidence(scene, **inputs)


@pytest.mark.parametrize("kind", ["unresolved", "gap", "group", "missing", "available"])
def test_metadata_preflight_preserves_identity_and_never_queries_geometry(kind):
    identity = {"status": "assumed_model_identity", "physical_identity_verified": False,
                "gap_seen": False, "unresolved_group_id": None}
    if kind == "unresolved":
        identity["status"] = "unresolved_association"
    if kind == "gap":
        identity["gap_seen"] = True
    if kind == "group":
        identity["unresolved_group_id"] = "original-unresolved-group"
    rows = [{"frame": frame, "object_id": "item", "id": str(frame), "time_s": frame / 3,
             "identity": deepcopy(identity), "mask_status": "decoded", "proposal_status": "returned"} for frame in [1, 2]]
    if kind == "missing":
        rows.pop()
    metadata = {"profile": "decoded-segmentation-masks-v1", "observations": rows}
    before = deepcopy(metadata)
    result = module.temporal_equipment_eligibility(metadata, "item", construction_frames=[1, 2])
    assert metadata == before
    assert result["status"] == ("requires_prepared_custody" if kind == "available" else "ineligible_source_track")
    assert not result["source_custody_verified"]
    assert result["slots"][0]["identity"] == identity


def test_stale_unresolved_group_is_not_a_resolved_association():
    prepared = context()
    rows = _thaw(prepared._rows)
    rows[(2, TARGET)]["identity"]["unresolved_group_id"] = "group"
    object.__setattr__(prepared, "_rows", _freeze(rows))
    with pytest.raises(ValueError, match="unresolved target identity"):
        propose(prepared)


def test_study_keeps_each_rejection_without_decoding_or_source_proof(tmp_path, monkeypatch):
    loaded = runpy.run_path(str(Path(__file__).parents[1] / "scripts/studies/qualify_temporal_equipment.py"))
    qualify = loaded["qualify"]
    monkeypatch.setitem(qualify.__globals__, "prepare_silhouette_evidence", lambda *a, **k: pytest.fail("no preparation"))
    monkeypatch.setitem(qualify.__globals__, "load_source_equivalence", lambda *a, **k: pytest.fail("no source proof"))
    scene, inputs = two_frames()
    masks = json.loads(inputs["mask_document"])
    for row in masks["observations"]:
        if row["object_id"] == TARGET:
            row["identity"]["status"] = "unresolved_association"
    packages = {"scene": {"scene.json": raw(scene),
                 **{scene["source_binding"]["evidence_documents"][digest]: data for digest, data in inputs["source_documents"].items()}},
                "masks": {"masks/mask-evidence.json": raw(masks), "masks/original-evidence.json": inputs["segmentation_evidence"]},
                "binaries": {"unused": b"retained but not read"}}
    plan = {"schema_version": 1, "objects": ["marker_box"], "construction_frames": [1, 2],
            "static_prior": PRIOR, "conditional_exterior": True}
    for key, members in packages.items():
        root = tmp_path / key
        root.mkdir()
        for name, data in members.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        manifest = raw({"files": {name: silhouette_geometry._sha(data) for name, data in members.items()}})
        (root / "manifest.json").write_bytes(manifest)
        plan[key] = {"path": str(root), "manifest_sha256": silhouette_geometry._sha(manifest)}
    result = qualify(plan, tmp_path / "out")
    assert result["status"] == "completed"
    assert result["objects"][0]["rejection_reasons"] == ["target_identity_unresolved"]
    assert result["cached_binary_masks"] == result["cached_png_masks"] == 0
    assert not result["prepared_source_context"]
    assert (tmp_path / "out/objects/marker_box-eligibility.json").exists()
    assert not (tmp_path / "out/objects/marker_box-geometry-config.json").exists()
