"""Analytical sealed inputs exercise real joins; only original detector/body admission is stubbed."""

from copy import deepcopy
import json
import time

import pytest

from test_moving_equipment_inputs import fixture as source_fixture, raw, pin, reseal, seal
from weft import moving_equipment_inputs as inputs
from weft.moving_equipment_proposal import derive_moving_proposals, validate_moving_settings


def settings(*, free=False):
    initial = {"method": "first_initialized_box_zero_velocity"}
    if free:
        initial = {"method": "authored_free_state", "position_model_units": [.3, -.3, .8],
                   "velocity_model_units_per_second": [0., 0., 0.], "angular_velocity_rad_per_second": [0., 0., 0.]}
    return {"schema_version": 1, "profile": "moving-equipment-priors-v1", "camera": {"distance_model_units": 2.},
            "depth_offsets_model_units": [-.3], "shape": {"method": "sphere_from_median_image_box_diagonal"},
            "mass_kg": 1., "friction": [.8, .01, .001], "initialization": initial}


def packet(monkeypatch, *, raw_tracks=2, initialized=True, image_shift=(0, 0), vary_shape=False):
    from weft import shared_world, exercise_body_source
    from weft.root_motion import build_root_reference
    from weft.segmentation_observations import adapt_segmentation
    from weft.shared_world_lab import authored_shared_world_scene

    validate_world = shared_world.validate_shared_world
    files, _ = source_fixture(monkeypatch)
    monkeypatch.setattr(shared_world, "validate_shared_world", validate_world)
    old_scene = json.loads(files["base/scene.json"])
    old_human = old_scene["human_spec"]
    scene = authored_shared_world_scene()
    human = scene["human_spec"]
    view = json.loads(files["base/inputs/human-source/reconstruction-evidence.json"])
    audit = build_root_reference(view, torso_length=human["lengths"]["torso"])
    first_angles = deepcopy(human["reference"][0]["joints"])
    human.update(source_sha256=old_human["source_sha256"], native_clock=old_human["native_clock"],
                 reference_evidence=old_human["reference_evidence"],
                 reference=[{"time_s": row["time_s"], "joints": deepcopy(first_angles)} for row in audit["reference"]],
                 root_reference=[{"time_s": row["time_s"], "translation": row["translation"]} for row in audit["reference"]])
    human_raw = raw(human)
    scene["source_binding"] = old_scene["source_binding"]
    scene["source_binding"]["entity_bindings"] += [
        {"world_entity_id": obj["id"], "entity_kind": "rigid_object", "observation_entity_id": None,
         "observation_document_sha256": None, "provenance_mode": "authored_addition",
         "observed_slots": 0, "missing_slots": 10, "pre_init_slots": 0} for obj in scene["objects"]]
    scene["coordinate_frame"]["clock_binding"].update(native_clock=deepcopy(human["native_clock"]),
        alignment_method="native_pts_elapsed_time_v1", source_fps=human["native_clock"]["nominal_fps"], source_time_offset_s=0.)
    scene["task"]["horizon_steps"] = 13
    scene = validate_world(scene)
    files["base/scene.json"] = raw(scene)
    files["base/human.json"] = files["base/inputs/human-source/human.json"] = human_raw
    files["base/root-motion.json"] = raw(audit)
    marker_name = "base/inputs/human-source/body-source.json"
    marker = json.loads(files[marker_name]); marker["artifacts"]["human.json"] = pin(human_raw)
    files[marker_name] = raw(marker)
    original_name = "base/inputs/human-source/source-manifest.json"
    original = json.loads(files[original_name])
    for name, value in {"human.json": human_raw, "root-motion.json": raw(audit), "body-source.json": files[marker_name]}.items():
        original["files"][name] = pin(value)
    files[original_name] = raw(original)
    segmentation = json.loads(files["equipment/segmentation-evidence.json"])
    if raw_tracks != 2:
        template = [row for row in segmentation["observations"] if row["object_id"] == "sam-1"]
        segmentation["objects"] = [{"id": f"sam-{i}", "raw_model_id": str(i)} for i in range(1, raw_tracks + 1)]
        segmentation["observations"] = []
        for i in range(1, raw_tracks + 1):
            for row in template:
                clone = deepcopy(row)
                clone.update(object_id=f"sam-{i}", raw_model_id=str(i), id=row["id"].rsplit(":", 1)[0] + f":{i}")
                segmentation["observations"].append(clone)
        segmentation["observations"].sort(key=lambda row: (row["frame"], row["object_id"]))
    if not initialized:
        for row in segmentation["observations"]:
            row["phase"] = "pre_initialization"
    for row in segmentation["observations"]:
        if row["geometry"] is not None:
            geometry = row["geometry"]
            geometry["box_xywh"][0] += image_shift[0]
            geometry["box_xywh"][1] += image_shift[1]
            if vary_shape and row["frame"] == 8:
                geometry["box_xywh"][2] *= 3
            x, y, width, height = geometry["box_xywh"]
            geometry["center_px"] = [x + width / 2, y + height / 2]
    files["equipment/segmentation-evidence.json"] = raw(segmentation)
    files["equipment/scene-observations.json"] = raw(adapt_segmentation(segmentation))
    reseal(files, "base"); reseal(files, "equipment")
    calls = []
    def body_admission(source, *, deadline):
        calls.append(1)
        assert source["human.json"] == human_raw
        return json.loads(source["human.json"]), json.loads(source["observations.json"])
    monkeypatch.setattr(exercise_body_source, "validate_body_source_files", body_admission)
    admitted = inputs.admit_moving_equipment_files(files)
    return admitted, calls


def test_complete_family_preserves_every_slot_phase_clock_and_shape_dispersion(monkeypatch):
    admitted, calls = packet(monkeypatch)
    priors = settings()
    before = deepcopy(priors)
    bundle = derive_moving_proposals(admitted, priors)
    assert priors == before and len(calls) == 1
    assert bundle["status"] == "conditional_candidates" and len(bundle["candidates"]) == 3
    assert [row["identity_assignment"]["equipment_ids"] for row in bundle["candidates"]] == [["raw-0"], ["raw-1"], ["raw-0", "raw-1"]]
    assert bundle["identity_family"]["empty_assignment"]["status"] == "unsupported_person_only_scope"
    for track in bundle["track_proposals"]:
        assert len(track["rows"]) == 10
        assert track["rows"][2]["phase_records"][0]["phase"] == "pre_initialization"
        assert track["rows"][2]["source_time_s"] == .117 and track["rows"][2]["body_time_s"] == .1
        assert not track["rows"][2]["shape_support"] and track["rows"][3]["position_model_units"] is None
        assert track["shape_prior"]["support_count"] == 6
        assert track["shape_prior"]["radius_min_model_units"] == track["shape_prior"]["radius_max_model_units"]
        initial = track["initial_state_hypothesis"]
        assert initial["initial_state_status"] == "authored_prefix_hypothesis" and initial["basis_frame"] == 4
        assert initial["body_start_s"] == 0 and initial["equipment_source_start_s"] == .017
        assert initial["basis_source_time_s"] == track["rows"][4]["source_time_s"]
        assert initial["basis_body_time_s"] == .2
        assert initial["basis_source_time_s"] == pytest.approx(.217)
        assert initial["unknown_prefix_slot_count"] == 4 and initial["source_slots_filled"] == 0
        assert initial["method"] == "zero_velocity_back_extrapolation" and not initial["physical_state_measured"]
    assert derive_moving_proposals(admitted, priors) == bundle and len(calls) == 1


def test_overflow_retains_factorized_complete_inventory_without_truncation(monkeypatch):
    admitted, _ = packet(monkeypatch, raw_tracks=3)
    bundle = derive_moving_proposals(admitted, settings())
    assert bundle["status"] == "abstained" and bundle["candidates"] == bundle["track_proposals"] == []
    assert bundle["identity_family"]["assignment_count"] == 7
    assert bundle["identity_family"]["raw_object_ids"] == ["raw-0", "raw-1", "raw-2"]
    assert bundle["reasons"] == ["complete_factorized_family_exceeds_candidate_bound"]


def test_prefix_only_returns_explicit_unavailable_candidates_not_guessed_shape(monkeypatch):
    admitted, _ = packet(monkeypatch, initialized=False)
    bundle = derive_moving_proposals(admitted, settings())
    assert len(bundle["candidates"]) == 3
    assert all(row["status"] == "unsupported_input" for row in bundle["candidates"])
    assert all(row["shape_prior"] is None for row in bundle["track_proposals"])


@pytest.mark.parametrize("change", [
    lambda s: s.update(extra=True), lambda s: s.update(schema_version=True),
    lambda s: s["camera"].update(distance_model_units=float("nan")),
    lambda s: s.update(depth_offsets_model_units=[0, 0]), lambda s: s.update(depth_offsets_model_units=[-2]),
    lambda s: s.update(mass_kg=True), lambda s: s.update(friction=[3, 0, 0]),
    lambda s: s["initialization"].update(source_frame=5),
    lambda s: s.update(shape={"method": "recovered_kettlebell"}),
])
def test_explicit_prior_schema_rejects_unbounded_or_source_picking_fields(change):
    value = settings(); change(value)
    with pytest.raises(ValueError): validate_moving_settings(value)


def test_deadline_and_admitted_packet_required(monkeypatch):
    admitted, _ = packet(monkeypatch)
    with pytest.raises(TimeoutError): derive_moving_proposals(admitted, settings(), deadline=time.monotonic() - 1)
    with pytest.raises(ValueError): derive_moving_proposals(object(), settings())


def test_fixed_camera_rays_keep_exact_image_support_and_constant_shape_dispersion(monkeypatch):
    admitted, _ = packet(monkeypatch, raw_tracks=1, image_shift=(300, 190), vary_shape=True)
    priors = settings(); priors["depth_offsets_model_units"] = [-.3, 0., .3]
    bundle = derive_moving_proposals(admitted, priors)
    camera = bundle["camera_hypothesis"]
    assert len(bundle["candidates"]) == 3
    ledger_track = next(track for source in admitted.motion_evidence["sources"] for track in source["tracks"]
                        if "raw-0" in track["world_entity_ids"])
    for track in bundle["track_proposals"]:
        depth = camera["distance_model_units"] + track["depth_offset_model_units"]
        for row, original in zip(track["rows"], ledger_track["slots"], strict=True):
            if row["position_model_units"] is None:
                continue
            x, y, z = row["position_model_units"]
            projected = [camera["pixel_origin"][0] + camera["focal_px"] * x / depth,
                         camera["pixel_origin"][1] + camera["focal_px"] * (camera["world_anchor_model_units"][2] - z) / depth]
            box = original["measurements"]["box"]["value"]
            assert projected == pytest.approx([box[0] + box[2] / 2, box[1] + box[3] / 2])
        shape = track["shape_prior"]
        assert shape["radius_max_model_units"] > shape["radius_model_units"]
        assert track["rows"][8]["shape_support"]
        assert shape["support_count"] == 6  # The larger box stays in the constant-size prior and dispersion.


def test_proposal_output_bound_fails_without_partial_bundle(monkeypatch):
    from weft import moving_equipment_proposal as proposal
    admitted, _ = packet(monkeypatch)
    monkeypatch.setattr(proposal, "MAX_BYTES", 100)
    with pytest.raises(ValueError, match="byte bound"):
        proposal.derive_moving_proposals(admitted, settings())
