"""Conditional image constraints never become measured cavities or an admitted world."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from io import BytesIO
import json
import time

import numpy as np
from PIL import Image
import pytest

from weft import silhouette_geometry as module
from weft.equipment_proposal import floor_box_from_bounds
from weft.io import canonical_hash
from weft.primitive_geometry import silhouette_overlap
from weft.segmentation_observations import adapt_segmentation
from weft.shared_world import validate_shared_world
from weft.shared_world_evidence import build_shared_world_binding
from weft.shared_world_geometry import prepare_shared_world_geometry
from weft.shared_world_lab import authored_shared_world_scene
from weft.shared_world_packaging import write_package_files
from weft.shared_world_reconstruction import _project_box
from weft.source_equivalence import load_source_equivalence
from test_source_equivalence import fixture as mapping_fixture


SOURCE = "a" * 64
TARGET = "foreground-tool"
OTHERS = ["person-estimate", "spare-item"]


def raw(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def png(raster):
    stream = BytesIO()
    Image.fromarray(raster.astype(np.uint8)).save(stream, format="PNG")
    return stream.getvalue()


def fixture(*, target=None, occlusion=False, decode_failed=False, empty_row=False, holes=False, withheld=False, occluder_box=None):
    scene = authored_shared_world_scene()
    scene["task"]["horizon_steps"] = 2
    scene["human_spec"]["source_sha256"] = SOURCE
    height = scene["human_spec"]["mobility"]["base_height"]
    camera = {"distance_model_units": .6, "focal_px": 200., "original_scale_at_anchor_model_units_per_pixel": .003,
              "pixel_origin": [320., 240.], "world_anchor_model_units": [0., 0., height], "world_axes": ["x_right", "y_depth", "z_up"]}
    bounds = [360, 350, 410, 420]
    box = floor_box_from_bounds(bounds, camera, height, 0.)
    obj = scene["objects"][1]
    obj["position"] = (np.array(box["center_model_units"]) + scene["coordinate_frame"]["scale_bridge"]["origin_world"]).tolist()
    obj["size"] = box["dimensions_model_units"]
    target = np.zeros((56, 40), dtype=np.uint8) if target is None else target.copy()
    if not target.any():
        target[:28, 14:26] = 1
        target[28:, 5:35] = 1
    if empty_row: target[30] = 0
    if holes: target[35:40, 18:22] = 0
    arrays = {TARGET: target, OTHERS[0]: np.ones((20, 20), dtype=np.uint8), OTHERS[1]: np.ones((20, 20), dtype=np.uint8)}
    boxes = {TARGET: bounds, OTHERS[0]: bounds if occlusion else [10, 10, 30, 30], OTHERS[1]: [80, 100, 100, 130]}
    if occluder_box is not None: boxes[OTHERS[0]] = occluder_box
    ids = [TARGET, *OTHERS]
    times = [0., .033, .067]
    seg = {"profile": "segmentation-evidence-v1", "physics_ready": False,
        "source": {"submitted_video_sha256": SOURCE, "width_px": 640, "height_px": 480, "frame_count": 3},
        "time": {"timestamps_s": times}, "objects": [{"id": name, "raw_model_id": "native-" + str(index)} for index, name in enumerate(ids)],
        "observations": [], "identity_groups": [], "provenance": {"scope": "Analytic saved-mask producer, not a model decode claim"},
        "limits": ["Analytic fixture, not physical observations"]}
    masks = {"profile": "decoded-segmentation-masks-v1", "physics_ready": False, "source": deepcopy(seg["source"]),
             "identity_groups": [], "observations": [], "frames": []}
    rasters = {}
    for frame, stamp in enumerate(times):
        indices = []
        for index, name in enumerate(ids):
            returned = frame > 0
            b = boxes[name]
            encoded = "!fixture-" + name
            row = {"id": f"{SOURCE}:frame:{frame}:model-object:{index}", "object_id": name, "raw_model_id": "native-" + str(index),
                "frame": frame, "time_s": stamp, "phase": "pre_initialization" if frame == 0 else "initialization" if frame == 1 else "post_initialization",
                "proposal_status": "returned" if returned else "missing", "identity": {"status": "assumed_model_identity", "physical_identity_verified": False},
                "confidence": {"mask": .9, "box": .9},
                "geometry": {"box_xyxy_halfopen": b, "box_xywh": [b[0], b[1], b[2]-b[0], b[3]-b[1]],
                             "center_px": [(b[0]+b[2])/2, (b[1]+b[3])/2], "coordinate_system": "submitted_image_pixels", "status": "inferred"} if returned else None,
                "raw_model_output": {"mask_encoded": encoded} if returned else {},
                "mask_provenance": {"encoded_sha256": module._sha(encoded.encode())} if returned else None}
            seg["observations"].append(row)
            decoded = {key: deepcopy(row[key]) for key in ("id", "frame", "time_s", "object_id", "raw_model_id", "phase", "proposal_status", "identity", "confidence")}
            decoded.update(raw_observation_sha256=canonical_hash(row), box=deepcopy(row["geometry"]),
                encoded_sha256=row["mask_provenance"]["encoded_sha256"] if returned else None,
                mask_status="decoded" if returned else "not_returned", raster_path=None, raster_sha256=None, silhouette=None)
            if returned:
                array = arrays[name]
                path = f"masks/{frame}-{index}.png"
                decoded.update(raster_path=path, raster_sha256=module._sha(array.tobytes()), raster_dimensions=list(array.shape),
                               silhouette={"foreground_pixels": int(array.sum())})
                if frame == 1: rasters[path] = png(array)
                if decode_failed and name == OTHERS[0]:
                    decoded.update(mask_status="decode_failed", raster_path=None, raster_sha256=None, silhouette=None)
                    rasters.pop(path, None)
            indices.append(len(masks["observations"]))
            masks["observations"].append(decoded)
        masks["frames"].append({"frame": frame, "time_s": stamp, "observation_indices": indices})
    masks["original_evidence_sha256"] = canonical_hash(seg)
    doc = adapt_segmentation(seg)
    if withheld:
        for row in doc["observations"]:
            if row["frame"] == 1: row["partition"] = "audit"
    doc_raw = raw(doc)
    sha = module._sha(doc_raw)
    documents = {sha: doc_raw}
    binding, _ = build_shared_world_binding(documents, source_sha256=SOURCE, world_ids=["human", "ball", "marker_box", "table"],
        entity_refs={"human": (sha, OTHERS[0]), "marker_box": (sha, TARGET), "ball": None, "table": None})
    scene["source_binding"] = binding
    ref = next(row for row in doc["observations"] if row["frame"] == 1 and row["object_id"] == TARGET)
    obj["provenance"].update(camera_hypothesis=camera, source_reference={"document_sha256": sha, "object_id": TARGET,
        "frame": 1, "time_s": times[1], "observation_id": ref["id"], "evidence_id": ref["evidence_ids"][0]})
    scene = validate_shared_world(scene)
    return scene, {"source_documents": documents, "segmentation_evidence": raw(seg), "mask_document": raw(masks), "raster_documents": rasters}


def propose(context, **changes):
    return module.propose_silhouette_geometry(context, "marker_box", source_frame=1, occluder_ids=OTHERS,
                                              **({"conditional_exterior": True} | changes))


def repin(scene, kwargs, seg, masks, *, source_equivalences=(), second_object=False):
    original = {(row["frame"], row["object_id"]): row for row in seg["observations"]}
    for row in masks["observations"]:
        item = original[(row["frame"], row["object_id"])]
        for key in ("id", "frame", "time_s", "object_id", "raw_model_id", "phase", "proposal_status", "identity", "confidence"):
            row[key] = deepcopy(item[key])
        if "original_source" in item: row["original_source"] = deepcopy(item["original_source"])
        row["box"], row["raw_observation_sha256"] = deepcopy(item["geometry"]), canonical_hash(item)
    masks["source"], masks["original_evidence_sha256"] = deepcopy(seg["source"]), canonical_hash(seg)
    document = adapt_segmentation(seg)
    doc_raw = raw(document)
    digest = module._sha(doc_raw)
    documents = {digest: doc_raw}
    canonical = scene["human_spec"]["source_sha256"]
    refs = {"human": (digest, OTHERS[0]), "marker_box": (digest, TARGET),
            "ball": (digest, OTHERS[1]) if second_object else None, "table": None}
    scene["source_binding"], _ = build_shared_world_binding(documents, source_sha256=canonical,
        world_ids=["human", *(obj["id"] for obj in scene["objects"])], entity_refs=refs, source_equivalences=source_equivalences)
    for obj in scene["objects"]:
        if refs[obj["id"]] is None: continue
        name = refs[obj["id"]][1]
        row = next(item for item in document["observations"] if item["object_id"] == name and item["frame"] == 1)
        obj["provenance"]["source_reference"] = {"document_sha256": digest, "object_id": name, "frame": 1,
            "time_s": row["time_s"], "observation_id": row["id"], "evidence_id": row["evidence_ids"][0]}
    return validate_shared_world(scene), {**kwargs, "source_documents": documents, "segmentation_evidence": raw(seg),
        "mask_document": raw(masks), "source_equivalences": source_equivalences}


def test_reusable_source_context_proposes_non_equivalent_solid_geometry_and_pins_every_slot(tmp_path):
    scene, kwargs = fixture()
    before = deepcopy((scene, kwargs))
    context = module.prepare_silhouette_evidence(scene, **kwargs)
    result = propose(context)
    assert result["status"] == "conditional_proposal"
    report, geometry = result["report"], result["candidate_geometry"]
    assert (scene, kwargs) == before and len(geometry["parts"]) <= 8
    assert report["candidate"]["scores"]["conditional_clear_iou"] > report["baseline"]["scores"]["conditional_clear_iou"]
    assert any(part["size"][0] < scene["objects"][1]["size"][0] - 1e-6 for part in geometry["parts"])
    assert report["lost_original_foreground_cells"] == report["lost_original_uncertain_cells"] == report["new_outside_original_cells"] == 0
    assert len(report["all_source_slots"]) == 9 and report["source_slots_filled"] == 0
    assert sum(row["mask_status"] == "not_returned" for row in report["all_source_slots"]) == 3
    assert all(part["size"][1] == scene["objects"][1]["size"][1] for part in geometry["parts"])
    assert sum(part["size"][2] for part in geometry["parts"]) == pytest.approx(scene["objects"][1]["size"][2])
    assert report["physics_steps"] == report["video_decodes"] == report["inference_calls"] == 0
    assert report["physical_validation"] == "unverified" and not report["initial_clearance_checked"]
    assert not report["full_reference_clearance_checked"]
    assert result["geometry_config"]["base_scene_sha256"] == canonical_hash(scene)
    source = tmp_path / "source"
    source.mkdir()
    files = {"scene.json": raw(scene), **{scene["source_binding"]["evidence_documents"][sha]: data for sha, data in kwargs["source_documents"].items()}}
    write_package_files(source, files)
    write_package_files(source, {"manifest.json": raw({"files": {name: module._sha(data) for name, data in files.items()}})})
    config = tmp_path / "config.json"
    config.write_bytes(raw(result["geometry_config"]))
    admitted = prepare_shared_world_geometry(source, config, tmp_path / "admitted")
    assert admitted["status"] == "ready", admitted["reasons"]


def test_context_and_results_cannot_mutate_shared_inputs_or_other_proposals():
    scene, kwargs = fixture()
    context = module.prepare_silhouette_evidence(scene, **kwargs)
    with pytest.raises(FrozenInstanceError): context._ids = ()
    with pytest.raises(TypeError): context._scene["bounds"] = []
    with pytest.raises(TypeError): context._scene["objects"][1]["position"][0] = 100
    first = propose(context)
    first["candidate_geometry"]["parts"][0]["size"][0] = 100
    second = propose(context)
    assert second["candidate_geometry"]["parts"][0]["size"][0] < 1
    assert first["report"]["settings_sha256"] == second["report"]["settings_sha256"]


@pytest.mark.parametrize("change", [{"conditional_exterior": False}, {"max_parts": 1}])
def test_no_opt_in_or_no_qualified_image_gain_keeps_original_alternative(change):
    scene, kwargs = fixture(target=np.ones((56, 40), dtype=np.uint8))
    result = propose(module.prepare_silhouette_evidence(scene, **kwargs), **change)
    assert result["status"] == "no_qualified_improvement" and result["geometry_config"] is None
    assert result["report"]["baseline"]["geometry"] == {"shape": "box", "size": scene["objects"][1]["size"]}
    if not change.get("conditional_exterior", True): assert result["report"]["fit_count"] == 0


@pytest.mark.parametrize("decode_failed", [False, True])
def test_person_foreground_or_unknown_raster_freezes_original_envelope(decode_failed):
    scene, kwargs = fixture(occlusion=True, decode_failed=decode_failed)
    result = propose(module.prepare_silhouette_evidence(scene, **kwargs))
    assert result["status"] == "no_qualified_improvement" and result["geometry_config"] is None
    assert all(row["frozen_original_width"] for row in result["report"]["candidate"]["bands"])
    assert all(part["size"][0] == scene["objects"][1]["size"][0] for part in result["candidate_geometry"]["parts"])


def test_holes_are_filled_and_empty_rows_are_uncertain_not_carved():
    scene, kwargs = fixture(holes=True, empty_row=True)
    result = propose(module.prepare_silhouette_evidence(scene, **kwargs))
    assert result["report"]["filled_internal_hole_cells"] == 20
    assert 30 in result["report"]["uncertain_rows"]
    assert any(row["frozen_original_width"] for row in result["report"]["candidate"]["bands"])
    assert all(row["lost_required_cells"] == 0 for row in result["report"]["candidate"]["bands"])
    assert all(part["shape"] == "box" for part in result["candidate_geometry"]["parts"])


def test_thin_foreground_between_target_cell_centers_still_marks_the_row_uncertain():
    scene, kwargs = fixture(occluder_box=[360.01, 350.01, 360.02, 350.02])
    result = propose(module.prepare_silhouette_evidence(scene, **kwargs))
    assert 0 in result["report"]["uncertain_rows"]
    person = next(row for row in result["report"]["occluder_candidates"] if row["object_id"] == OTHERS[0])
    assert person["overlapping_target_grid_cells"] == 1


def test_empty_target_cannot_authorize_removal():
    scene, kwargs = fixture()
    masks = json.loads(kwargs["mask_document"])
    row = next(row for row in masks["observations"] if row["frame"] == 1 and row["object_id"] == TARGET)
    zero = np.zeros(row["raster_dimensions"], dtype=np.uint8)
    row["raster_sha256"], row["silhouette"]["foreground_pixels"] = module._sha(zero.tobytes()), 0
    kwargs["mask_document"] = raw(masks)
    kwargs["raster_documents"][row["raster_path"]] = png(zero)
    with pytest.raises(ValueError, match="empty target"):
        propose(module.prepare_silhouette_evidence(scene, **kwargs))


@pytest.mark.parametrize("ids", [[], [OTHERS[0]], [*OTHERS, TARGET], [OTHERS[0], OTHERS[0]]])
def test_known_foreground_entities_cannot_be_silently_omitted(ids):
    scene, kwargs = fixture()
    context = module.prepare_silhouette_evidence(scene, **kwargs)
    with pytest.raises(ValueError, match="every other returned"):
        module.propose_silhouette_geometry(context, "marker_box", source_frame=1, occluder_ids=ids, conditional_exterior=True)


def test_withheld_selected_frame_cannot_influence_geometry():
    scene, kwargs = fixture(withheld=True)
    context = module.prepare_silhouette_evidence(scene, **kwargs)
    with pytest.raises(ValueError, match="unavailable or withheld"):
        propose(context)


@pytest.mark.parametrize("mutation", ["pixel", "mask_document", "model", "slot", "duplicate_slot", "clock", "path", "source", "frame_registry"])
def test_custody_and_complete_slot_tampering_rejects(mutation):
    scene, kwargs = fixture()
    masks = json.loads(kwargs["mask_document"])
    if mutation == "pixel":
        kwargs["raster_documents"]["masks/1-0.png"] = png(np.ones((56, 40), dtype=np.uint8))
    elif mutation == "mask_document": masks["original_evidence_sha256"] = "f" * 64
    elif mutation == "model": masks["observations"][3]["raw_observation_sha256"] = "f" * 64
    elif mutation == "slot": masks["observations"].pop()
    elif mutation == "duplicate_slot": masks["observations"][-1] = deepcopy(masks["observations"][-2])
    elif mutation == "clock": masks["observations"][3]["time_s"] += .01
    elif mutation == "path": masks["observations"][3]["raster_path"] = "../outside.png"
    elif mutation == "source": masks["source"]["submitted_video_sha256"] = "f" * 64
    else: masks["frames"][1]["observation_indices"] = [0, 1, 2]
    kwargs["mask_document"] = raw(masks)
    with pytest.raises(ValueError): module.prepare_silhouette_evidence(scene, **kwargs)


def test_saved_decoded_occluder_raster_is_required_even_when_target_bytes_exist():
    scene, kwargs = fixture()
    kwargs["raster_documents"].pop("masks/1-1.png")
    context = module.prepare_silhouette_evidence(scene, **kwargs)
    with pytest.raises(ValueError, match="Every decoded"):
        propose(context)


@pytest.mark.parametrize("change", [{"max_parts": 9}, {"max_parts": True}, {"conditional_exterior": 1},
    {"max_seconds": float("inf")}, {"max_seconds": 31}])
def test_invalid_work_options_reject(change):
    scene, kwargs = fixture()
    with pytest.raises(ValueError, match="bounded"):
        propose(module.prepare_silhouette_evidence(scene, **kwargs), **change)


def test_native_ray_projection_matches_existing_convex_pixel_inclusion():
    camera = {"distance_model_units": 2., "focal_px": 100., "pixel_origin": [60., 40.],
              "world_anchor_model_units": [0., 0., .8]}
    low, high = np.array([-.2, .1, .1]), np.array([.2, .3, .5])
    vertices = np.asarray(module._projected_vertices(low, high, camera))
    box = np.r_[vertices.min(axis=0), vertices.max(axis=0)]
    mask = np.ones((37, 41), dtype=np.uint8)
    u, v = np.meshgrid(box[0]+(np.arange(41)+.5)*(box[2]-box[0])/41,
                       box[1]+(np.arange(37)+.5)*(box[3]-box[1])/37)
    start, end = module._ray_intervals(u, v, low, high, camera)
    expected = silhouette_overlap(mask, box, vertices)
    assert np.count_nonzero(start <= end + 1e-12) == expected["projected_cells"]


@pytest.mark.parametrize("wrong_original_clock", [False, True])
def test_verified_derivative_and_native_clocks_remain_distinct(tmp_path, wrong_original_clock):
    equivalence = load_source_equivalence(mapping_fixture(tmp_path / "mapping"))
    scene, kwargs = fixture()
    scene["human_spec"]["source_sha256"] = equivalence.original_sha256
    seg, masks = json.loads(kwargs["segmentation_evidence"]), json.loads(kwargs["mask_document"])
    seg["source"].update(submitted_video_sha256=equivalence.derivative_sha256,
                         original_source_sha256=equivalence.original_sha256, width_px=2, height_px=2)
    for row in seg["observations"]:
        pair = equivalence.frames[row["frame"]]
        row["id"] = row["id"].replace(SOURCE, equivalence.derivative_sha256)
        row["original_source"] = {"sha256": equivalence.original_sha256, "frame": pair.source_frame,
                                   "time_s": pair.source_native_pts / 1000}
        if wrong_original_clock and row["frame"] == 1: row["original_source"]["time_s"] -= .003
        if row["geometry"]:
            for key in ("box_xyxy_halfopen", "box_xywh", "center_px"):
                row["geometry"][key] = [value / 320 for value in row["geometry"][key]]
    camera = scene["objects"][1]["provenance"]["camera_hypothesis"]
    camera["pixel_origin"] = [value / 320 for value in camera["pixel_origin"]]
    camera["focal_px"] /= 320
    camera["original_scale_at_anchor_model_units_per_pixel"] *= 320
    scene, kwargs = repin(scene, kwargs, seg, masks, source_equivalences=[equivalence])
    if wrong_original_clock:
        with pytest.raises(ValueError, match="original clock"):
            module.prepare_silhouette_evidence(scene, **kwargs)
    else:
        result = propose(module.prepare_silhouette_evidence(scene, **kwargs))
        assert result["report"]["submitted_time_s"] == .033
        assert result["report"]["original_source"]["time_s"] == .036
        assert result["report"]["input_hashes"]["source_mapping_manifests"] == [equivalence.manifest_sha256]
        assert len(result["report"]["all_source_slots"]) == 9


def test_one_prepared_context_serves_two_world_objects_without_rebuilding_evidence(monkeypatch):
    scene, kwargs = fixture()
    scene["bounds"] = [3., 3., 3.]
    box = scene["objects"][1]
    second = scene["objects"][0]
    second.pop("radius")
    second.update(shape="box", size=deepcopy(box["size"]), position=[box["position"][0] + .4, *box["position"][1:]])
    second["provenance"]["camera_hypothesis"] = deepcopy(box["provenance"]["camera_hypothesis"])
    camera = second["provenance"]["camera_hypothesis"]
    center = np.asarray(second["position"]) - scene["coordinate_frame"]["scale_bridge"]["origin_world"]
    bounds = _project_box(center, np.asarray(second["size"]) / 2, camera, camera["world_anchor_model_units"][2]).tolist()
    seg, masks = json.loads(kwargs["segmentation_evidence"]), json.loads(kwargs["mask_document"])
    for row in seg["observations"]:
        if row["object_id"] == OTHERS[1] and row["geometry"]:
            row["geometry"].update(box_xyxy_halfopen=bounds,
                box_xywh=[bounds[0], bounds[1], bounds[2]-bounds[0], bounds[3]-bounds[1]],
                center_px=[(bounds[0]+bounds[2])/2, (bounds[1]+bounds[3])/2])
    scene, kwargs = repin(scene, kwargs, seg, masks, second_object=True)
    context = module.prepare_silhouette_evidence(scene, **kwargs)
    monkeypatch.setattr(module, "build_shared_world_binding", lambda *a, **k: pytest.fail("Prepared evidence was rebuilt"))
    first = propose(context)
    second = module.propose_silhouette_geometry(context, "ball", source_frame=1,
        occluder_ids=[TARGET, OTHERS[0]], conditional_exterior=True)
    assert first["status"] == "conditional_proposal" and second["status"] == "no_qualified_improvement"
    assert first["report"]["input_hashes"] == second["report"]["input_hashes"]
    assert second["report"]["source_reference"]["object_id"] == OTHERS[1]


def test_existing_world_scale_does_not_change_depth_pose_or_centered_inertia_prior():
    scene, kwargs = fixture()
    scale = 1.25
    obj = scene["objects"][1]
    origin = np.array(scene["coordinate_frame"]["scale_bridge"]["origin_world"])
    obj["position"] = (origin + (np.asarray(obj["position"]) - origin) * scale).tolist()
    obj["size"] = (np.array(obj["size"]) * scale).tolist()
    scene["coordinate_frame"]["scale_bridge"]["body_height_to_world_scale"] = scale
    scene["bounds"] = [3., 3., 3.]
    scene.pop("action_contract")
    scene = validate_shared_world(scene)
    before = deepcopy(scene)
    result = propose(module.prepare_silhouette_evidence(scene, **kwargs))
    assert scene == before and result["status"] == "conditional_proposal"
    dims = np.array(obj["size"])
    expected = obj["mass"] / 12 * (np.sum(dims ** 2) - dims ** 2)
    np.testing.assert_array_equal(result["candidate_geometry"]["inertia"]["diagonal"], expected)
    assert all(part["position"][1] == 0 and part["size"][1] == obj["size"][1] for part in result["candidate_geometry"]["parts"])


def test_cooperative_timeout_never_returns_a_promotable_config(monkeypatch):
    scene, kwargs = fixture()
    context = module.prepare_silhouette_evidence(scene, **kwargs)
    def expired(_):
        raise TimeoutError("Injected finite budget exhausted")
    monkeypatch.setattr(module, "_deadline", expired)
    with pytest.raises(TimeoutError, match="finite budget"):
        propose(context)
    assert canonical_hash(scene) == context._hashes["scene_sha256"]


def test_maximum_native_grid_finishes_with_bounded_output(record_property):
    target = np.zeros((1000, 1000), dtype=np.uint8)
    target[:500, 350:650] = 1
    target[500:, 125:875] = 1
    scene, kwargs = fixture(target=target)
    started, cpu = time.monotonic(), time.process_time()
    result = propose(module.prepare_silhouette_evidence(scene, **kwargs))
    record_property("native_cells", target.size)
    record_property("wall_seconds", time.monotonic() - started)
    record_property("cpu_seconds", time.process_time() - cpu)
    record_property("result_bytes", len(raw(result)))
    assert result["status"] == "conditional_proposal"
    assert len(raw(result)) < module.MAX_RESULT_BYTES
    assert result["report"]["seconds"] < module.CONTRACT["maximum_seconds"]


@pytest.mark.parametrize("height,center,count", [
    (.22119963623361028, .11059981811680514, 8),
    (.22119963623361028 * 1.25, .11059981811680514 * 1.25, 8),
    (.22119963623361028, .4 + .11059981811680514, 8),
    (.22119963623361028, 6 - .11059981811680514, 8),
    (.08, .04, 8), (.01, .005, 1), (.12, .06, 8),
])
def test_serialized_band_roundoff_preserves_strict_world_envelope_and_connectivity(height, center, count):
    obj = {"position": [1., 1.2, center], "size": [.3, .4, height]}
    local = np.linspace(-height / 2, height / 2, count + 1)
    world = center + local
    lower, upper = np.asarray(obj["position"]) - np.asarray(obj["size"]) / 2, np.asarray(obj["position"]) + np.asarray(obj["size"]) / 2
    world[0], world[-1] = lower[2], upper[2]
    parts, corrections = [], []
    for index in range(count):
        position, size, low, high, correction = module._serialize_band(obj,
            [0., 0., (local[index] + local[index+1])/2], [.3, .4, local[index+1]-local[index]], world[index:index+2])
        assert np.all(low >= lower) and np.all(high <= upper)
        assert np.all(size >= .01) and low[2] <= world[index] and high[2] >= world[index+1]
        assert low[0] <= obj["position"][0]-.005 and high[0] >= obj["position"][0]+.005
        np.testing.assert_array_equal(low, np.asarray(obj["position"])+position-size/2)
        np.testing.assert_array_equal(high, np.asarray(obj["position"])+position+size/2)
        if parts:
            assert low[2] <= parts[-1][1][2]
        parts.append((low,high))
        if correction:
            corrections.append(correction)
            assert correction["maximum_absolute_change_world"] < 1e-12
    if height == .22119963623361028 and center == .11059981811680514:
        assert corrections
        assert corrections[0]["composed_world_bounds_before"][0][2] == -6.938893903907228e-18
        assert parts[0][0][2] == 0.


@pytest.mark.parametrize("scale", [1., 1.25])
def test_final_serialized_geometry_drives_saved_projection_and_coverage(scale):
    from weft.shared_world import _primitive_bounds, _world_primitives
    scene, kwargs = fixture()
    obj = scene["objects"][1]
    obj["size"][2] = .22119963623361028
    obj["position"][2] = .11059981811680514
    bridge = scene["coordinate_frame"]["scale_bridge"]
    origin = np.asarray(bridge["origin_world"])
    obj["size"] = (np.asarray(obj["size"])*scale).tolist()
    obj["position"] = (origin+(np.asarray(obj["position"])-origin)*scale).tolist()
    bridge["body_height_to_world_scale"] = scale
    scene["bounds"] = [3.,3.,3.]
    scene.pop("action_contract")
    camera = obj["provenance"]["camera_hypothesis"]
    bounds = _project_box((np.asarray(obj["position"])-origin)/scale, np.asarray(obj["size"])/(2*scale),
                          camera, camera["world_anchor_model_units"][2]).tolist()
    seg, masks = json.loads(kwargs["segmentation_evidence"]), json.loads(kwargs["mask_document"])
    for row in seg["observations"]:
        if row["object_id"] == TARGET and row["geometry"]:
            row["geometry"].update(box_xyxy_halfopen=bounds, box_xywh=[bounds[0],bounds[1],bounds[2]-bounds[0],bounds[3]-bounds[1]],
                                   center_px=[(bounds[0]+bounds[2])/2,(bounds[1]+bounds[3])/2])
    scene, kwargs = repin(scene, kwargs, seg, masks)
    before = deepcopy(scene)
    context = module.prepare_silhouette_evidence(scene, **kwargs)
    result = propose(context)
    assert scene == before
    changed = deepcopy(scene)
    replacement = changed["objects"][1]
    replacement.pop("size")
    replacement.update(shape="compound", geometry=json.loads(raw(result["candidate_geometry"])))
    validate_shared_world(changed)
    parts = _world_primitives(replacement)
    coverage = np.zeros((56,40),dtype=bool)
    u,v=np.meshgrid(bounds[0]+(np.arange(40)+.5)*(bounds[2]-bounds[0])/40,
                    bounds[1]+(np.arange(56)+.5)*(bounds[3]-bounds[1])/56)
    for part, band in zip(parts,result["report"]["candidate"]["bands"]):
        world_low,world_high=_primitive_bounds(part)
        np.testing.assert_array_equal([world_low,world_high],band["composed_world_bounds"])
        low,high=(world_low-origin)/scale,(world_high-origin)/scale
        assert band["projected_vertices_px"] == module._projected_vertices(low,high,camera)
        entry,leave=module._ray_intervals(u,v,low,high,camera)
        coverage |= entry <= leave+1e-12
    target=module._raster(context,module._thaw(context._rows[(1,TARGET)]))
    uncertain=np.zeros_like(target)
    uncertain[result["report"]["uncertain_rows"]] = True
    assert module._scores(coverage,target,uncertain) == result["report"]["candidate"]["scores"]
    assert any(band["representation_roundoff"] for band in result["report"]["candidate"]["bands"])
    dims=np.asarray(obj["size"])
    np.testing.assert_array_equal(result["candidate_geometry"]["inertia"]["diagonal"],obj["mass"]/12*(np.sum(dims**2)-dims**2))


@pytest.mark.parametrize("size", [.009, .31])
def test_representation_correction_cannot_resize_a_genuine_geometry_error(size):
    obj={"position":[1.,1.,.15],"size":[.3,.4,.3]}
    with pytest.raises(ValueError,match="roundoff alone"):
        module._serialize_band(obj,[0.,0.,0.],[.3,.4,size],[.14,.16])
