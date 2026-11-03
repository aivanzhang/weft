"""Source-conditioned constant-shape priors; never trajectory driving or recovered grasp."""

from collections.abc import Mapping
from hashlib import sha256
import itertools
import json
import math
from statistics import median
import time

from weft.io import canonical_hash


PROFILE = "moving-equipment-proposals-v1"
SETTINGS_PROFILE = "moving-equipment-priors-v1"
WORLD_PROFILE = "moving-equipment-free-body-v1"
MAX_BYTES = 16 << 20
SCOPE = "Conditional uncalibrated free-body hypotheses; no physical identity, grasp, recovered shape, task success or trajectory driving."


def _require(condition, message):
    if not condition:
        raise ValueError("Moving equipment proposal: " + message)


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _deadline(deadline):
    _require(deadline is None or type(deadline) in (int, float) and math.isfinite(deadline), "finite deadline required")
    return min(time.monotonic() + 30, deadline) if deadline is not None else time.monotonic() + 30


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Moving equipment proposal aggregate deadline exceeded")


def _number(value, low, high):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def _vector(value, low, high):
    return isinstance(value, list) and len(value) == 3 and all(_number(item, low, high) for item in value)


def validate_moving_settings(settings):
    keys = {"schema_version", "profile", "camera", "depth_offsets_model_units", "shape", "mass_kg", "friction", "initialization"}
    _require(isinstance(settings, dict) and set(settings) == keys
             and type(settings["schema_version"]) is int and settings["schema_version"] == 1
             and settings["profile"] == SETTINGS_PROFILE, "closed explicit prior settings required")
    camera = settings["camera"]
    _require(isinstance(camera, dict) and set(camera) == {"distance_model_units"}
             and _number(camera["distance_model_units"], .1, 50), "bounded authored camera distance required")
    depths = settings["depth_offsets_model_units"]
    _require(isinstance(depths, list) and 1 <= len(depths) <= 3
             and all(_number(value, -2, 2) for value in depths) and len(set(depths)) == len(depths)
             and all(camera["distance_model_units"] + value > .05 for value in depths), "unique positive camera-depth hypotheses required")
    _require(settings["shape"] == {"method": "sphere_from_median_image_box_diagonal"}, "unsupported constant-shape prior")
    _require(_number(settings["mass_kg"], .05, 200) and _vector(settings["friction"], 0, 2), "bounded authored material priors required")
    initial = settings["initialization"]
    _require(isinstance(initial, dict), "explicit initial-state model required")
    if initial.get("method") == "first_initialized_box_zero_velocity":
        _require(set(initial) == {"method"}, "zero-velocity prefix prior has no adjustable source frame")
    else:
        _require(set(initial) == {"method", "position_model_units", "velocity_model_units_per_second", "angular_velocity_rad_per_second"}
                 and initial["method"] == "authored_free_state"
                 and _vector(initial["position_model_units"], -2, 2)
                 and _vector(initial["velocity_model_units_per_second"], -4, 4)
                 and _vector(initial["angular_velocity_rad_per_second"], -200, 200), "bounded authored free-state prior required")
    return json.loads(json.dumps(settings, allow_nan=False))


def _family(packet, ids):
    context = packet.identity_context
    _require(context is None, "this version requires explicit qualification of an attached role context")
    return {"method": "complete_nonempty_raw_track_subsets_v1", "raw_object_ids": ids,
            "assignment_count": 2 ** len(ids) - 1,
            "empty_assignment": {"equipment_ids": [], "status": "unsupported_person_only_scope"},
            "complement_status": "unknown_non_equipment_hypothesis_not_verified_person",
            "physical_identity_verified": False,
            "scope": "Exhaustive subsets of supplied raw tracks only, not exhaustive physical identity/depth/shape possibilities; not the prior conditional body-correspondence family."}


def _track_proposal(track, object_id, depth_index, offset, camera, settings, body_start, first_frame, deadline):
    depth = camera["distance_model_units"] + offset
    focal = camera["focal_px"]
    origin = camera["pixel_origin"]
    base_height = camera["world_anchor_model_units"][2]
    rows, support = [], []
    for slot in track["slots"]:
        _check(deadline)
        phases = _plain(slot["phase_records"])
        row = {"source_frame": slot["source_frame"], "source_time_s": slot["source_time_s"],
               "body_time_s": slot["body_time_s"], "observation_id": slot["observation_id"],
               "evidence_refs": _plain(slot["evidence_refs"]), "phase_records": phases,
               "position_model_units": None, "box_radius_model_units": None, "shape_support": False,
               "unavailable_reasons": _plain(slot["unavailable_reasons"])}
        box = slot["measurements"].get("box", {})
        if (slot["admission"] == "included" and slot["visibility"] in ("detected", "partial")
                and box.get("status") == "inferred" and box.get("unit") == "pixels"
                and box.get("coordinate_system") == "image"):
            values = box.get("value")
            _require(isinstance(values, (list, tuple)) and len(values) == 4
                     and all(_number(v, -10000, 10000) for v in values), "invalid admitted image box")
            x, y, width, height = values
            if width > 0 and height > 0:
                row["position_model_units"] = [(x + width / 2 - origin[0]) * depth / focal, offset,
                                                base_height - (y + height / 2 - origin[1]) * depth / focal]
                row["box_radius_model_units"] = math.hypot(width, height) * depth / (2 * focal)
                row["shape_support"] = bool(phases) and all(
                    item["phase"] in ("initialization", "post_initialization") and item["proposal_status"] == "returned"
                    for item in phases)
                if row["shape_support"]:
                    support.append(row)
            else:
                row["unavailable_reasons"].append("nonpositive_image_extent")
        rows.append(row)
    reasons = []
    shape = None
    if len(support) < 3:
        reasons.append("fewer_than_three_initialized_shape_support_rows")
    else:
        radii = [row["box_radius_model_units"] for row in support]
        center_radius = median(radii)
        shape = {"method": settings["shape"]["method"], "radius_model_units": center_radius,
                 "support_count": len(radii), "radius_min_model_units": min(radii), "radius_max_model_units": max(radii),
                 "radius_median_absolute_deviation_model_units": median([abs(value - center_radius) for value in radii]),
                 "scope": "Constant sphere proxy from image-box diagonal under fixed authored depth; not exact silhouette inversion or recovered collision geometry."}
    basis = next((row for row in support if row["source_frame"] >= first_frame), None)
    source_start = next((slot["source_time_s"] for slot in track["slots"] if slot["source_frame"] == first_frame), None)
    initial = {"initial_state_status": "authored_prefix_hypothesis", "body_start_s": body_start,
               "equipment_source_start_s": source_start,
               "body_start_frame": first_frame, "basis_frame": None if basis is None else basis["source_frame"],
               "basis_source_time_s": None if basis is None else basis["source_time_s"],
               "basis_body_time_s": None if basis is None else basis["body_time_s"],
               "basis_evidence_refs": [] if basis is None else basis["evidence_refs"],
               "unknown_prefix_slot_count": None if basis is None else max(0, basis["source_frame"] - first_frame),
               "source_slots_filled": 0, "physical_state_measured": False}
    authored = settings["initialization"]
    if authored["method"] == "authored_free_state":
        initial.update(method="authored_free_state", **{key: _plain(authored[key]) for key in authored if key != "method"})
    elif basis is None:
        reasons.append("no_initialized_source_box_for_prefix_hypothesis")
    else:
        initial.update(method="zero_velocity_back_extrapolation", position_model_units=basis["position_model_units"],
                       velocity_model_units_per_second=[0., 0., 0.], angular_velocity_rad_per_second=[0., 0., 0.])
        if basis["source_frame"] == first_frame:
            initial["initial_state_status"] = "source_conditioned_authored_shape_depth_and_zero_velocity"
    return {"id": f"depth-{depth_index}:{object_id}", "object_id": object_id,
            "entity_ref": _plain(track["entity_ref"]), "depth_offset_model_units": offset,
            "shape_prior": shape, "initial_state_hypothesis": initial, "rows": rows,
            "status": "conditional_free_body_hypothesis" if not reasons else "unsupported_input", "reasons": reasons}


def derive_moving_proposals(inputs, settings, *, max_candidates=3, deadline=None):
    """Reuse one admitted immutable source packet; no native model is built or stepped."""
    from weft.moving_equipment_inputs import validate_moving_equipment_inputs

    stop = _deadline(deadline)
    packet = validate_moving_equipment_inputs(inputs, deadline=stop)
    settings = validate_moving_settings(settings)
    _require(type(max_candidates) is int and 1 <= max_candidates <= 3, "one to three candidate slots required")
    binding, body = packet.source_binding, packet.body_binding
    entities = {row["world_entity_id"]: row for row in binding["entity_bindings"] if row["world_entity_id"] != "human"}
    ids = sorted(entities)
    _require(1 <= len(ids) <= 12, "complete bounded raw inventory required")
    family = _family(packet, ids)
    scale = body["root_scale_unchanged"]["model_units_per_pixel"]
    distance = settings["camera"]["distance_model_units"]
    base = packet.base_scene
    camera = {"distance_model_units": distance, "focal_px": distance / scale,
              "pixel_origin": body["root_origin_unchanged"]["pelvis_image_px"],
              "original_scale_at_anchor_model_units_per_pixel": scale,
              "world_anchor_model_units": [0., 0., base["human_spec"]["mobility"]["base_height"]],
              "world_axes": ["x_right", "y_depth", "z_up"], "status": "authored_uncalibrated_fixed_camera"}
    first = packet.body_references[0]
    bundle = {"schema_version": 1, "profile": PROFILE, "settings": settings, "maximum_candidates": max_candidates,
              "input_pins": {"admission_receipt_sha256": sha256(packet.admission_receipt_bytes).hexdigest(),
                             "full_human_sha256": sha256(packet.human_bytes).hexdigest(),
                             "motion_evidence_sha256": sha256(packet.motion_evidence_bytes).hexdigest(),
                             "body_binding_sha256": canonical_hash(body)},
              "identity_family": family, "camera_hypothesis": camera, "track_proposals": [], "candidates": [],
              "status": "abstained", "reasons": [], "scope": SCOPE}
    count = family["assignment_count"] * len(settings["depth_offsets_model_units"])
    family["total_depth_assignment_candidates"] = count
    if count > max_candidates:
        bundle["reasons"].append("complete_factorized_family_exceeds_candidate_bound")
        return bundle
    track_index = {(source["document_sha256"], track["entity_ref"]["object_id"]): track
                   for source in packet.motion_evidence["sources"] for track in source["tracks"]}
    by_id = {}
    for depth_index, offset in enumerate(settings["depth_offsets_model_units"]):
        for name in ids:
            entity = entities[name]
            track = track_index[(entity["observation_document_sha256"], entity["observation_entity_id"])]
            proposal = _track_proposal(track, name, depth_index, offset, camera, settings, first["time_s"], first["source_frame"], stop)
            by_id[proposal["id"]] = proposal
            bundle["track_proposals"].append(proposal)
        for size in range(1, len(ids) + 1):
            for subset in itertools.combinations(ids, size):
                _check(stop)
                refs = [{"object_id": name, "track_proposal_id": f"depth-{depth_index}:{name}"} for name in subset]
                reasons = [f"{item['object_id']}:{reason}" for item in refs for reason in by_id[item["track_proposal_id"]]["reasons"]]
                bundle["candidates"].append({"id": f"candidate-{len(bundle['candidates']) + 1}",
                    "identity_assignment": {"equipment_ids": list(subset), "unknown_complement_ids": sorted(set(ids) - set(subset)),
                                            "physical_identity_verified": False},
                    "depth_offset_model_units": offset, "objects": refs,
                    "status": "unsupported_input" if reasons else "conditional_free_body_hypothesis", "reasons": reasons})
    bundle["status"] = "conditional_candidates"
    size = 0
    for chunk in json.JSONEncoder(allow_nan=False).iterencode(bundle):
        _check(stop)
        size += len(chunk.encode())
        _require(size <= MAX_BYTES, "complete proposal bundle exceeds byte bound")
    return bundle
