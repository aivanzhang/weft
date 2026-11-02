"""Conditional multi-frame solid equipment bands; image agreement is not contact."""

from copy import deepcopy
import json
import math
import time

import numpy as np

from weft.io import canonical_hash
from weft.silhouette_geometry import (
    MAX_RIGID_PRIMITIVES, SilhouetteEvidence, _deadline, _mapped_foreground, _projected_vertices, _raster,
    _ray_intervals, _scores, _serialize_band,
)
from weft.world_evidence import _thaw


PROFILE = "conditional-temporal-equipment-bands-v1"
CONTRACT = {
    "maximum_frames": 16, "maximum_parts": 8, "maximum_seconds": 20.,
    "maximum_pixels": 1_000_000, "maximum_result_bytes": 4 << 20,
    "minimum_span_s": .2, "maximum_edge_range_px": 3., "maximum_edge_drift_px": 2.,
    "stationarity": "Fixed authored image-box screen; neither rigid-object stationarity nor a fixed camera is measured.",
    "method": "Union every selected baseline-reachable positive row hull; narrow a band only with at least one completely clear supporting view.",
    "supporting_view": "Its target box contains the full original projected band, every reachable row is nonempty, and no competing foreground overlaps those rows.",
    "occlusion": "Unknown rows never supply exterior evidence; another clear frame may resolve them only under the explicit static-camera/static-object prior.",
    "holes": "Every nonempty target row is filled between its outermost foreground cells; no handle or cavity is inferred.",
    "preserved": "Original depth, pose, floor, centered box inertia, mass, materials, task, source clocks and source slots.",
    "scope": "Unadmitted conditional source-image proposal; no recovered depth, grasp, support, load, contact or physical validation.",
}


def _require(condition, message):
    if not condition:
        raise ValueError("Temporal equipment: " + message)


def temporal_equipment_eligibility(mask_metadata, object_id, *, construction_frames):
    """Reject unresolved tracks cheaply; passing still requires full prepared custody."""
    _require(type(construction_frames) is list and 2 <= len(construction_frames) <= CONTRACT["maximum_frames"]
             and all(type(frame) is int for frame in construction_frames)
             and all(a < b for a, b in zip(construction_frames, construction_frames[1:])), "invalid construction frame selection")
    _require(type(mask_metadata) is dict and mask_metadata.get("profile") == "decoded-segmentation-masks-v1"
             and type(mask_metadata.get("observations")) is list and len(mask_metadata["observations"]) <= 4000,
             "bounded decoded-mask metadata required")
    selected, rows = set(construction_frames), {}
    for row in mask_metadata["observations"]:
        _require(type(row) is dict, "mask metadata rows must be objects")
        if row.get("frame") in selected and row.get("object_id") == object_id:
            _require(row["frame"] not in rows, "duplicate selected mask identity")
            rows[row["frame"]] = row
    slots = []
    for frame in construction_frames:
        row = rows.get(frame)
        reasons = []
        if row is None:
            slots.append({"frame": frame, "reasons": ["selected_mask_slot_missing"]})
            continue
        identity = row.get("identity", {})
        if (type(identity) is not dict or identity.get("status") != "assumed_model_identity"
                or identity.get("physical_identity_verified") is not False or identity.get("gap_seen", False)
                or identity.get("unresolved_group_id") is not None):
            reasons.append("target_identity_unresolved")
        if row.get("mask_status") != "decoded" or row.get("proposal_status") != "returned":
            reasons.append("target_mask_unavailable")
        slots.append({"frame": frame, "evidence_id": row.get("id"), "time_s": row.get("time_s"),
                      "original_source": deepcopy(row.get("original_source")), "identity": deepcopy(identity), "reasons": reasons})
    passed = not any(row["reasons"] for row in slots)
    return {"status": "requires_prepared_custody" if passed else "ineligible_source_track",
            "object_id": object_id, "slots": slots, "selected_frames": construction_frames.copy(),
            "source_custody_verified": False, "scope": "Metadata rejection only; passing does not qualify geometry or physical identity."}


def _stationarity(samples):
    boxes = np.asarray([sample["box"] for sample in samples])
    stamps = np.asarray([sample["time_s"] for sample in samples])
    span = float(stamps[-1] - stamps[0])
    centered = stamps - stamps.mean()
    ranges = np.ptp(boxes, axis=0)
    drift = (centered @ boxes / float(centered @ centered)) * span
    checks = {"enough_time_span": span >= CONTRACT["minimum_span_s"],
              "edge_ranges_within_authored_limit": bool(np.all(ranges <= CONTRACT["maximum_edge_range_px"])),
              "edge_drifts_within_authored_limit": bool(np.all(np.abs(drift) <= CONTRACT["maximum_edge_drift_px"]))}
    return {"checks": checks, "passed": all(checks.values()), "time_span_s": span,
            "edge_ranges_px": ranges.tolist(), "fitted_edge_drifts_px": drift.tolist(),
            "fixed_camera_verified": False, "physical_stationarity_verified": False}


def _sample(context, entity, frame, camera, lower, upper, deadline):
    selected = _thaw(context._rows[(frame, entity["observation_entity_id"])])
    source = context._construction[(context._document_sha256, selected["object_id"], frame)]
    _require(source["partition"] == "construction" and source["admission"] == "included",
             "selected frames must be construction evidence, never assessment/final evidence")
    _require(source["visibility"] in ("detected", "partial") and selected["mask_status"] == "decoded"
             and selected["proposal_status"] == "returned", "every selected target needs its original available mask")
    _require(selected["identity"].get("status") == "assumed_model_identity"
             and selected["identity"].get("physical_identity_verified") is False
             and not selected["identity"].get("gap_seen", False)
             and selected["identity"].get("unresolved_group_id") is None,
             "unresolved target identity cannot join a static shape")
    target = _raster(context, selected)
    _require(target.any(), "an empty mask cannot supply exterior evidence")
    box = np.asarray(selected["box"]["box_xyxy_halfopen"], dtype=float)
    height, width = target.shape
    u, v = np.meshgrid(box[0] + (np.arange(width) + .5) * (box[2] - box[0]) / width,
                       box[1] + (np.arange(height) + .5) * (box[3] - box[1]) / height)
    hull = np.zeros_like(target)
    empty = ~target.any(axis=1)
    for row in np.flatnonzero(~empty):
        first, last = np.flatnonzero(target[row])[[0, -1]]
        hull[row, first:last + 1] = True
    uncertain = np.broadcast_to(empty[:, None], target.shape).copy()
    conflicts = []
    for name in context._ids:
        _deadline(deadline)
        if name == selected["object_id"]:
            continue
        other = _thaw(context._rows[(frame, name)])
        slot = context._construction[(context._document_sha256, name, frame)]
        _require(slot["partition"] == "construction" and slot["admission"] == "included",
                 "competing entity evidence must use the same construction partition")
        if other["proposal_status"] != "returned":
            # A missing foreground entity is not evidence that the target is unoccluded.
            uncertain[:] = True
            conflicts.append({"object_id": name, "status": "missing_competitor_freezes_view"})
            continue
        occupied, kind = _mapped_foreground(context, other, box, target.shape)
        uncertain |= occupied
        conflicts.append({"object_id": name, "status": kind, "overlapping_cells": int(occupied.sum())})
    uncertain = np.broadcast_to(uncertain.any(axis=1)[:, None], target.shape).copy()
    entry, leave = _ray_intervals(u, v, lower, upper, camera)
    baseline = entry <= leave + 1e-12
    return {"frame": frame, "time_s": selected["time_s"], "evidence_id": selected["id"],
            "original_source": selected.get("original_source"), "identity": selected["identity"],
            "box": box, "target": target, "hull": hull, "uncertain": uncertain,
            "conflicts": conflicts, "u": u, "v": v, "baseline": baseline,
            "candidate": np.zeros_like(target)}


def propose_temporal_equipment_geometry(context, object_id, *, construction_frames,
                                        static_prior, conditional_exterior=False,
                                        max_parts=8, max_seconds=20.):
    """Return one fixed-envelope candidate; admission and task evaluation stay separate."""
    _require(isinstance(context, SilhouetteEvidence), "use a verified SilhouetteEvidence context")
    _require(type(static_prior) is dict and set(static_prior) == {"mode", "provenance"}
             and static_prior["mode"] == "fixed_camera_static_rigid"
             and type(static_prior["provenance"]) is str and 1 <= len(static_prior["provenance"].strip()) <= 2000
             and len(static_prior["provenance"]) <= 2000, "explicit fixed-camera/static-rigid prior and provenance required")
    _require(type(construction_frames) is list and 2 <= len(construction_frames) <= CONTRACT["maximum_frames"]
             and all(type(frame) is int for frame in construction_frames)
             and all(a < b for a, b in zip(construction_frames, construction_frames[1:])),
             "use two to sixteen unique increasing construction frames")
    _require(type(conditional_exterior) is bool and type(max_parts) is int and 1 <= max_parts <= 8
             and type(max_seconds) in (int, float) and math.isfinite(max_seconds)
             and 0 < max_seconds <= CONTRACT["maximum_seconds"], "invalid bounded options")
    started = time.monotonic()
    deadline = started + max_seconds
    scene = _thaw(context._scene)
    objects = {obj["id"]: obj for obj in scene["objects"]}
    _require(object_id in objects, "selected object must already exist")
    obj = objects[object_id]
    _require(obj["shape"] == "box" and obj["role"] != "static" and obj["quaternion"] == [1., 0., 0., 0.],
             "this family requires an unchanged axis-aligned dynamic box")
    entity = next(row for row in scene["source_binding"]["entity_bindings"] if row["world_entity_id"] == object_id)
    _require(entity["observation_document_sha256"] == context._document_sha256, "object/source document differs")
    ref = obj["provenance"]["source_reference"]
    _require(ref["frame"] in construction_frames, "include the original pinned construction frame")
    for frame in construction_frames:
        _require((frame, entity["observation_entity_id"]) in context._rows, "selected frame is unavailable")
        slot = context._construction[(context._document_sha256, entity["observation_entity_id"], frame)]
        _require(slot["partition"] == "construction" and slot["admission"] == "included",
                 "selected frames must be construction evidence, never assessment/final evidence")
    original = context._rows[(ref["frame"], entity["observation_entity_id"])]
    original_slot = context._construction[(context._document_sha256, entity["observation_entity_id"], ref["frame"])]
    _require(ref["document_sha256"] == context._document_sha256 and ref["object_id"] == entity["observation_entity_id"]
             and ref["evidence_id"] == original["id"] and ref["observation_id"] == original_slot["id"]
             and ref["time_s"] == original_slot["time_s"], "original construction identity/clock differs")
    camera = deepcopy(obj["provenance"]["camera_hypothesis"])
    _require(camera.get("world_axes") == ["x_right", "y_depth", "z_up"]
             and camera.get("world_anchor_model_units") == [0., 0., scene["human_spec"]["mobility"]["base_height"]],
             "the existing fixed body-anchor camera is required")
    bridge = scene["coordinate_frame"]["scale_bridge"]
    scale, origin = bridge["body_height_to_world_scale"], np.asarray(bridge["origin_world"])
    center = (np.asarray(obj["position"]) - origin) / scale
    half = np.asarray(obj["size"]) / (2 * scale)
    lower, upper = center - half, center + half
    _require(lower[1] + camera["distance_model_units"] > .05, "original envelope is behind the camera")
    vertices = np.asarray(_projected_vertices(lower, upper, camera))
    projected = [vertices[:, 0].min(), vertices[:, 1].min(), vertices[:, 0].max(), vertices[:, 1].max()]
    _require(np.allclose(projected, original["box"]["box_xyxy_halfopen"], rtol=0, atol=1e-6),
             "original box must retain its pinned projection, without moving camera or floor")
    samples, pixel_count = [], 0
    for frame in construction_frames:
        _deadline(deadline)
        sample = _sample(context, entity, frame, camera, lower, upper, deadline)
        pixel_count += sample["target"].size
        _require(pixel_count <= CONTRACT["maximum_pixels"], "selected native pixels exceed the aggregate work bound")
        samples.append(sample)
    stamps = [sample["time_s"] for sample in samples]
    _require(all(a < b for a, b in zip(stamps, stamps[1:])), "selected clocks must be strictly increasing")
    stationarity = _stationarity(samples)
    enabled = conditional_exterior and stationarity["passed"]
    other_parts = sum(len(other["geometry"]["parts"]) if other["shape"] == "compound" else 1
                      for other in scene["objects"] if other["id"] != object_id)
    count = min(max_parts, MAX_RIGID_PRIMITIVES - other_parts, max(1, math.floor(obj["size"][2] / .01 + 1e-9)))
    _require(count >= 1, "scene has no room for a bounded compound")
    edges = np.linspace(-obj["size"][2] / 2, obj["size"][2] / 2, count + 1)
    while count > 1 and np.min(np.diff(edges)) < .01:
        count -= 1
        edges = np.linspace(-obj["size"][2] / 2, obj["size"][2] / 2, count + 1)
    world_edges = obj["position"][2] + edges
    minimum = .01 / scale
    parts, bands = [], []
    for band in range(count):
        _deadline(deadline)
        lo, hi = lower.copy(), upper.copy()
        lo[2], hi[2] = center[2] + edges[band:band + 2] / scale
        corners = np.asarray(_projected_vertices(lo, hi, camera))
        bounds = np.array([corners[:, 0].min(), corners[:, 1].min(), corners[:, 0].max(), corners[:, 1].max()])
        support, required_samples, x_low, x_high = [], [], [], []
        for sample in samples:
            entry, leave = _ray_intervals(sample["u"], sample["v"], lo, hi, camera)
            reachable = entry <= leave + 1e-12
            required = reachable & sample["hull"]
            required_samples.append(required)
            box = sample["box"]
            contained = bool(np.all(bounds[:2] >= box[:2] - 1e-6) and np.all(bounds[2:] <= box[2:] + 1e-6))
            if contained and required.any() and not np.any(reachable & sample["uncertain"]):
                support.append(sample["frame"])
            if required.any():
                ray = (sample["u"][required] - camera["pixel_origin"][0]) / camera["focal_px"]
                near, far = ray * entry[required], ray * leave[required]
                x_low.append(float(np.maximum(near, far).min()))
                x_high.append(float(np.minimum(near, far).max()))
        frozen = not enabled or not support or not x_low
        if not frozen:
            lo[0] = max(lower[0], min(min(x_low), center[0] - minimum / 2))
            hi[0] = min(upper[0], max(max(x_high), center[0] + minimum / 2))
        position = ((lo + hi) / 2 - center) * scale
        position[1], position[2] = 0., (edges[band] + edges[band + 1]) / 2
        if frozen:
            position[0] = 0.
        size = (hi - lo) * scale
        size[0] = obj["size"][0] if frozen else max(.01, float(size[0]))
        size[1], size[2] = obj["size"][1], edges[band + 1] - edges[band]
        position, size, world_lo, world_hi, correction = _serialize_band(obj, position, size, world_edges[band:band + 2])
        lo, hi = (world_lo - origin) / scale, (world_hi - origin) / scale
        losses = []
        for sample, required in zip(samples, required_samples):
            entry, leave = _ray_intervals(sample["u"], sample["v"], lo, hi, camera)
            covered = entry <= leave + 1e-12
            sample["candidate"] |= covered
            losses.append(int(np.count_nonzero(required & ~covered)))
        parts.append({"id": f"band-{band}", "shape": "box", "position": position.tolist(),
                      "quaternion": [1., 0., 0., 0.], "size": size.tolist()})
        bands.append({"part_id": f"band-{band}", "frozen_original_width": frozen,
                      "clear_support_frames": support, "required_row_hull_cells": [int(row.sum()) for row in required_samples],
                      "lost_row_hull_cells": losses, "composed_world_bounds": [world_lo.tolist(), world_hi.tolist()],
                      "representation_roundoff": correction})
    frame_results = []
    for sample in samples:
        baseline, candidate, target = sample["baseline"], sample["candidate"], sample["target"]
        frame_results.append({key: deepcopy(sample[key]) for key in
                              ("frame", "time_s", "evidence_id", "original_source", "identity", "conflicts")} | {
            "source_bounds_px": sample["box"].tolist(),
            "baseline": _scores(baseline, target, sample["uncertain"]),
            "candidate": _scores(candidate, target, sample["uncertain"]),
            "baseline_missed_foreground_cells": int(np.count_nonzero(target & ~baseline)),
            "candidate_missed_foreground_cells": int(np.count_nonzero(target & ~candidate)),
            "new_foreground_misses": int(np.count_nonzero(target & baseline & ~candidate)),
            "new_outside_original_cells": int(np.count_nonzero(candidate & ~baseline)),
            "removed_uncertain_cells": int(np.count_nonzero(baseline & sample["uncertain"] & ~candidate)),
            "filled_hole_cells": int(np.count_nonzero(sample["hull"] & ~target))})
    before = sum(row["baseline"]["conditional_exterior_cells"] for row in frame_results)
    after = sum(row["candidate"]["conditional_exterior_cells"] for row in frame_results)
    checks = {"explicit_conditional_exterior": conditional_exterior, "image_stationarity_screen": stationarity["passed"],
              "no_new_foreground_misses": not any(row["new_foreground_misses"] for row in frame_results),
              "all_positive_row_hulls_preserved": not any(any(row["lost_row_hull_cells"]) for row in bands),
              "within_original_envelope": not any(row["new_outside_original_cells"] for row in frame_results),
              "strict_conditional_exterior_reduction": after < before}
    assumptions = ["Fixed camera and static rigid shape are explicitly supplied, unverified assumptions.",
                   "Only a fully clear frame constrains a band exterior; all selected foreground row hulls remain required.",
                   "Rows are solid: no recovered hole, handle, contact, grasp or load; original depth and centered box inertia remain authored."]
    dims = np.asarray(obj["size"])
    inertia = obj["mass"] / 12 * (np.sum(dims ** 2) - dims ** 2)
    geometry = {"profile": "rigid-compound-v1", "origin": "authored_hypothesis", "assumptions": assumptions,
                "inertia": {"mode": "authored_centered_diagonal", "diagonal": inertia.tolist()}, "parts": parts}
    config = {"schema_version": 1, "profile": "shared-world-geometry-hypothesis-config-v1",
              "base_scene_sha256": canonical_hash(scene), "assumptions": assumptions,
              "replacements": [{"object_id": object_id, "geometry": geometry}]}
    status = "conditional_proposal" if all(checks.values()) else "no_qualified_improvement"
    baseline_volume = float(np.prod(obj["size"]))
    candidate_volume = sum(float(np.prod(part["size"])) for part in parts)
    report = {"profile": PROFILE, "status": status, "object_id": object_id, "contract": deepcopy(CONTRACT),
              "static_prior": deepcopy(static_prior), "construction_frames": construction_frames.copy(),
              "source_reference": deepcopy(ref), "input_hashes": _thaw(context._hashes),
              "all_source_slots": _thaw(context._slots), "source_slots_filled": 0,
              "stationarity": stationarity, "qualification": checks,
              "rejection_reasons": [name for name, passed in checks.items() if not passed],
              "frames": frame_results, "bands": bands,
              "source_fit": {"baseline_conditional_exterior_cells": before, "candidate_conditional_exterior_cells": after,
                             "scope": "Selected construction native-cell centers only; inherited foreground misses remain explicit."},
              "collision_proxy": {"baseline_volume_world3": baseline_volume, "candidate_volume_world3": candidate_volume,
                                  "removed_fraction": 1 - candidate_volume / baseline_volume,
                                  "initial_clearance_checked": False, "full_reference_clearance_checked": False},
              "baseline_geometry": {"shape": "box", "size": deepcopy(obj["size"])},
              "candidate_geometry": geometry, "physical_validation": "unverified", "contact": "unknown",
              "fit_count": int(enabled), "physics_steps": 0, "video_decodes": 0, "inference_calls": 0, "api_calls": 0,
              "seconds": time.monotonic() - started}
    _deadline(deadline)
    _require(len(json.dumps(report, allow_nan=False).encode()) <= CONTRACT["maximum_result_bytes"], "result exceeds byte bound")
    return {"status": status, "geometry_config": config if all(checks.values()) else None,
            "candidate_geometry": deepcopy(geometry), "report": report}
