"""Conditional outer-edge bands preserving each positive and occluded image cell."""

from copy import deepcopy
import json
import math
import time

import numpy as np

from weft.association_geometry_inputs import _validate_study, validate_association_archive
from weft.equipment_temporal_silhouette import _sample_selected, _stationarity
from weft.io import canonical_hash
from weft.silhouette_geometry import (
    SilhouetteEvidence, _deadline, _mapped_foreground, _projected_vertices, _ray_intervals,
    _scores, _serialize_band,
)
from weft.world_evidence import _thaw


PROFILE = "conditional-cell-occlusion-equipment-bands-v1"
CONTRACT = {
    "maximum_seconds": 20., "maximum_pixels": 1_000_000, "maximum_parts": 8,
    "protected": "Every baseline-reachable positive target row-hull cell and every actual competing-foreground cell across every selected view; entire empty target rows and missing-competitor views remain uncertain.",
    "method": "Narrow only original left/right band faces while retaining all protected cell-ray intersections; freeze bands without protected support.",
    "uncertainty": "Image overlap is unknown foreground ownership, not measured depth ordering. A clear opposite-side edge may constrain a band without clearing its occupied cells.",
    "preserved": "Original source slots, identities, groups, clocks, depth, pose, mass, materials, floor and centered box inertia.",
    "scope": "Unadmitted conditional source-image hypothesis; no cavity, physical identity, contact, grasp or support recovery.",
}


def _require(condition, message):
    if not condition:
        raise ValueError("Cell-occlusion geometry: " + message)


def _cell_sample(context, selected, frame, camera, lower, upper, deadline):
    sample = _sample_selected(context, selected, frame, camera, lower, upper, deadline)
    row_uncertain = sample["uncertain"]
    uncertain = np.broadcast_to(~sample["target"].any(axis=1)[:, None], sample["target"].shape).copy()
    for name in context._ids:
        _deadline(deadline)
        if name == selected["object_id"]:
            continue
        other = _thaw(context._rows[(frame, name)])
        slot = context._construction[(context._document_sha256, name, frame)]
        _require(slot["partition"] == "construction" and slot["admission"] == "included",
                 "competitor evidence must remain construction-only")
        if other["proposal_status"] != "returned":
            uncertain[:] = True
        else:
            occupied, _ = _mapped_foreground(context, other, sample["box"], sample["target"].shape)
            uncertain |= occupied
    sample.update(uncertain=uncertain, prior_whole_row_uncertain_cells=int(row_uncertain.sum()),
                  actual_cell_uncertain_cells=int(uncertain.sum()))
    return sample


def _bands(obj, camera, center, lower, upper, scale, origin, samples, original_parts, *, enabled, deadline):
    parts, bands = [], []
    minimum = .01 / scale
    for original in original_parts:
        _deadline(deadline)
        lo, hi = lower.copy(), upper.copy()
        z = obj["position"][2] + original["position"][2]
        edges = np.array([z - original["size"][2] / 2, z + original["size"][2] / 2])
        lo[2], hi[2] = (edges - origin[2]) / scale
        required_views, low_constraints, high_constraints = [], [], []
        for sample in samples:
            entry, leave = _ray_intervals(sample["u"], sample["v"], lo, hi, camera)
            reachable = entry <= leave + 1e-12
            required = reachable & (sample["hull"] | sample["uncertain"])
            required_views.append(required)
            if required.any():
                ray = (sample["u"][required] - camera["pixel_origin"][0]) / camera["focal_px"]
                near, far = ray * entry[required], ray * leave[required]
                low_constraints.append(float(np.maximum(near, far).min()))
                high_constraints.append(float(np.minimum(near, far).max()))
        frozen = not enabled or not low_constraints
        if not frozen:
            lo[0] = max(lower[0], min(min(low_constraints), center[0] - minimum / 2))
            hi[0] = min(upper[0], max(max(high_constraints), center[0] + minimum / 2))
        position = ((lo + hi) / 2 - center) * scale
        position[1], position[2] = 0., original["position"][2]
        if frozen:
            position[0] = 0.
        size = (hi - lo) * scale
        size[0] = obj["size"][0] if frozen else max(.01, float(size[0]))
        size[1], size[2] = obj["size"][1], original["size"][2]
        position, size, world_lo, world_hi, correction = _serialize_band(obj, position, size, edges)
        lo, hi = (world_lo - origin) / scale, (world_hi - origin) / scale
        losses, positive_losses, uncertain_losses = [], [], []
        for sample, required in zip(samples, required_views):
            entry, leave = _ray_intervals(sample["u"], sample["v"], lo, hi, camera)
            covered = entry <= leave + 1e-12
            sample["candidate"] |= covered
            losses.append(int(np.count_nonzero(required & ~covered)))
            positive_losses.append(int(np.count_nonzero(required & sample["hull"] & ~covered)))
            uncertain_losses.append(int(np.count_nonzero(required & sample["uncertain"] & ~covered)))
        parts.append({"id": original["id"], "shape": "box", "position": position.tolist(),
                      "quaternion": [1., 0., 0., 0.], "size": size.tolist()})
        bands.append({"part_id": original["id"], "frozen_original_width": frozen,
            "protected_cells": [int(row.sum()) for row in required_views], "lost_protected_cells": losses,
            "lost_row_hull_cells": positive_losses, "lost_uncertain_cells": uncertain_losses,
            "composed_world_bounds": [world_lo.tolist(), world_hi.tolist()], "representation_roundoff": correction})
    return parts, bands


def _fit(context, object_id, prior, *, max_seconds):
    started, deadline = time.monotonic(), time.monotonic() + max_seconds
    scene = _thaw(context._scene)
    obj = next(row for row in scene["objects"] if row["id"] == object_id)
    _require(obj["shape"] == "box" and obj["quaternion"] == [1., 0., 0., 0.], "original unchanged axis-aligned box required")
    camera = deepcopy(obj["provenance"]["camera_hypothesis"])
    bridge = scene["coordinate_frame"]["scale_bridge"]
    scale, origin = bridge["body_height_to_world_scale"], np.asarray(bridge["origin_world"])
    center = (np.asarray(obj["position"]) - origin) / scale
    half = np.asarray(obj["size"]) / (2 * scale)
    lower, upper = center - half, center + half
    ref = obj["provenance"]["source_reference"]
    _require(prior["source_reference"] == ref and ref["document_sha256"] == context._document_sha256,
             "cached construction reference differs")
    projected = np.asarray(_projected_vertices(lower, upper, camera))
    bounds = [projected[:, 0].min(), projected[:, 1].min(), projected[:, 0].max(), projected[:, 1].max()]
    original = _thaw(context._rows[(ref["frame"], ref["object_id"])])
    _require(np.allclose(bounds, original["box"]["box_xyxy_halfopen"], rtol=0, atol=1e-6), "original camera/envelope projection changed")
    samples, pixels = [], 0
    for frame in prior["construction_frames"]:
        _deadline(deadline)
        selected = _thaw(context._rows[(frame, ref["object_id"])])
        slot = context._construction[(context._document_sha256, ref["object_id"], frame)]
        _require(slot["partition"] == "construction" and slot["admission"] == "included"
                 and slot["visibility"] in ("detected", "partial"), "complete visible construction samples required")
        sample = _cell_sample(context, selected, frame, camera, lower, upper, deadline)
        pixels += sample["target"].size
        _require(pixels <= CONTRACT["maximum_pixels"], "selected pixel bound")
        samples.append(sample)
    stationarity = _stationarity(samples)
    original_parts = prior["candidate_geometry"]["parts"]
    _require(1 <= len(original_parts) <= CONTRACT["maximum_parts"]
             and all(band["frozen_original_width"] for band in prior["bands"]), "reuse only fully frozen original-width band partition")
    parts, bands = _bands(obj, camera, center, lower, upper, scale, origin, samples, original_parts,
                         enabled=stationarity["passed"], deadline=deadline)
    frames = []
    for sample in samples:
        before, after, uncertain = sample["baseline"], sample["candidate"], sample["uncertain"]
        frames.append({key: deepcopy(sample[key]) for key in ("frame", "time_s", "evidence_id", "original_source", "identity", "conflicts",
            "prior_whole_row_uncertain_cells", "actual_cell_uncertain_cells")} | {
            "baseline": _scores(before, sample["target"], uncertain), "candidate": _scores(after, sample["target"], uncertain),
            "new_foreground_misses": int(np.count_nonzero(before & sample["target"] & ~after)),
            "removed_uncertain_cells": int(np.count_nonzero(before & uncertain & ~after)),
            "new_outside_original_cells": int(np.count_nonzero(after & ~before)),
            "filled_hole_cells": int(np.count_nonzero(sample["hull"] & ~sample["target"]))})
    before = sum(row["baseline"]["conditional_exterior_cells"] for row in frames)
    after = sum(row["candidate"]["conditional_exterior_cells"] for row in frames)
    checks = {"image_stationarity_screen": stationarity["passed"],
        "all_positive_row_hulls_preserved": not any(any(row["lost_row_hull_cells"]) for row in bands),
        "all_actual_uncertain_cells_preserved": not any(any(row["lost_uncertain_cells"]) for row in bands)
            and not any(row["removed_uncertain_cells"] for row in frames),
        "no_new_foreground_misses": not any(row["new_foreground_misses"] for row in frames),
        "within_original_envelope": not any(row["new_outside_original_cells"] for row in frames),
        "strict_conditional_exterior_reduction": after < before}
    assumptions = deepcopy(prior["candidate_geometry"]["assumptions"]) + [
        "Cell-level unknown foreground ownership and empty rows remain protected in every selected image; clear opposite-side exterior is only an explicit conditional prior."]
    geometry = {**deepcopy(prior["candidate_geometry"]), "assumptions": assumptions, "parts": parts}
    status = "conditional_proposal" if all(checks.values()) else "no_qualified_improvement"
    config = {"schema_version": 1, "profile": "shared-world-geometry-hypothesis-config-v1",
        "base_scene_sha256": canonical_hash(scene), "assumptions": assumptions,
        "replacements": [{"object_id": object_id, "geometry": geometry}]}
    report = {"profile": PROFILE, "status": status, "object_id": object_id, "contract": deepcopy(CONTRACT),
        "source_reference": deepcopy(ref), "input_hashes": _thaw(context._hashes),
        "static_prior": deepcopy(prior["static_prior"]), "initial_association_prior": deepcopy(prior["initial_association_prior"]),
        "association_proof": deepcopy(prior["association_proof"]), "conditional_source_chain": deepcopy(prior["conditional_source_chain"]),
        "all_source_slots": _thaw(context._slots), "source_slots_filled": 0, "construction_frames": prior["construction_frames"].copy(),
        "stationarity": stationarity, "qualification": checks, "rejection_reasons": [key for key, value in checks.items() if not value],
        "frames": frames, "bands": bands, "candidate_geometry": geometry,
        "source_fit": {"baseline_cell_conditional_exterior_cells": before, "candidate_cell_conditional_exterior_cells": after,
            "previous_whole_row_exterior_cells": prior["source_fit"]["baseline_conditional_exterior_cells"],
            "scope": "Within-method cell-level conditional comparison; not directly comparable to the previous whole-row exterior count."},
        "collision_proxy": {"baseline_volume_world3": float(np.prod(obj["size"])),
                            "candidate_volume_world3": sum(float(np.prod(part["size"])) for part in parts)},
        "raw_identity_reclassified": False, "physical_identity_verified": False, "physical_validation": "unverified", "contact": "unknown",
        "physics_steps": 0, "api_calls": 0, "inference_calls": 0, "video_decodes": 0, "fit_count": 1, "seconds": time.monotonic()-started}
    _deadline(deadline)
    _require(len(json.dumps(report, allow_nan=False).encode()) <= 4 << 20, "bounded result required")
    return {"status": status, "geometry_config": config if status == "conditional_proposal" else None,
            "candidate_geometry": geometry, "report": report}


def propose_cell_occlusion_geometry(context, object_id, *, retained_archive, expected_archive_sha256, max_seconds=20.):
    """One new cell-protected fit using a complete pinned prior association study."""
    _require(isinstance(context, SilhouetteEvidence), "verified source silhouette context required")
    _require(type(max_seconds) in (int, float) and math.isfinite(max_seconds)
             and 0 < max_seconds <= CONTRACT["maximum_seconds"], "bounded fit time required")
    started = time.monotonic()
    files = validate_association_archive(retained_archive, expected_sha256=expected_archive_sha256,
                                         deadline=started+max_seconds)
    plan, _, _ = _validate_study(files)
    _require(object_id in plan["objects"] and plan["conditional_exterior"] is True, "original explicit construction selection required")
    old = json.loads(files["study/" + object_id + ".json"])
    prior = old["report"]
    _require(old["status"] == "no_qualified_improvement" and old["geometry_config"] is None,
             "do not refit an already qualified object")
    _require(prior["input_hashes"]["scene_sha256"] == context._hashes["scene_sha256"]
             and prior["input_hashes"]["mask_document_sha256"] == context._hashes["mask_document_sha256"]
             and prior["input_hashes"]["source_documents"] == _thaw(context._hashes["source_documents"])
             and prior["input_hashes"]["segmentation_file_sha256"] == context._hashes["segmentation_file_sha256"]
             and prior["input_hashes"]["source_mapping_manifests"] == _thaw(context._hashes["source_mapping_manifests"]),
             "new fit must retain exact original source/camera/context")
    result = _fit(context, object_id, prior, max_seconds=max_seconds-(time.monotonic()-started))
    result["report"].update(retained_archive_sha256=expected_archive_sha256,
        previous_proposal_sha256=canonical_hash(old), previous_source_fit_reexecuted=False)
    _deadline(started+max_seconds)
    return result
