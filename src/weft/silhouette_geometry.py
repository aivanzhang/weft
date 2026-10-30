"""Conditional solid silhouette bands; absent pixels never establish physical empty space."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
import hashlib
from io import BytesIO
import math
from pathlib import PurePosixPath
import time

import numpy as np
from PIL import Image

from weft.arm_reference_alignment import perspective_points
from weft.io import canonical_hash
from weft.shared_world import MAX_RIGID_PRIMITIVES, validate_shared_world
from weft.shared_world_evidence import build_shared_world_binding
from weft.shared_world_geometry import CONFIG_PROFILE
from weft.shared_world_reconstruction import _json_bytes, _project_box, _source_row, _vector
from weft.source_equivalence import SourceEquivalence
from weft.world_evidence import _freeze, _thaw


PROFILE = "conditional-silhouette-bands-v1"
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_DOCUMENT_BYTES = 16 * 1024 * 1024
MAX_MASK_PIXELS = 1_000_000
MAX_PREPARED_PIXELS = 8_000_000
MAX_RASTERS = 48
MAX_RESULT_BYTES = 4 * 1024 * 1024
CONTRACT = {
    "profile": PROFILE, "maximum_frames": 500, "maximum_entities": 8, "maximum_parts": 8, "maximum_scene_primitives": MAX_RIGID_PRIMITIVES,
    "maximum_seconds": 30, "minimum_part_size_world": .01, "center_spine_world": .01,
    "method": "equal_height_bands_with_closed_form_pixel_ray_interval_constraints",
    "band_partition": "Largest count up to max_parts satisfying the exact world-unit minimum size; independent of image scores.",
    "selection": "One candidate only; preserve each band's original covered row-hull cells; freeze uncertain bands; require strict conditional native-cell IoU improvement with no original foreground loss.",
    "uncertain": "Every returned non-target entity competes for foreground ownership. Its mask, or its entire box when no decoded raster exists, freezes intersected bands. Empty target rows also freeze bands.",
    "negative_pixels": "Only explicit conditional_exterior=True permits clear-row exterior to influence widths; it is an authored assumption, never measured empty space.",
    "holes": "Fill each nonempty row between its first and last target cell; all vertical bands and a common center spine remain solid.",
    "inertia": "Keep the original centered box diagonal inertia and mass as authored priors; no density or COM recovery.",
    "serialization": "Resolve representation roundoff within 32 input ULPs, using at most four adjacent local-center floats and two adjacent extent floats in each direction. Strict composed-world bounds, shared vertical edges and the 0.01-world-unit minimum remain unchanged; score the final serialized parts.",
    "scope": "Unadmitted construction-image hypothesis; no camera/depth recovery, cavity, grasp, contact, exercise success or physical validation.",
}


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _deadline(limit):
    if time.monotonic() >= limit:
        raise TimeoutError("Silhouette proposal exceeded its cooperative wall-time bound")


def _same(left, right, message):
    if left != right:
        raise ValueError(message)


def _index(rows, ids, count, label):
    if not isinstance(rows, list) or len(rows) != len(ids) * count:
        raise ValueError(label + " must preserve every entity/frame slot")
    result = {}
    for row in rows:
        if (not isinstance(row, dict) or type(row.get("frame")) is not int or not 0 <= row["frame"] < count
                or row.get("object_id") not in ids or (row["frame"], row["object_id"]) in result):
            raise ValueError(label + " has duplicate, missing or invalid slot identities")
        result[(row["frame"], row["object_id"])] = row
    return result


def _read_raster(raw, row):
    if type(raw) is not bytes or not 1 <= len(raw) <= 4 * 1024 * 1024:
        raise ValueError("Saved mask PNG requires bounded exact bytes")
    shape = row.get("raster_dimensions")
    if (not isinstance(shape, list) or len(shape) != 2 or any(type(n) is not int or not 1 <= n <= 4096 for n in shape)
            or math.prod(shape) > MAX_MASK_PIXELS):
        raise ValueError("Mask dimensions exceed the native-cell bound")
    with Image.open(BytesIO(raw)) as image:
        if image.format != "PNG" or image.mode not in ("P", "L", "1") or list(image.size) != shape[::-1]:
            raise ValueError("Saved mask raster format or dimensions differ")
        raster = np.asarray(image, dtype=np.uint8)
    if np.any(raster > 1) or _sha(raster.tobytes()) != row.get("raster_sha256"):
        raise ValueError("Saved binary raster differs from its decoded-mask SHA256")
    silhouette = row.get("silhouette")
    if not isinstance(silhouette, dict) or silhouette.get("foreground_pixels") != int(raster.sum()):
        raise ValueError("Saved mask foreground accounting differs")
    return (tuple(shape), raster.tobytes(), _sha(raw))


@dataclass(frozen=True, slots=True, init=False)
class SilhouetteEvidence:
    """Prepared immutable input snapshots; use prepare_silhouette_evidence()."""

    _scene: object
    _rows: object
    _rasters: object
    _construction: object
    _document_sha256: str
    _ids: tuple
    _hashes: object
    _slots: tuple

    def __init__(self):
        raise TypeError("Use prepare_silhouette_evidence() to verify inputs")


def prepare_silhouette_evidence(scene, *, source_documents, segmentation_evidence: bytes, mask_document: bytes,
                                raster_documents, source_equivalences=()):
    """Verify complete clocks/source custody once; saved PNG decoding performs no inference or video work."""
    deadline = time.monotonic() + CONTRACT["maximum_seconds"]
    base = validate_shared_world(scene)
    binding = base["source_binding"]
    if binding["mode"] != "source_qualified":
        raise ValueError("Silhouette proposals require a complete source-qualified base")
    if (not isinstance(source_equivalences, (tuple, list)) or len(source_equivalences) > 7
            or any(not isinstance(item, SourceEquivalence) for item in source_equivalences)
            or not isinstance(source_documents, Mapping) or not 1 <= len(source_documents) <= 8
            or any(type(raw) is not bytes for raw in source_documents.values())
            or not isinstance(raster_documents, Mapping) or len(raster_documents) > MAX_RASTERS
            or any(type(raw) is not bytes for raw in raster_documents.values())):
        raise ValueError("Use bounded exact source documents and saved raster bytes")
    seg = _json_bytes(segmentation_evidence, "Segmentation evidence", MAX_DOCUMENT_BYTES)
    masks = _json_bytes(mask_document, "Mask evidence", MAX_DOCUMENT_BYTES)
    size = len(segmentation_evidence) + len(mask_document) + sum(map(len, source_documents.values())) + sum(map(len, raster_documents.values()))
    if size + sum(item.total_input_bytes for item in source_equivalences) > MAX_INPUT_BYTES:
        raise ValueError("Prepared silhouette inputs exceed 64 MiB")
    refs = {row["world_entity_id"]: None if row["provenance_mode"] == "authored_addition" else
            (row["observation_document_sha256"], row["observation_entity_id"]) for row in binding["entity_bindings"]}
    expected, construction_raw = build_shared_world_binding(source_documents, source_sha256=binding["source_sha256"],
        world_ids=["human", *(obj["id"] for obj in base["objects"])], entity_refs=refs,
        source_equivalences=source_equivalences, deadline=deadline)
    expected["evidence_documents"] = deepcopy(binding["evidence_documents"])
    _same(expected, binding, "Prepared evidence does not reproduce the exact complete base source binding")
    if (seg.get("profile") != "segmentation-evidence-v1" or seg.get("physics_ready") is not False
            or masks.get("profile") != "decoded-segmentation-masks-v1" or masks.get("physics_ready") is not False
            or masks.get("original_evidence_sha256") != canonical_hash(seg) or masks.get("source") != seg.get("source")
            or masks.get("identity_groups") != seg.get("identity_groups")):
        raise ValueError("Decoded masks must pin unchanged original unpromoted segmentation evidence")
    source = seg["source"]
    count = source.get("frame_count")
    if (type(count) is not int or not 1 <= count <= CONTRACT["maximum_frames"]
            or count != binding["total_source_frames"] or source.get("original_source_sha256", binding["source_sha256"]) not in (None, binding["source_sha256"])):
        raise ValueError("Mask/source frame count or original video identity differs")
    objects = seg.get("objects")
    if not isinstance(objects, list) or not 1 <= len(objects) <= CONTRACT["maximum_entities"]:
        raise ValueError("Mask evidence requires bounded entity identities")
    ids = [obj.get("id") for obj in objects]
    if any(not isinstance(name, str) or not 1 <= len(name) <= 80 for name in ids) or len(set(ids)) != len(ids):
        raise ValueError("Mask entities must have unique bounded identities")
    documents = {sha: _json_bytes(raw, "Source observations", MAX_DOCUMENT_BYTES) for sha, raw in source_documents.items()}
    matches = [(sha, doc) for sha, doc in documents.items() if doc["source"]["sha256"] == source["submitted_video_sha256"]
               and doc.get("provenance", {}).get("input_evidence_sha256") == canonical_hash(seg)]
    if len(matches) != 1:
        raise ValueError("Exactly one pinned observation document must bind the original mask-model output")
    document_sha, document = matches[0]
    _same(document["source"].get("frame_times_s"), seg["time"]["timestamps_s"], "Mask and observation clocks differ")
    _same({obj["id"] for obj in document["objects"]}, set(ids), "Mask and observation entity sets differ")
    if (source.get("width_px"), source.get("height_px")) != (document["image"]["width_px"], document["image"]["height_px"]):
        raise ValueError("Mask and source image dimensions differ")
    original = _index(seg["observations"], ids, count, "Original segmentation")
    decoded = _index(masks["observations"], ids, count, "Decoded masks")
    frames = masks.get("frames")
    if not isinstance(frames, list) or len(frames) != count:
        raise ValueError("Decoded mask frame registry must retain all original slots")
    indices = {frame: [] for frame in range(count)}
    for index, row in enumerate(masks["observations"]):
        indices[row["frame"]].append(index)
    for frame, item in enumerate(frames):
        if (item.get("frame") != frame or type(item.get("frame")) is not int
                or item.get("time_s") != seg["time"]["timestamps_s"][frame]
                or item.get("observation_indices") != indices[frame]):
            raise ValueError("Decoded mask frame registry does not resolve to its original observations")
    evidence = {row["id"]: row for row in document["evidence"]}
    filtered = _json_bytes(construction_raw, "Filtered construction")
    construction = {(item["document_sha256"], row["object_id"], row["frame"]): row
                    for item in filtered["sources"] for row in item["observations"]}
    equivalence = next((item for item in source_equivalences if item.derivative_sha256 == source["submitted_video_sha256"]), None)
    paths, slots = {}, []
    for key, row in decoded.items():
        _deadline(deadline)
        old = original[key]
        if row.get("raw_observation_sha256") != canonical_hash(old) or row.get("box") != old.get("geometry"):
            raise ValueError("Decoded mask row differs from its pinned original model observation")
        for name in ("id", "frame", "time_s", "object_id", "raw_model_id", "phase", "proposal_status", "identity", "confidence", "original_source"):
            _same(row.get(name), old.get(name), "Decoded mask identity, uncertainty or source clock differs")
        payload = evidence.get(old["id"], {}).get("payload")
        _same(payload, old, "Source observations do not preserve the exact originating segmentation row")
        if old["time_s"] != seg["time"]["timestamps_s"][old["frame"]]:
            raise ValueError("Mask slot clock differs from the submitted source clock")
        original_clock = old.get("original_source")
        if equivalence is not None:
            pair = equivalence.frames[old["frame"]]
            expected_clock = {"sha256": binding["source_sha256"], "frame": pair.source_frame,
                              "time_s": float(pair.source_native_pts * Fraction(equivalence.time_base))}
            _same(original_clock, expected_clock, "Mask original clock differs from the verified source mapping")
        elif source["submitted_video_sha256"] != binding["source_sha256"]:
            raise ValueError("Derivative silhouettes require their verified complete source mapping")
        elif original_clock is not None:
            _same(original_clock, {"sha256": binding["source_sha256"], "frame": old["frame"], "time_s": old["time_s"]},
                  "Unmapped source clock cannot be silently reinterpreted")
        slot = construction[(document_sha, old["object_id"], old["frame"])]
        slots.append({"frame": old["frame"], "object_id": old["object_id"], "submitted_time_s": old["time_s"],
                      "original_source": deepcopy(original_clock), "mask_status": row["mask_status"],
                      "partition": slot["partition"], "admission": slot["admission"], "phase": old["phase"]})
        if row["mask_status"] == "decoded":
            name = row.get("raster_path")
            if (not isinstance(name, str) or not 1 <= len(name) <= 256 or "\\" in name
                    or PurePosixPath(name).is_absolute() or any(part in ("", ".", "..") for part in name.split("/"))
                    or name in paths or old["proposal_status"] != "returned"
                    or row.get("encoded_sha256") != old.get("mask_provenance", {}).get("encoded_sha256")):
                raise ValueError("Saved mask raster path or encoded-model identity differs")
            paths[name] = row
        elif row["mask_status"] not in ("not_returned", "decode_failed"):
            raise ValueError("Unsupported mask availability state")
    if not set(raster_documents) <= set(paths):
        raise ValueError("Raster bytes contain an unknown or unavailable source slot")
    rasters, pixels = {}, 0
    for name, raw in raster_documents.items():
        _deadline(deadline)
        rasters[name] = _read_raster(raw, paths[name])
        pixels += math.prod(rasters[name][0])
        if pixels > MAX_PREPARED_PIXELS:
            raise ValueError("Prepared native mask rasters exceed their aggregate bound")
    result = object.__new__(SilhouetteEvidence)
    values = {"_scene": _freeze(base), "_rows": _freeze(decoded), "_rasters": _freeze(rasters),
              "_construction": _freeze(construction), "_document_sha256": document_sha, "_ids": tuple(ids),
              "_slots": tuple(_freeze(row) for row in sorted(slots, key=lambda row: (row["frame"], row["object_id"]))),
              "_hashes": _freeze({"scene_sha256": canonical_hash(base), "source_documents": {sha: _sha(raw) for sha, raw in source_documents.items()},
                  "segmentation_file_sha256": _sha(segmentation_evidence), "segmentation_canonical_sha256": canonical_hash(seg),
                  "mask_document_sha256": _sha(mask_document), "construction_sha256": _sha(construction_raw),
                  "source_mapping_manifests": [item.manifest_sha256 for item in source_equivalences],
                  "raster_files": {name: value[2] for name, value in rasters.items()}})}
    for name, value in values.items():
        object.__setattr__(result, name, value)
    _deadline(deadline)
    return result


def _raster(context, row):
    value = context._rasters.get(row["raster_path"])
    if value is None:
        raise ValueError("Every decoded selected-frame foreground entity requires its saved raster")
    return np.frombuffer(value[1], dtype=np.uint8).reshape(value[0]).astype(bool)


def _ray_intervals(u, v, lower, upper, camera):
    """Depth intervals of exact source-pixel rays through an axis-aligned model-space box."""
    focal, distance = camera["focal_px"], camera["distance_model_units"]
    directions = ((u - camera["pixel_origin"][0]) / focal, np.ones_like(u), -(v - camera["pixel_origin"][1]) / focal)
    origin = (0., -distance, camera["world_anchor_model_units"][2])
    lo, hi = np.full_like(u, .05), np.full_like(u, np.inf)
    for axis, direction in enumerate(directions):
        parallel = np.abs(direction) <= 1e-14
        denominator = np.where(parallel, 1., direction)
        first, second = (lower[axis] - origin[axis]) / denominator, (upper[axis] - origin[axis]) / denominator
        valid = ~parallel | (lower[axis] <= origin[axis] <= upper[axis])
        lo = np.maximum(lo, np.where(valid, np.where(parallel, -np.inf, np.minimum(first, second)), np.inf))
        hi = np.minimum(hi, np.where(valid, np.where(parallel, np.inf, np.maximum(first, second)), -np.inf))
    return lo, hi


def _projected_vertices(lower, upper, camera):
    points = np.array([[x, y, z] for x in (lower[0], upper[0]) for y in (lower[1], upper[1]) for z in (lower[2], upper[2])])
    return perspective_points(points, camera)[0].tolist()


def _adjacent(value, steps):
    values, low, high = [float(value)], float(value), float(value)
    for _ in range(steps):
        low, high = math.nextafter(low, -math.inf), math.nextafter(high, math.inf)
        values.extend((low, high))
    return values


def _serialized_interval(parent, extent, local, size, required_low, required_high):
    """Close IEEE composition gaps without weakening geometry bounds or selecting a new fit."""
    low, high = parent - extent / 2, parent + extent / 2
    budget = 32 * max(math.ulp(float(v)) for v in (parent, extent, local, size, low, high))
    if required_low < low or required_high > high or required_low >= required_high:
        raise ValueError("Serialized band requires an interval inside the original object")
    center = parent + local
    if size >= .01 and low <= center - size / 2 <= required_low and required_high <= center + size / 2 <= high:
        return local, size
    choices = []
    for offset in _adjacent(local, 4):
        center = parent + offset
        needed = max(.01, 2 * max(center - required_low, required_high - center))
        available = 2 * min(center - low, high - center)
        for width in {size, *_adjacent(needed, 2), *_adjacent(available, 2)}:
            if (not math.isfinite(width) or width < .01 or abs(offset - local) > budget
                    or abs(width - size) > budget):
                continue
            bottom, top = center - width / 2, center + width / 2
            if low <= bottom <= required_low and required_high <= top <= high:
                choices.append((abs(offset - local) + abs(width - size), abs(offset - local),
                                abs(width - size), offset, width))
    if not choices:
        raise ValueError("Band cannot preserve strict bounds, connectivity and minimum size by representational roundoff alone")
    _, _, _, local, size = min(choices)
    return local, size


def _serialize_band(obj, position, size, vertical_edges):
    """Use the same parent-plus-local composition as the runtime before projecting or scoring."""
    parent, original_size = np.asarray(obj["position"], dtype=float), np.asarray(obj["size"], dtype=float)
    position, size = np.asarray(position, dtype=float).copy(), np.asarray(size, dtype=float).copy()
    old_position, old_size = position.copy(), size.copy()
    original_low, original_high = parent - original_size / 2, parent + original_size / 2
    initial_low, initial_high = parent + position - size / 2, parent + position + size / 2
    required_low = max(original_low[0], min(initial_low[0], parent[0] - .005))
    required_high = min(original_high[0], max(initial_high[0], parent[0] + .005))
    position[0], size[0] = _serialized_interval(parent[0], original_size[0], position[0], size[0], required_low, required_high)
    position[2], size[2] = _serialized_interval(parent[2], original_size[2], position[2], size[2], *vertical_edges)
    low, high = parent + position - size / 2, parent + position + size / 2
    if (np.any(low < original_low) or np.any(high > original_high) or np.any(size < .01)
            or position[1] != 0. or size[1] != original_size[1]):
        raise ValueError("Serialized band changed its fixed depth or original strict envelope")
    correction = None
    if not np.array_equal(position, old_position) or not np.array_equal(size, old_size):
        correction = {"scope": "IEEE representation correction only; no parent pose, floor, camera, inertia or gate change",
                      "position_world_local_before": old_position.tolist(), "position_world_local_after": position.tolist(),
                      "size_world_before": old_size.tolist(), "size_world_after": size.tolist(),
                      "composed_world_bounds_before": [initial_low.tolist(), initial_high.tolist()],
                      "composed_world_bounds_after": [low.tolist(), high.tolist()],
                      "maximum_absolute_change_world": float(max(np.max(np.abs(position - old_position)), np.max(np.abs(size - old_size))))}
    return position, size, low, high, correction


def _mapped_foreground(context, row, source_box, shape):
    box = row["box"]["box_xyxy_halfopen"]
    height, width = shape
    x = np.linspace(source_box[0], source_box[2], width + 1)
    y = np.linspace(source_box[1], source_box[3], height + 1)
    inside = ((x[:-1] < box[2]) & (x[1:] > box[0]))[None, :] & ((y[:-1] < box[3]) & (y[1:] > box[1]))[:, None]
    if row["mask_status"] != "decoded":
        return inside, "returned_box_uncertain_without_decoded_mask"
    raster = _raster(context, row)
    # Any positive-area cell overlap is uncertain; center sampling can miss a thin occluder.
    cols = (x - box[0]) * raster.shape[1] / (box[2] - box[0])
    rows = (y - box[1]) * raster.shape[0] / (box[3] - box[1])
    cl = np.clip(np.floor(cols[:-1]), 0, raster.shape[1]).astype(int)
    cr = np.clip(np.ceil(cols[1:]), 0, raster.shape[1]).astype(int)
    rl = np.clip(np.floor(rows[:-1]), 0, raster.shape[0]).astype(int)
    rr = np.clip(np.ceil(rows[1:]), 0, raster.shape[0]).astype(int)
    summed = np.pad(raster.astype(np.int64).cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
    total = summed[rr[:, None], cr] - summed[rl[:, None], cr] - summed[rr[:, None], cl] + summed[rl[:, None], cl]
    return inside & (total > 0), "inferred_foreground_cell_overlap_not_depth_order"


def _scores(projected, target, uncertain):
    clear = ~uncertain
    union = int(np.count_nonzero((projected | target) & clear))
    intersection = int(np.count_nonzero(projected & target & clear))
    return {"conditional_clear_iou": intersection / union if union else None,
        "foreground_cells": int(target.sum()), "covered_foreground_cells": int(np.count_nonzero(projected & target)),
        "projected_cells": int(projected.sum()), "conditional_exterior_cells": int(np.count_nonzero(projected & ~target & clear)),
        "uncertain_projected_cells": int(np.count_nonzero(projected & uncertain)),
        "scope": "Native target-mask cell centers; clear exterior remains an opt-in assumption, not measured empty space."}


def propose_silhouette_geometry(context, object_id, *, source_frame, occluder_ids, conditional_exterior=False,
                                max_parts=8, max_seconds=30):
    """Propose one deterministic unadmitted compound, retaining the original box and all slot identities."""
    if not isinstance(context, SilhouetteEvidence):
        raise ValueError("Use a verified prepared silhouette context")
    if (type(conditional_exterior) is not bool or type(max_parts) is not int or not 1 <= max_parts <= 8
            or type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 0 < max_seconds <= 30
            or type(source_frame) is not int or not 0 <= source_frame < context._scene["source_binding"]["total_source_frames"]):
        raise ValueError("Use explicit bounded frame, parts, time and conditional-exterior options")
    deadline, started = time.monotonic() + max_seconds, time.monotonic()
    scene = _thaw(context._scene)
    objects = {obj["id"]: obj for obj in scene["objects"]}
    if object_id not in objects:
        raise ValueError("Selected rigid object is not in the original scene")
    obj = objects[object_id]
    if obj["shape"] != "box" or obj["role"] == "static" or not np.array_equal(obj["quaternion"], [1., 0., 0., 0.]):
        raise ValueError("This band family requires an unchanged axis-aligned dynamic box")
    entity = next(row for row in scene["source_binding"]["entity_bindings"] if row["world_entity_id"] == object_id)
    if entity["observation_document_sha256"] != context._document_sha256:
        raise ValueError("Selected object is not bound to these exact mask observations")
    selected = _thaw(context._rows[(source_frame, entity["observation_entity_id"])])
    ref = obj["provenance"].get("source_reference", {})
    if (ref.get("frame") != source_frame or ref.get("evidence_id") != selected["id"]
            or ref.get("document_sha256") != context._document_sha256 or ref.get("object_id") != selected["object_id"]):
        raise ValueError("The selected frame must be the original object's pinned construction observation")
    row = _thaw(_source_row(context._construction, entity, source_frame, selected["id"]))
    if ref.get("observation_id") != row["id"] or ref.get("time_s") != row["time_s"]:
        raise ValueError("Original object source clock or observation identity differs")
    others = [name for name in context._ids if name != selected["object_id"]
              and context._rows[(source_frame, name)]["proposal_status"] == "returned"]
    if (not isinstance(occluder_ids, (tuple, list)) or len(occluder_ids) != len(others)
            or any(not isinstance(name, str) for name in occluder_ids) or set(occluder_ids) != set(others)):
        raise ValueError("Explicit occluders must include every other returned foreground entity without omissions")
    if selected["mask_status"] != "decoded":
        raise ValueError("Selected construction object lacks an exact decoded silhouette")
    target = _raster(context, selected)
    if not target.any():
        raise ValueError("An empty target mask cannot authorize removing geometry")
    box = _vector(selected["box"]["box_xyxy_halfopen"], 4, "Original source mask box", 0., 4096.)
    measurement = row["measurements"].get("box", {})
    if (measurement.get("coordinate_system"), measurement.get("unit"), measurement.get("status")) != ("image", "pixels", "inferred"):
        raise ValueError("Selected box must remain inferred image-space construction evidence")
    _same(measurement["value"], [box[0], box[1], box[2] - box[0], box[3] - box[1]], "Source box and mask placement differ")
    camera = deepcopy(obj["provenance"].get("camera_hypothesis"))
    if not isinstance(camera, dict):
        raise ValueError("The original object must retain its fixed authored camera")
    for name in ("distance_model_units", "focal_px", "original_scale_at_anchor_model_units_per_pixel"):
        if type(camera.get(name)) not in (int, float) or not math.isfinite(camera[name]) or not 0 < camera[name] <= 100000:
            raise ValueError("Camera parameters require bounded positive priors")
    if (camera.get("world_axes") != ["x_right", "y_depth", "z_up"]
            or camera.get("world_anchor_model_units") != [0., 0., scene["human_spec"]["mobility"]["base_height"]]
            or not math.isclose(camera["focal_px"], camera["distance_model_units"] / camera["original_scale_at_anchor_model_units_per_pixel"], rel_tol=0, abs_tol=1e-9)):
        raise ValueError("The original fixed body-anchor camera contract differs")
    _vector(camera["pixel_origin"], 2, "Camera pixel origin")
    bridge = scene["coordinate_frame"]["scale_bridge"]
    scale = bridge["body_height_to_world_scale"]
    center = (np.asarray(obj["position"]) - bridge["origin_world"]) / scale
    half = np.asarray(obj["size"]) / (2 * scale)
    lower, upper = center - half, center + half
    if lower[1] + camera["distance_model_units"] <= .05:
        raise ValueError("Fixed original depth envelope is behind or too near the camera")
    projected_bounds = _project_box(center, half, camera, camera["world_anchor_model_units"][2])
    if not np.allclose(projected_bounds, box, rtol=0, atol=1e-6):
        raise ValueError("The fixed original proxy must project to its pinned source box; no camera or envelope adjustment")
    height, width = target.shape
    x = box[0] + (np.arange(width) + .5) * (box[2] - box[0]) / width
    y = box[1] + (np.arange(height) + .5) * (box[3] - box[1]) / height
    u, v = np.meshgrid(x, y)
    uncertain, conflict_rows = np.zeros_like(target), []
    for name in others:
        other = _thaw(context._rows[(source_frame, name)])
        other_entity = {"observation_document_sha256": context._document_sha256, "observation_entity_id": name}
        _source_row(context._construction, other_entity, source_frame, other["id"])
        occupied, kind = _mapped_foreground(context, other, box, target.shape)
        uncertain |= occupied
        conflict_rows.append({"object_id": name, "mask_status": other["mask_status"], "classification": kind,
                              "overlapping_target_grid_cells": int(occupied.sum()), "identity": deepcopy(other["identity"])})
    empty = ~target.any(axis=1)
    uncertain[empty] = True
    hull = np.zeros_like(target)
    for index in np.flatnonzero(~empty):
        first, last = np.flatnonzero(target[index])[[0, -1]]
        hull[index, first:last + 1] = True
    baseline_lo, baseline_hi = _ray_intervals(u, v, lower, upper, camera)
    baseline = baseline_lo <= baseline_hi + 1e-12
    # Row uncertainty applies across the old projected envelope, not only the competing pixel.
    uncertain_rows = uncertain.any(axis=1)
    uncertain = np.broadcast_to(uncertain_rows[:, None], target.shape).copy()
    other_primitives = sum(len(item["geometry"]["parts"]) if item["shape"] == "compound" else 1
                           for item in scene["objects"] if item["id"] != object_id)
    part_count = min(max_parts, MAX_RIGID_PRIMITIVES - other_primitives,
                     max(1, int(math.floor(obj["size"][2] / .01 + 1e-9))))
    local_edges = np.linspace(-obj["size"][2] / 2, obj["size"][2] / 2, part_count + 1)
    while part_count > 1 and np.min(np.diff(local_edges)) < .01:
        part_count -= 1
        local_edges = np.linspace(-obj["size"][2] / 2, obj["size"][2] / 2, part_count + 1)
    edges = center[2] + local_edges / scale
    world_edges = obj["position"][2] + local_edges
    world_edges[0], world_edges[-1] = obj["position"][2] - obj["size"][2] / 2, obj["position"][2] + obj["size"][2] / 2
    minimum = .01 / scale
    parts, bands, candidate = [], [], np.zeros_like(target)
    for index in range(part_count):
        _deadline(deadline)
        lo, hi = lower.copy(), upper.copy()
        lo[2], hi[2] = edges[index], edges[index + 1]
        entry, leave = _ray_intervals(u, v, lo, hi, camera)
        reachable = entry <= leave + 1e-12
        required = reachable & hull
        frozen = not conditional_exterior or bool(np.any(reachable & uncertain)) or not required.any()
        if not frozen:
            ray_x = (u - camera["pixel_origin"][0]) / camera["focal_px"]
            near, far = ray_x[required] * entry[required], ray_x[required] * leave[required]
            lo[0] = max(lower[0], min(float(np.maximum(near, far).min()), center[0] - minimum / 2))
            hi[0] = min(upper[0], max(float(np.minimum(near, far).max()), center[0] + minimum / 2))
            # A common central spine keeps neighboring solid bands connected without moving the parent COM.
            if hi[0] - lo[0] < minimum - 1e-10:
                raise ValueError("Fixed bounds cannot accommodate the minimum solid center spine")
        part_center = ((lo + hi) / 2 - center) * scale
        part_center[1] = 0.
        part_center[2] = (local_edges[index] + local_edges[index + 1]) / 2
        if frozen:
            part_center[0] = 0.
        size = (hi - lo) * scale
        size[0] = obj["size"][0] if frozen else max(.01, float(size[0]))
        size[1], size[2] = obj["size"][1], local_edges[index + 1] - local_edges[index]
        part_center, size, world_low, world_high, correction = _serialize_band(
            obj, part_center, size, world_edges[index:index + 2])
        lo = (world_low - np.asarray(bridge["origin_world"])) / scale
        hi = (world_high - np.asarray(bridge["origin_world"])) / scale
        fitted_entry, fitted_leave = _ray_intervals(u, v, lo, hi, camera)
        covered = fitted_entry <= fitted_leave + 1e-12
        candidate |= covered
        losses = int(np.count_nonzero(required & ~covered))
        parts.append({"id": f"band-{index}", "shape": "box", "position": part_center.tolist(),
                      "quaternion": [1., 0., 0., 0.], "size": size.tolist()})
        bands.append({"part_id": f"band-{index}", "frozen_original_width": frozen,
            "required_row_hull_cells": int(required.sum()), "lost_required_cells": losses,
            "original_interval_model_units": [float(lower[0]), float(upper[0])],
            "proposed_interval_model_units": [float(lo[0]), float(hi[0])],
            "composed_world_bounds": [world_low.tolist(), world_high.tolist()],
            "representation_roundoff": correction,
            "projected_vertices_px": _projected_vertices(lo, hi, camera)})
    original_scores, candidate_scores = _scores(baseline, target, uncertain), _scores(candidate, target, uncertain)
    loses = int(np.count_nonzero(baseline & target & ~candidate))
    loses_unknown = int(np.count_nonzero(baseline & uncertain & ~candidate))
    extends = int(np.count_nonzero(candidate & ~baseline))
    qualification = {"conditional_exterior_opt_in": conditional_exterior, "original_foreground_preserved": not loses,
        "uncertain_envelope_preserved": not loses_unknown, "within_original_envelope": not extends,
        "all_band_row_hulls_preserved": not any(row["lost_required_cells"] for row in bands),
        "strict_conditional_image_improvement": (candidate_scores["conditional_clear_iou"] is not None
            and candidate_scores["conditional_clear_iou"] > original_scores["conditional_clear_iou"] + 1e-12)}
    improved = all(qualification.values())
    dims = np.asarray(obj["size"])
    inertia = obj["mass"] / 12 * (np.sum(dims ** 2) - dims ** 2)
    assumptions = ["Conditional clear-row silhouette extrusion; absent pixels are not measured empty space.",
        "Competing foreground and empty rows retain the original envelope; silhouette holes are filled, not carved.",
        "Original object pose/depth/mass/materials and camera remain fixed; original box centered inertia is retained as an authored prior."]
    geometry = {"profile": "rigid-compound-v1", "origin": "authored_hypothesis", "assumptions": assumptions,
                "inertia": {"mode": "authored_centered_diagonal", "diagonal": inertia.tolist()}, "parts": parts}
    replacement = deepcopy(scene)
    changed = next(item for item in replacement["objects"] if item["id"] == object_id)
    changed.pop("size")
    changed.update(shape="compound", geometry=geometry)
    validate_shared_world(replacement)
    config = {"schema_version": 1, "profile": CONFIG_PROFILE, "base_scene_sha256": canonical_hash(scene),
              "assumptions": assumptions, "replacements": [{"object_id": object_id, "geometry": geometry}]}
    settings = {"profile": PROFILE, "object_id": object_id, "source_frame": source_frame,
                "occluder_ids": sorted(others), "conditional_exterior": conditional_exterior, "max_parts": max_parts,
                "resolved_parts": part_count,
                "max_seconds": max_seconds, "camera_sha256": canonical_hash(camera), "contract_sha256": canonical_hash(CONTRACT)}
    report = {"profile": PROFILE, "status": "conditional_proposal" if improved else "no_qualified_improvement",
        "contract": deepcopy(CONTRACT), "settings": settings, "settings_sha256": canonical_hash(settings),
        "input_hashes": _thaw(context._hashes), "camera_hypothesis": camera, "source_reference": deepcopy(ref),
        "submitted_time_s": selected["time_s"], "original_source": deepcopy(selected.get("original_source")),
        "all_source_slots": _thaw(context._slots), "source_slots_filled": 0, "identity": deepcopy(selected["identity"]),
        "source_confidence": deepcopy(selected["confidence"]), "qualification": qualification,
        "rejection_reasons": [name for name, passed in qualification.items() if not passed],
        "mask_scope": "Saved producer rasters are hash-verified; codec decoding and independent mask accuracy are not reverified.",
        "source_slot_scope": "Complete mask-model entity/frame registry; every body reference and other source document remains pinned in the unchanged base scene.",
        "occluder_candidates": conflict_rows, "uncertain_rows": np.flatnonzero(uncertain_rows).tolist(),
        "filled_internal_hole_cells": int(np.count_nonzero(hull & ~target)),
        "baseline": {"geometry": {"shape": "box", "size": deepcopy(obj["size"])}, "scores": original_scores},
        "candidate": {"geometry": deepcopy(geometry), "scores": candidate_scores, "bands": bands},
        "lost_original_foreground_cells": loses, "lost_original_uncertain_cells": loses_unknown, "new_outside_original_cells": extends,
        "initial_clearance_checked": False, "full_reference_clearance_checked": False, "physical_validation": "unverified",
        "fit_count": 1 if conditional_exterior else 0, "physics_steps": 0, "video_decodes": 0, "inference_calls": 0, "api_calls": 0,
        "seconds": time.monotonic() - started}
    _deadline(deadline)
    from json import dumps
    if len(dumps(report, allow_nan=False).encode()) > MAX_RESULT_BYTES:
        raise ValueError("Silhouette report exceeds its bounded output size")
    return {"status": report["status"], "geometry_config": deepcopy(config) if improved else None,
            "candidate_geometry": deepcopy(geometry), "report": report}
