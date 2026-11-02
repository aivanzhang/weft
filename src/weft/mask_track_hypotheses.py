"""Independent mask-overlap associations; never proof of physical identity or contact."""

from copy import deepcopy
from hashlib import sha256
import json
import math
import time

import numpy as np

from weft.io import canonical_hash
from weft.world_evidence import load_world_evidence


CONTRACT = {
    "profile": "adjacent-mask-association-v1", "maximum_frames": 500,
    "maximum_objects": 8, "maximum_total_pixels": 50_000_000,
    "maximum_pair_pixels": 100_000_000, "maximum_seconds": 15.,
    "minimum_iou": .5, "minimum_competitor_margin": .2,
    "maximum_time_gap_s": .2,
    "association": "Mutual unique adjacent-mask IoU, without using provider IDs to choose matches",
    "continuity": "Never bridge gaps, restart ended anchors, interpolate masks or alter raw identities",
    "scope": "Conditional image-track hypothesis; identical replacements, depth and physical identity remain unverified",
}


def _require(condition, message):
    if not condition:
        raise ValueError("Mask association: " + message)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Mask association deadline exceeded; no partial result")


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def constant(_):
        raise ValueError("Mask association: nonfinite JSON number")
    try:
        result = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (RecursionError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Mask association: invalid bounded JSON") from error
    pending, count = [(result, 0)], 0
    while pending:
        value, depth = pending.pop()
        count += 1
        _require(count <= 250_000 and depth <= 32, "JSON structure exceeds bounds")
        if isinstance(value, (dict, list)):
            pending.extend((child, depth + 1) for child in (value.values() if isinstance(value, dict) else value))
        elif isinstance(value, float):
            _require(math.isfinite(value), "nonfinite JSON number")
    _require(type(result) is dict, "JSON object required")
    return result


def _prepare(slot, width, height, work):
    _require(type(slot) is dict and set(slot) == {"id", "raw_object_id", "box", "mask", "status"},
             "exact bounded slot fields required")
    _require(all(type(slot[key]) is str and 0 < len(slot[key]) <= 240 for key in ("id", "raw_object_id")),
             "explicit evidence and raw object identities required")
    _require(slot["status"] in ("available", "missing", "pre_initialization", "withheld"),
             "unknown slot status")
    if slot["status"] != "available":
        _require(slot["box"] is None and slot["mask"] is None,
                 "unavailable/withheld slots cannot supply pixels or geometry")
        return dict(slot, area=0, mask_sha256=None)
    box, mask = slot["box"], slot["mask"]
    _require(type(box) is list and len(box) == 4 and all(type(v) is int for v in box)
             and 0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height,
             "integer original-image cell bounds required; no resizing")
    _require(isinstance(mask, np.ndarray) and mask.dtype == np.uint8 and mask.ndim == 2
             and mask.shape == (box[3] - box[1], box[2] - box[0]),
             "exact one-source-pixel binary raster required")
    work["input_pixels"] += mask.size
    _require(work["input_pixels"] <= CONTRACT["maximum_total_pixels"], "aggregate pixel bound exceeded")
    _require(not np.any(mask > 1), "raster must be binary")
    # Own the two-frame working set; caller mutation cannot alter a previous association.
    owned = mask.copy()
    return dict(slot, box=box.copy(), mask=owned, area=int(owned.sum()),
                mask_sha256=sha256(owned.tobytes()).hexdigest())


def _iou(left, right, work):
    a, b = left["box"], right["box"]
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = 0
    if x0 < x1 and y0 < y1:
        work["pair_pixels"] += (x1 - x0) * (y1 - y0)
        _require(work["pair_pixels"] <= CONTRACT["maximum_pair_pixels"], "pair pixel bound exceeded")
        intersection = int(np.count_nonzero(
            left["mask"][y0-a[1]:y1-a[1], x0-a[0]:x1-a[0]] &
            right["mask"][y0-b[1]:y1-b[1], x0-b[0]:x1-b[0]]))
    union = left["area"] + right["area"] - intersection
    return {"intersection_pixels": intersection, "union_pixels": union,
            "iou": intersection / union if union else 0.}


def track_mask_hypotheses(frames, *, width, height, maximum_seconds=15.):
    """Stream every source slot; retain only previous/current rasters and bounded scalar history.

    Frames carry frame/time_s/slots. All raw-object slots must persist, including
    unavailable ones. The caller must separately verify source/partition custody.
    """
    _require(type(width) is int and type(height) is int and 1 <= width <= 4096
             and 1 <= height <= 4096 and width * height <= 8_847_360, "bounded source dimensions required")
    _require(type(maximum_seconds) in (int, float) and math.isfinite(maximum_seconds)
             and 0 < maximum_seconds <= CONTRACT["maximum_seconds"], "invalid time bound")
    deadline = time.monotonic() + maximum_seconds
    work = {"input_pixels": 0, "pair_pixels": 0, "pair_checks": 0}
    history, previous, previous_time, raw_ids = [], [], None, None
    anchors, anchored_once, evidence_ids = {}, set(), set()
    for index, frame in enumerate(frames):
        _check(deadline)
        _require(index < CONTRACT["maximum_frames"], "frame bound exceeded")
        _require(type(frame) is dict and set(frame) == {"frame", "time_s", "slots"}
                 and type(frame["frame"]) is int and frame["frame"] == index,
                 "complete original frame sequence starting at zero required")
        stamp, slots = frame["time_s"], frame["slots"]
        _require(type(stamp) in (int, float) and math.isfinite(stamp) and 0 <= stamp <= 3600
                 and (previous_time is None or stamp > previous_time), "strict source clock required")
        _require(type(slots) is list and 1 <= len(slots) <= CONTRACT["maximum_objects"], "bounded slots required")
        current = [_prepare(slot, width, height, work) for slot in slots]
        _check(deadline)
        ids = {row["raw_object_id"] for row in current}
        _require(len(ids) == len(current) and (raw_ids is None or raw_ids == ids),
                 "complete unique raw object slots required in every frame")
        raw_ids = ids
        for row in current:
            _require(row["id"] not in evidence_ids, "duplicate evidence identity")
            evidence_ids.add(row["id"])
        before = [row for row in previous if row["status"] == "available" and row["area"]]
        after = [row for row in current if row["status"] == "available" and row["area"]]
        adjacent = previous_time is not None and stamp - previous_time <= CONTRACT["maximum_time_gap_s"] + 1e-12
        pairs = []
        if adjacent:
            for left in before:
                for right in after:
                    _check(deadline)
                    pairs.append({"previous_id": left["id"], "current_id": right["id"], **_iou(left, right, work)})
                    work["pair_checks"] += 1
        accepted = {}
        for pair in pairs:
            forward = [p["iou"] for p in pairs if p["previous_id"] == pair["previous_id"] and p is not pair]
            reverse = [p["iou"] for p in pairs if p["current_id"] == pair["current_id"] and p is not pair]
            margin = pair["iou"] - max([0., *forward, *reverse])
            pair.update(competitor_margin=margin, accepted=pair["iou"] >= CONTRACT["minimum_iou"]
                        and margin >= CONTRACT["minimum_competitor_margin"])
            if pair["accepted"]:
                accepted[pair["current_id"]] = pair["previous_id"]
        old_by_raw = {row["raw_object_id"]: row for row in previous}
        next_anchors, rows = {}, []
        for row in current:
            old = old_by_raw.get(row["raw_object_id"])
            initialization = (index == 0 or old is not None and old["status"] == "pre_initialization")
            anchor, link = None, accepted.get(row["id"])
            if row["status"] != "available":
                status = row["status"]
            elif not row["area"]:
                status = "empty_mask"
            elif link is not None and link in anchors:
                anchor, status = anchors[link], "conditional_continuation"
            elif initialization and row["raw_object_id"] not in anchored_once:
                anchor, status = row["id"], "declared_initialization"
                anchored_once.add(row["raw_object_id"])
            else:
                status = "time_gap" if previous_time is not None and not adjacent else "unresolved_association"
            if anchor is not None:
                next_anchors[row["id"]] = anchor
            rows.append({"id": row["id"], "raw_object_id": row["raw_object_id"], "status": status,
                         "anchor_id": anchor, "previous_id": link, "mask_sha256": row["mask_sha256"],
                         "box": row["box"], "foreground_pixels": row["area"]})
        history.append({"frame": index, "time_s": stamp, "slots": rows, "pairs": pairs})
        previous, previous_time, anchors = current, stamp, next_anchors
    _require(history, "at least one complete frame required")
    _check(deadline)
    return {"contract": deepcopy(CONTRACT), "frames": history, "work": work,
            "source_custody_verified": False, "raw_identity_reclassified": False,
            "physical_identity_verified": False, "contact": "unknown"}


def associate_source_masks(source_document, mask_document, *, raster_loader, maximum_seconds=15.):
    """Associate construction-only saved masks on their submitted clock, not native 3D time."""
    _require(type(maximum_seconds) in (int, float) and math.isfinite(maximum_seconds)
             and 0 < maximum_seconds <= CONTRACT["maximum_seconds"], "invalid time bound")
    deadline = time.monotonic() + maximum_seconds
    _require(type(source_document) is bytes and type(mask_document) is bytes
             and 0 < len(source_document) <= 16 << 20 and 0 < len(mask_document) <= 16 << 20,
             "bounded exact source/mask documents required")
    source_hash, masks_hash = sha256(source_document).hexdigest(), sha256(mask_document).hexdigest()
    registry = load_world_evidence({source_hash: source_document})
    filtered = json.loads(registry.construction_bytes())
    construction = {(row["frame"], row["object_id"]): row for row in filtered["sources"][0]["observations"]}
    document, masks = _json(source_document), _json(mask_document)
    source = document["source"]
    _require(masks.get("profile") == "decoded-segmentation-masks-v1" and masks.get("physics_ready") is False
             and masks.get("source", {}).get("submitted_video_sha256") == source["sha256"]
             and masks["source"].get("frame_count") == source["decoded_frames"]
             and masks["source"].get("width_px") == document["image"]["width_px"]
             and masks["source"].get("height_px") == document["image"]["height_px"]
             and masks.get("original_evidence_sha256") == document["provenance"].get("input_evidence_sha256")
             and document["provenance"].get("adapter") == "saved_segmentation",
             "mask source/producer identity differs")
    expected_mapping = "Raster cell edges span the original half-open model box; centroid uses cell centers"
    _require(masks.get("contract", {}).get("mapping") == expected_mapping
             and masks["contract"].get("method") == "saved-sam-mask-raster-v1", "unknown mask cell mapping")
    count, ids = source["decoded_frames"], [obj["id"] for obj in document["objects"]]
    _require(1 <= count <= CONTRACT["maximum_frames"] and 1 <= len(ids) <= CONTRACT["maximum_objects"],
             "source exceeds frame/object bounds")
    rows, frames = masks.get("observations"), masks.get("frames")
    _require(type(rows) is list and len(rows) == count * len(ids)
             and type(frames) is list and len(frames) == count, "complete mask rows/registry required")
    indexed, row_indices, reserved_pixels = {}, {index: [] for index in range(count)}, 0
    for index, row in enumerate(rows):
        _check(deadline)
        _require(type(row) is dict and type(row.get("frame")) is int and 0 <= row["frame"] < count
                 and row.get("object_id") in ids, "invalid mask source slot")
        key = (row["frame"], row["object_id"])
        _require(key not in indexed, "duplicate mask source slot")
        _require(row.get("mask_status") in ("decoded", "not_returned", "decode_failed")
                 and row.get("phase") in ("pre_initialization", "initialization", "post_initialization")
                 and row.get("proposal_status") in ("returned", "missing")
                 and (row["mask_status"] == "not_returned" or row["proposal_status"] == "returned"),
                 "invalid mask availability/phase state")
        if (construction[key]["admission"] == "included" and row["phase"] != "pre_initialization"
                and row["mask_status"] == "decoded"):
            shape = row.get("raster_dimensions")
            _require(type(shape) is list and len(shape) == 2
                     and all(type(n) is int and 1 <= n <= 4096 for n in shape)
                     and math.prod(shape) <= 8_847_360, "bounded native raster dimensions required")
            reserved_pixels += math.prod(shape)
            _require(reserved_pixels <= CONTRACT["maximum_total_pixels"], "aggregate raster reservation exceeded")
        indexed[key] = row
        row_indices[row["frame"]].append(index)
    records = {row["id"]: row for row in document["evidence"]}
    loaded, withheld = [], []

    def stream():
        for index, frame in enumerate(frames):
            _check(deadline)
            stamp = source["frame_times_s"][index] if document["schema_version"] >= 3 else index / source["fps"]
            _require(type(frame) is dict and frame.get("frame") == index and frame.get("time_s") == stamp
                     and frame.get("observation_indices") == row_indices[index], "mask frame registry/clock differs")
            slots = []
            for name in ids:
                row, observed = indexed[index, name], construction[index, name]
                slot = {"id": row.get("id"), "raw_object_id": name, "status": "withheld", "box": None, "mask": None}
                if observed["admission"] != "included":
                    withheld.append(row.get("id"))
                    slots.append(slot)
                    continue
                record = records.get(row.get("id"), {})
                payload = record.get("payload")
                _require({"document_sha256": source_hash, "evidence_id": record.get("id")} in observed["evidence_refs"]
                         and record.get("object_id") == name
                         and record.get("frame") == index and record.get("time_s") == row.get("time_s") == stamp
                         and record.get("kind") == "model_segmentation_proposal"
                         and record.get("status") in ("inferred", "unknown")
                         and type(payload) is dict and canonical_hash(payload) == row.get("raw_observation_sha256"),
                         "mask row lost its originating source observation")
                _require(all(row.get(key) == payload.get(key) for key in
                             ("id", "frame", "time_s", "object_id", "raw_model_id", "phase", "proposal_status",
                              "identity", "confidence", "original_source")) and row.get("box") == payload.get("geometry"),
                         "mask metadata changed from original provider evidence")
                if row.get("phase") == "pre_initialization":
                    slot["status"] = "pre_initialization"
                elif row.get("mask_status") != "decoded" or row.get("proposal_status") != "returned":
                    slot["status"] = "missing"
                else:
                    _require(record.get("status") == "inferred", "available mask cannot have unknown evidence status")
                    _require(row.get("encoded_sha256") == payload.get("mask_provenance", {}).get("encoded_sha256"),
                             "encoded mask identity differs")
                    shape = row.get("raster_dimensions")
                    _require(type(shape) is list and len(shape) == 2
                             and all(type(n) is int and 1 <= n <= 4096 for n in shape)
                             and math.prod(shape) <= 8_847_360, "bounded native raster dimensions required")
                    raw = raster_loader(deepcopy(row))
                    _check(deadline)
                    _require(type(raw) is bytes and len(raw) == math.prod(shape)
                             and sha256(raw).hexdigest() == row.get("raster_sha256"), "exact original binary mask required")
                    slot.update(status="available", box=row["box"]["box_xyxy_halfopen"],
                                mask=np.frombuffer(raw, np.uint8).reshape(shape))
                    loaded.append({"id": row["id"], "sha256": row["raster_sha256"], "bytes": len(raw)})
                slots.append(slot)
            yield {"frame": index, "time_s": stamp, "slots": slots}

    result = track_mask_hypotheses(stream(), width=document["image"]["width_px"], height=document["image"]["height_px"],
                                   maximum_seconds=max(1e-9, deadline - time.monotonic()))
    _check(deadline)
    result.update(source_custody_verified=False, source_observation_binding_verified=True,
                  decoded_raster_hashes_verified=True, encoded_to_raster_derivation_verified=False,
                  raster_provenance_scope="Declared decoded raster hashes verified; encoded-mask decoding is not independently repeated",
                  source_document_sha256=source_hash,
                  mask_document_sha256=masks_hash, submitted_video_sha256=source["sha256"],
                  clock_scope="Unchanged submitted-evidence clock; original/native mapping is not requalified here",
                  loaded_rasters=loaded, withheld_slots=withheld)
    return result
