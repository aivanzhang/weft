"""Versioned scene evidence interchange; adapters do not expand perception capability."""

from copy import deepcopy
import json
import math
from pathlib import Path
import re

from weft.io import canonical_hash, digest


PROFILE = "scene-observations-v1"
STATUSES = {"observed", "inferred", "assumed", "unknown"}
PARTITIONS = {"generation", "development_validation", "audit", "construction", "unassigned", "unsampled"}
VISIBILITY = {"detected", "partial", "not_detected", "not_sampled", "unknown"}
MAX_BYTES = 64_000_000
COORDINATES = {
    "image": {"unit": "pixels", "axes": ["right", "down"], "metric_calibrated": False},
    "pose_depth": {"unit": "image_widths", "axes": ["model_relative_depth"], "metric_calibrated": False},
    "model_world": {"unit": "model_meters", "axes": ["model_x", "model_y", "model_z"], "metric_calibrated": False},
    "joint": {"unit": "radians", "axes": ["unsigned_bend"], "metric_calibrated": False},
}


def _finite(value, name, low=-1e6, high=1e6):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be a finite number in [{low}, {high}]")


def _identifier(value, name):
    if not isinstance(value, str) or not 1 <= len(value) <= 240:
        raise ValueError(f"{name} must be a bounded nonempty string")


def _claim(value, status="unknown", evidence_ids=()):
    return {"value": value, "status": status, "evidence_ids": list(evidence_ids)}


def _measurement(value, system="image", status="observed"):
    return {"value": value, "coordinate_system": system, "unit": COORDINATES[system]["unit"], "status": status}


def validate_observations(document):
    """Reject malformed evidence, reference drift, temporal conflicts, and mixed units."""
    if not isinstance(document, dict) or document.get("profile") != PROFILE or type(document.get("schema_version")) is not int or document["schema_version"] not in {1, 2, 3}:
        raise ValueError("Unsupported scene observations schema/profile")
    try:
        encoded = json.dumps(document, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Scene observations must contain finite JSON values") from error
    if len(encoded.encode()) > MAX_BYTES:
        raise ValueError("Scene observations exceed 64 MB")
    required = {"schema_version", "profile", "source", "time", "image", "coordinates", "camera", "objects",
                "evidence", "observations", "relationships", "provenance", "limits"}
    if set(document) != required:
        raise ValueError("Scene observations fields differ from schema")
    source = document["source"]
    _validate_source(source, document["schema_version"])
    time_contract = ({"unit": "seconds", "origin": "submitted_video_start", "basis": "decoder_presentation_timestamps", "status": "observed"}
                     if document['schema_version'] == 3 else
                     {"unit": "seconds", "origin": "video_start", "basis": "frame_index_divided_by_fps", "status": "assumed"})
    if document["time"] != time_contract:
        raise ValueError("Unsupported or conflicting time units/basis")
    image = document["image"]
    if not isinstance(image, dict) or set(image) != {"width_px", "height_px"}:
        raise ValueError("Image plane requires explicit pixel dimensions")
    for name in image:
        if type(image[name]) is not int or not 1 <= image[name] <= 4096:
            raise ValueError("Invalid image plane dimensions")
    if document['schema_version'] == 3 and any(image[k] != source[k] for k in ('width_px', 'height_px')):
        raise ValueError('Image dimensions differ from the submitted video')
    if document["coordinates"] != COORDINATES:
        raise ValueError("Conflicting coordinate units, axes, or metric calibration")
    objects, evidence, observations = document["objects"], document["evidence"], document["observations"]
    if not isinstance(objects, list) or not 1 <= len(objects) <= 128:
        raise ValueError("Need 1–128 scene objects")
    object_ids = set()
    for obj in objects:
        if not isinstance(obj, dict) or set(obj) != {"id", "label", "properties", "unknowns"}:
            raise ValueError("Object fields differ from schema")
        _identifier(obj["id"], "Object id")
        _identifier(obj["label"], "Object label")
        if obj["id"] in object_ids:
            raise ValueError("Duplicate object identity")
        object_ids.add(obj["id"])
        if not isinstance(obj["properties"], dict) or not {"kind", "role"} <= set(obj["properties"]):
            raise ValueError("Object kind and role require explicit claims")
        if not isinstance(obj["unknowns"], list) or any(not isinstance(v, str) for v in obj["unknowns"]):
            raise ValueError("Object unknowns must be text")
    if not isinstance(evidence, list) or len(evidence) > 40_000:
        raise ValueError("Evidence registry must contain at most 40000 records")
    evidence_ids, evidence_by_id = set(), {}
    for record in evidence:
        if not isinstance(record, dict) or set(record) != {"id", "frame", "time_s", "object_id", "kind", "status", "payload"}:
            raise ValueError("Evidence fields differ from schema")
        _identifier(record["id"], "Evidence id")
        if not record["id"].startswith(source["sha256"] + ":") or record["id"] in evidence_ids:
            raise ValueError("Duplicate or foreign source evidence identity")
        _validate_time(record, source)
        if document["schema_version"] in (2, 3) and not re.match(rf"^{source['sha256']}:frame:{record['frame']}(?::|$)", record["id"]):
            raise ValueError("Interval evidence identity must preserve its absolute source frame")
        if record["object_id"] not in object_ids:
            raise ValueError("Evidence references a dangling object")
        if record["status"] not in STATUSES or not isinstance(record["payload"], dict):
            raise ValueError("Evidence status and original payload are required")
        payload = record["payload"]
        if payload.get("id", record["id"]) != record["id"] or payload.get("frame", record["frame"]) != record["frame"]:
            raise ValueError("Evidence identity/frame conflicts with its original payload")
        if "time_s" in payload and not math.isclose(payload["time_s"], record["time_s"], abs_tol=1e-6):
            raise ValueError("Evidence time conflicts with its original payload")
        _identifier(record["kind"], "Evidence kind")
        evidence_ids.add(record["id"])
        evidence_by_id[record["id"]] = record

    def references(ids):
        if not isinstance(ids, list) or any(not isinstance(identifier, str) for identifier in ids) or len(set(ids)) != len(ids) or any(identifier not in evidence_ids for identifier in ids):
            raise ValueError("Dangling or duplicate evidence reference")

    def claim(value):
        if not isinstance(value, dict) or set(value) != {"value", "status", "evidence_ids"} or value["status"] not in STATUSES:
            raise ValueError("A claim needs value, evidence status, and references")
        references(value["evidence_ids"])
        if value["status"] == "unknown" and value["value"] is not None:
            raise ValueError("Unknown claims cannot assert a value")
        if value["status"] in {"observed", "inferred"} and not value["evidence_ids"]:
            raise ValueError("Observed/inferred claims need evidence references")

    for obj in objects:
        for value in obj["properties"].values():
            claim(value)
        if obj["properties"]["role"]["value"] not in {None, "active", "passive", "static"}:
            raise ValueError("Unknown object role")
    if not isinstance(document["camera"], dict) or set(document["camera"]) != {"motion", "calibration", "screen"}:
        raise ValueError("Camera fields differ from schema")
    claim(document["camera"]["motion"])
    claim(document["camera"]["calibration"])
    if not isinstance(observations, list) or len(observations) > 1800 * 128:
        raise ValueError("Observation count exceeds the frame/object envelope")
    previous, keys, observation_ids = -1, set(), set()
    for row in observations:
        if not isinstance(row, dict) or set(row) != {"id", "frame", "time_s", "object_id", "visibility", "partition", "evidence_ids", "measurements"}:
            raise ValueError("Observation fields differ from schema")
        _validate_time(row, source)
        if row["frame"] < previous:
            raise ValueError("Observation time order must not decrease")
        previous = row["frame"]
        key = (row["frame"], row["object_id"])
        if row["object_id"] not in object_ids or key in keys:
            raise ValueError("Dangling object or duplicate object/frame observation")
        keys.add(key)
        if row["id"] != f"{source['sha256']}:observation:{row['frame']}:{row['object_id']}" or row["id"] in observation_ids:
            raise ValueError("Observation identity conflicts with source/frame/object")
        observation_ids.add(row["id"])
        if row["visibility"] not in VISIBILITY or row["partition"] not in PARTITIONS:
            raise ValueError("Invalid visibility or evidence partition")
        references(row["evidence_ids"])
        if any(evidence_by_id[identifier]["frame"] != row["frame"] or evidence_by_id[identifier]["object_id"] != row["object_id"] for identifier in row["evidence_ids"]):
            raise ValueError("Observation evidence references another frame/object")
        if not isinstance(row["measurements"], dict):
            raise ValueError("Measurements must be a named mapping")
        if row["visibility"] in {"not_detected", "not_sampled", "unknown"} and row["measurements"]:
            raise ValueError("Unavailable observations cannot fabricate measurements")
        if row["measurements"] and not row["evidence_ids"]:
            raise ValueError("Measurements require original evidence")
        for name, measurement in row["measurements"].items():
            _validate_measurement(name, measurement)
    relationships = document["relationships"]
    if not isinstance(relationships, list) or len(relationships) > 1000:
        raise ValueError("Relationships must be a bounded list")
    relationship_ids = set()
    for relationship in relationships:
        if not isinstance(relationship, dict) or set(relationship) != {"id", "type", "objects", "status", "evidence_ids", "interval_s", "attributes"}:
            raise ValueError("Relationship fields differ from schema")
        _identifier(relationship["id"], "Relationship id")
        _identifier(relationship["type"], "Relationship type")
        if relationship["id"] in relationship_ids:
            raise ValueError("Duplicate relationship identity")
        relationship_ids.add(relationship["id"])
        participants = relationship["objects"]
        if not isinstance(participants, list) or not participants or len(set(participants)) != len(participants) or any(p not in object_ids for p in participants):
            raise ValueError("Relationship references dangling or repeated objects")
        claim({"value": None if relationship["status"] == "unknown" else relationship["type"],
               "status": relationship["status"], "evidence_ids": relationship["evidence_ids"]})
        if any(evidence_by_id[identifier]["object_id"] not in participants for identifier in relationship["evidence_ids"]):
            raise ValueError("Relationship cites evidence for unrelated objects")
        interval = relationship["interval_s"]
        if interval is not None:
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError("Relationship interval requires start/end seconds")
            first, last = _decoded_bounds(source)
            time_bounds = ((source['frame_times_s'][first], source['frame_times_s'][last])
                           if document['schema_version'] == 3 else
                           (first / source['fps'], (last if document['schema_version'] == 2 else last + 1) / source['fps']))
            for value in interval:
                _finite(value, "relationship time", *time_bounds)
            if interval[1] < interval[0]:
                raise ValueError("Relationship end precedes its start")
        if not isinstance(relationship["attributes"], dict):
            raise ValueError("Relationship attributes must be a mapping")
    if not isinstance(document["limits"], list) or any(not isinstance(v, str) for v in document["limits"]):
        raise ValueError("Limits must be explicit text")
    return deepcopy(document)


def _validate_source(source, version):
    fields = {"sha256", "width_px", "height_px", "fps", "metadata_frames", "decoded_frames"}
    if version == 2:
        fields |= {"decode_interval", "source_complete", "prefix_advanced_frames"}
    if version == 3:
        fields = (fields - {'fps'}) | {'frame_times_s'}
    if not isinstance(source, dict) or set(source) != fields:
        raise ValueError("Source fields differ from schema")
    if not isinstance(source["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", source["sha256"]):
        raise ValueError("Source requires a SHA-256 identity")
    for name, high in (("width_px", 4096), ("height_px", 4096 if version == 3 else 2160),
                       ("metadata_frames", 216000 if version == 2 else 1800), ("decoded_frames", 1800)):
        if type(source[name]) is not int or not 1 <= source[name] <= high:
            raise ValueError(f"Invalid source {name}")
    if version == 3:
        times = source['frame_times_s']
        if (not isinstance(times, list) or len(times) != source['decoded_frames']
                or source['metadata_frames'] != source['decoded_frames']):
            raise ValueError('Decoder timestamp count must match complete submitted-video frame counts')
        previous = -1.
        for stamp in times:
            _finite(stamp, 'decoder presentation time', 0, 60)
            if stamp <= previous:
                raise ValueError('Decoder presentation timestamps must strictly increase')
            previous = stamp
    else:
        _finite(source["fps"], "fps", 1, 240)
    if version == 2:
        interval = source["decode_interval"]
        if (not isinstance(interval, list) or len(interval) != 2 or any(type(value) is not int for value in interval)
                or not 0 <= interval[0] <= 20000 or not interval[0] <= interval[1] <= 21799
                or interval[1] >= source["metadata_frames"]):
            raise ValueError("Invalid decoded source interval")
        if source["decoded_frames"] != interval[1] - interval[0] + 1:
            raise ValueError("Decoded frames must equal the inclusive selected interval count")
        if (source["source_complete"] is not False or type(source["prefix_advanced_frames"]) is not int
                or source["prefix_advanced_frames"] != interval[0]):
            raise ValueError("Interval source completeness/prefix conflicts with the selected interval")
        if source["metadata_frames"] / source["fps"] > 3600 or source["decoded_frames"] / source["fps"] > 60:
            raise ValueError("Source or selected interval exceeds its duration bound")


def _decoded_bounds(source):
    return source.get("decode_interval", (0, source["decoded_frames"] - 1))


def _validate_time(row, source):
    frame = row["frame"]
    first, last = _decoded_bounds(source)
    if type(frame) is not int or not first <= frame <= last:
        raise ValueError("Frame lies outside decoded source")
    _finite(row["time_s"], "observation time", 0, 3600 if "decode_interval" in source else 1800)
    expected = source['frame_times_s'][frame] if 'frame_times_s' in source else frame / source['fps']
    if not math.isclose(row["time_s"], expected, abs_tol=1e-6):
        raise ValueError("Frame/time mismatch under declared time basis")


def _validate_measurement(name, value):
    systems = {"center": "image", "radius": "image", "box": "image", "landmarks": "image",
               "landmark_depth": "pose_depth", "world_landmarks": "model_world", "joint_angles": "joint"}
    if name not in systems or not isinstance(value, dict) or set(value) != {"value", "coordinate_system", "unit", "status"}:
        raise ValueError("Unknown or malformed measurement")
    system = systems[name]
    if value["coordinate_system"] != system or value["unit"] != COORDINATES[system]["unit"] or value["status"] not in {"observed", "inferred"}:
        raise ValueError("Conflicting measurement coordinate units/status")
    data = value["value"]
    if name in {"world_landmarks", "landmark_depth", "landmarks", "joint_angles", "box", "radius"} and value["status"] != "inferred":
        raise ValueError("Derived geometry and model outputs must remain inferred")
    if name in {"center", "box"}:
        if not isinstance(data, list) or len(data) != (2 if name == "center" else 4):
            raise ValueError("Invalid image coordinate shape")
        for coordinate in data:
            _finite(coordinate, name)
        if name == "box" and min(data[2:]) <= 0:
            raise ValueError("Box extents must be positive")
    elif name == "radius":
        _finite(data, "radius", 0.001, 4096)
    elif name == "joint_angles":
        if not isinstance(data, dict) or len(data) > 128:
            raise ValueError("Joint angles must be bounded")
        for angle in data.values():
            _finite(angle, "joint angle", -math.pi * 2, math.pi * 2)
    else:
        if not isinstance(data, list) or len(data) > 512:
            raise ValueError("Landmarks must be bounded")
        indices = set()
        dimensions = 1 if name == "landmark_depth" else 3 if name == "world_landmarks" else 2
        for point in data:
            if not isinstance(point, dict) or set(point) != {"index", "position", "visibility", "presence"}:
                raise ValueError("Landmark fields differ from schema")
            if type(point["index"]) is not int or not 0 <= point["index"] < 512 or point["index"] in indices:
                raise ValueError("Invalid or duplicate landmark index")
            indices.add(point["index"])
            if not isinstance(point["position"], list) or len(point["position"]) != dimensions:
                raise ValueError("Landmark coordinate dimension mismatch")
            for coordinate in point["position"]:
                _finite(coordinate, "landmark coordinate")
            for confidence in (point["visibility"], point["presence"]):
                if confidence is not None:
                    _finite(confidence, "landmark confidence", 0, 1)


def _base(evidence, profile):
    source = {"sha256": evidence["source_sha256"], "width_px": evidence.get("source_width", evidence["width"]),
              "height_px": evidence.get("source_height", evidence["height"]), "fps": evidence["fps"],
              "metadata_frames": evidence["frame_count"], "decoded_frames": evidence.get("decoded_frames", evidence["frame_count"])}
    audit = evidence.get("decode_audit", {})
    version = 2 if audit.get("scope") == "selected_interval" else 1
    if version == 2:
        if audit.get("selected_frames_complete") is not True:
            raise ValueError("Selected interval decoding must be complete before interchange")
        source.update(decode_interval=deepcopy(audit.get("selected_interval")),
                      source_complete=audit.get("source_complete"), prefix_advanced_frames=audit.get("prefix_advanced_frames"))
    _validate_source(source, version)
    return {"schema_version": version, "profile": PROFILE, "source": source,
            "time": {"unit": "seconds", "origin": "video_start", "basis": "frame_index_divided_by_fps", "status": "assumed"},
            "image": {"width_px": evidence["width"], "height_px": evidence["height"]},
            "coordinates": deepcopy(COORDINATES), "camera": {"motion": _claim("fixed", "assumed"),
              "calibration": _claim(None), "screen": deepcopy(evidence.get("camera_check"))},
            "objects": [], "evidence": [], "observations": [], "relationships": [],
            "provenance": {"adapter": profile, "input_evidence_sha256": canonical_hash(evidence),
               "source_metadata": {k: deepcopy(v) for k, v in evidence.items()
                  if k not in {"generation", "development_validation", "audit_episodes", "frames", "visibility", "reference"}}},
            "limits": ["Interchange only: no new detections, geometry, physics, or general-video capability are established.",
                       "Generation, development, and audit access must remain separate in downstream repair prompts.",
                       "Missing detections do not establish absence, and model-derived coordinates are not calibrated measurements."] +
                      deepcopy(evidence.get("limits", evidence.get("assumptions", [])))}


def _object(identifier, label, kind=None, role=None, refs=(), role_status="assumed"):
    # Appearance configs spell unknown explicitly; interchange represents it without a claim.
    kind = None if kind == 'unknown' else kind
    role = None if role == 'unknown' else role
    return {"id": identifier, "label": label, "properties": {
                "kind": _claim(kind, "assumed" if kind else "unknown"),
                "role": _claim(role, role_status if role else "unknown", refs if role_status == "inferred" else ())},
            "unknowns": ["metric scale", "true mass", "unobserved contacts", "physical response to new actions"]}


def _record(document, raw, object_id, kind, status="observed", suffix=None):
    frame = raw["frame"]
    identifier = raw.get("id") or f"{document['source']['sha256']}:frame:{frame}:{kind}:{object_id}"
    if suffix:
        identifier += ":" + suffix
    record = {"id": identifier, "frame": frame, "time_s": raw.get("time_s", frame / document["source"]["fps"]),
              "object_id": object_id, "kind": kind, "status": status, "payload": deepcopy(raw)}
    document["evidence"].append(record)
    return identifier


def _observation(document, frame, object_id, visibility, partition, refs=(), measurements=None):
    return {"id": f"{document['source']['sha256']}:observation:{frame}:{object_id}", "frame": frame,
            "time_s": frame / document["source"]["fps"], "object_id": object_id, "visibility": visibility,
            "partition": partition, "evidence_ids": list(refs), "measurements": measurements or {}}


def _partition_rows(evidence):
    result, partitions = {}, {}
    groups = [("generation", evidence.get("generation", [])), ("development_validation", evidence.get("development_validation", []))]
    groups += [("audit", group) for group in evidence.get("audit_episodes", [])]
    for partition, rows in groups:
        for row in rows:
            if row["id"] in result:
                raise ValueError("Evidence identity appears in multiple partitions")
            result[row["id"]] = row
            partitions[row["id"]] = partition
    return result, partitions


def _track_measurements(row):
    measurements = {"center": _measurement([row["x_px"], row["y_px"]])}
    if "radius_px" in row:
        measurements["radius"] = _measurement(row["radius_px"], status="inferred")
    if "width_px" in row and "height_px" in row:
        width, height = row["width_px"], row["height_px"]
        measurements["box"] = _measurement([row["x_px"] - width / 2, row["y_px"] - height / 2, width, height], status="inferred")
    return measurements


def adapt_tracks(evidence, *, profile, tracks=None, scene=None, graph=None):
    """Adapt rolling/planar tracks; planar full observations are mandatory."""
    if profile not in {"rolling", "planar"}:
        raise ValueError("Track adapter supports rolling or planar")
    document = _base(evidence, profile)
    rows, partitions = _partition_rows(evidence)
    if profile == "planar":
        if not isinstance(tracks, dict) or not isinstance(tracks.get("observations"), list):
            raise ValueError("Planar adaptation requires tracked-objects.json to preserve excluded-frame measurements")
        track_ids = set()
        for row in tracks["observations"]:
            if row["id"] in track_ids:
                raise ValueError("Duplicate tracked-object evidence identity")
            track_ids.add(row["id"])
            if row["id"] in rows and row != rows[row["id"]]:
                raise ValueError("Conflicting records share an evidence identity")
            rows[row["id"]] = row
    by_object = {}
    for row in rows.values():
        by_object.setdefault(row["object_id"], []).append(row["id"])
    model_objects = {obj["id"]: obj for obj in (scene or {}).get("objects", [])}
    catalog = evidence.get("objects", [{"id": key, "label": key} for key in by_object])
    known = set()
    for obj in catalog:
        identifier = obj["id"]
        model = model_objects.get(identifier, obj)
        role = model.get("role")
        explicit = next((item for item in evidence.get("config", {}).get("objects", []) if item["id"] == identifier), {})
        role_status = "assumed" if role == "active" or "role" in explicit else "inferred"
        document["objects"].append(_object(identifier, obj.get("label", identifier), model.get("shape", "rigid_circle" if profile == "rolling" else None),
                                            role, by_object.get(identifier, ()), role_status))
        known.add(identifier)
    for obj in (graph or {}).get("objects", []):
        if obj["id"] not in known:
            document["objects"].append(_object(obj["id"], obj["id"], obj.get("kind"), "static" if obj.get("state") == "fixed" else None))
            known.add(obj["id"])
    observations = {}
    for row in sorted(rows.values(), key=lambda item: (item["frame"], item["object_id"])):
        authored = row.get("status") in {"initialized", "pre_initialization"}
        identifier = _record(document, row, row["object_id"], "image_track", "unknown" if authored else "observed")
        key = (row["frame"], row["object_id"])
        if key in observations:
            raise ValueError("Multiple tracks for one object/frame")
        visibility = "not_sampled" if row.get("status") == "pre_initialization" else "unknown" if authored else "detected"
        observations[key] = _observation(document, row["frame"], row["object_id"], visibility,
                                         partitions.get(identifier, "unassigned"), [identifier], {} if authored else _track_measurements(row))
    default_id = next(iter(by_object), None)
    for raw in evidence.get("visibility", []):
        object_id = raw.get("object_id", default_id)
        authored = raw["status"] in {"initialized", "pre_initialization"}
        identifier = _record(document, raw, object_id, "visibility", "unknown" if authored else "observed")
        key = (raw["frame"], object_id)
        if key in observations:
            observations[key]["evidence_ids"].append(identifier)
        else:
            visible = ("not_sampled" if raw["status"] == "pre_initialization" else
                       "unknown" if raw["status"] in {"tracked", "measured", "initialized"} else "not_detected")
            observations[key] = _observation(document, raw["frame"], object_id, visible, "unassigned", [identifier])
    document["observations"] = [row for _, row in sorted(observations.items())]
    for candidate in evidence.get("contact_candidates", []):
        document["relationships"].append({"id": f"contact-{len(document['relationships'])}", "type": "projected_contact_candidate",
            "objects": candidate["objects"], "status": "inferred", "evidence_ids": candidate["evidence_ids"],
            "interval_s": [candidate["time_s"], candidate["time_s"]], "attributes": {"verified_contact": False}})
    for relation in (scene or {}).get("relationships", []):
        if any(row["type"] == relation["type"] and row["objects"] == relation["objects"] for row in document["relationships"]):
            continue
        document["relationships"].append({"id": f"scene-{len(document['relationships'])}", "type": relation["type"],
            "objects": relation["objects"], "status": "assumed" if relation.get("status") == "assumed" else "unknown",
            "evidence_ids": [], "interval_s": None,
            "attributes": {"original_claim": deepcopy(relation), "note": "No referenced source observations were supplied for this scene relationship"}})
    for relation in (graph or {}).get("relationships", []):
        document["relationships"].append({"id": f"graph-{len(document['relationships'])}", "type": relation["relation"],
            "objects": [relation["subject"], relation["object"]], "status": "inferred",
            "evidence_ids": (graph or {}).get("evidence_ids", []), "interval_s": None, "attributes": deepcopy(relation)})
    document["provenance"]["scene"] = deepcopy(scene)
    document["provenance"]["graph"] = deepcopy(graph)
    if profile == "rolling":
        document["limits"].append("Legacy rolling evidence omits detections outside retained episodes; visibility records without retained geometry remain unknown.")
    document["limits"].append("Track boxes are inferred centroid-centered extents, not preserved segmentation masks or exact contour bounds.")
    return validate_observations(document)


def adapt_exercise(evidence, *, scene=None):
    """Retain pose outputs and confidence without upgrading them to anatomy measurements."""
    document = _base(evidence, "exercise")
    document["objects"] = [_object("person-0", "Tracked person", "person", "active" if scene else None)]
    frames = {frame["frame"]: frame for frame in evidence["frames"]}
    if len(frames) != len(evidence["frames"]):
        raise ValueError("Duplicate exercise frame")
    width, height = evidence["width"], evidence["height"]
    first, last = _decoded_bounds(document["source"])
    for index in range(first, last + 1):
        raw = frames.get(index)
        if raw is None:
            document["observations"].append(_observation(document, index, "person-0", "not_sampled", "unsampled"))
            continue
        identifier = _record(document, raw, "person-0", "pose_model_output", "inferred")
        present = bool(raw.get("landmarks"))
        visibility = "partial" if present and raw.get("status") != "tracked" else "detected" if present else "not_detected"
        measurements = {}
        if present:
            def points(landmarks, axes):
                return [{"index": i, "position": [landmark[key] * scale for key, scale in axes],
                         "visibility": landmark.get("visibility"), "presence": landmark.get("presence")}
                        for i, landmark in enumerate(landmarks)]
            measurements["landmarks"] = _measurement(points(raw["landmarks"], [("x", width), ("y", height)]), status="inferred")
            measurements["landmark_depth"] = _measurement(points(raw["landmarks"], [("z", 1)]), "pose_depth", "inferred")
            if raw.get("world_landmarks"):
                measurements["world_landmarks"] = _measurement(points(raw["world_landmarks"], [("x", 1), ("y", 1), ("z", 1)]), "model_world", "inferred")
            if raw.get("joints"):
                measurements["joint_angles"] = _measurement(deepcopy(raw["joints"]), "joint", "inferred")
        document["observations"].append(_observation(document, index, "person-0", visibility, "construction", [identifier], measurements))
    if scene:
        for side in ("left", "right"):
            for joint in ("shoulder", "elbow", "hip", "knee"):
                document["relationships"].append({"id": f"joint-{side}-{joint}", "type": "internal_hinge_joint",
                    "objects": ["person-0"], "status": "assumed", "evidence_ids": [], "interval_s": None,
                    "attributes": {"name": f"{side}_{joint}", "scope": "Authored simulator topology, not recovered joint mechanics"}})
        document["relationships"].append({"id": "anchored-pelvis", "type": "anchored_support",
            "objects": ["person-0"], "status": "assumed", "evidence_ids": [], "interval_s": None,
            "attributes": {"scope": "Authored fixed pelvis/torso, not observed support or ground contact"}})
    document["provenance"]["scene"] = deepcopy(scene)
    document["limits"].append("Model-world meters and normalized depth are uncalibrated model estimates; muscle activation, mass, force and ground contacts remain unknown.")
    return validate_observations(document)


def adapt_appearance(tracks):
    """Expose failed appearance traces without constructing scene or physics claims."""
    if (not isinstance(tracks, dict) or tracks.get("profile") != "appearance-tracks-v1"
            or type(tracks.get("schema_version")) is not int or tracks["schema_version"] != 1):
        raise ValueError("Unsupported appearance trace schema/profile")
    if not isinstance(tracks.get("observations"), list) or not isinstance(tracks.get("objects"), list):
        raise ValueError("Appearance trace requires objects and original observations")
    metadata_keys = ("source_sha256", "width", "height", "source_width", "source_height", "fps",
                     "frame_count", "decoded_frames", "decode_audit", "camera_check", "limits")
    evidence = {key: deepcopy(tracks[key]) for key in metadata_keys if key in tracks}
    evidence["objects"] = deepcopy(tracks["objects"])
    # ROI roles and shapes are authored even when a passive object was successfully tracked.
    evidence["config"] = {"objects": deepcopy(tracks["objects"])}
    measured, visibility = [], []
    for row in tracks["observations"]:
        if not isinstance(row, dict) or not isinstance(row.get("status"), str):
            raise ValueError("Appearance observations require their original status")
        (measured if row["status"] == "tracked" else visibility).append(deepcopy(row))
    evidence["visibility"] = visibility
    document = adapt_tracks(evidence, profile="planar", tracks={"observations": measured})
    document["camera"]["motion"] = _claim(None)
    document["provenance"].update(adapter="appearance", input_appearance_sha256=canonical_hash(tracks),
                                   appearance_metadata={key: deepcopy(value) for key, value in tracks.items()
                                                        if key not in {"observations", "frames", "objects"}})
    document["limits"].append("Appearance-only trace: object roles, labels and shapes are authored; no scene, world geometry, contacts or physical response are established.")
    return validate_observations(document)


def adapt_run(directory):
    """Load existing artifacts read-only; no inference, training, or file mutation."""
    directory = Path(directory).resolve()
    if ((directory / "run" / "evidence.json").exists()
            or (directory / "run" / "appearance" / "appearance-tracks.json").exists()):
        directory = directory / "run"
    files = {}

    def load(name, required=False):
        path = directory / name
        if not path.resolve().is_relative_to(directory):
            raise ValueError("Input artifact escapes the run directory")
        if not path.is_file():
            if required:
                raise ValueError(f"Required observations input missing: {name}")
            return None
        if path.stat().st_size > MAX_BYTES:
            raise ValueError(f"Input artifact exceeds 64 MB: {name}")
        value = json.loads(path.read_text())
        files[name] = digest(path)
        return value

    evidence_path = directory / "evidence.json"
    appearance_path = directory / "appearance" / "appearance-tracks.json"
    evidence = load("evidence.json", required=(evidence_path.exists() or evidence_path.is_symlink()
                                               or not (appearance_path.exists() or appearance_path.is_symlink())))
    profile = evidence.get("profile") if evidence is not None else None
    if evidence is None:
        document = adapt_appearance(load("appearance/appearance-tracks.json", required=True))
    elif profile == "exercise-pose-v1":
        document = adapt_exercise(evidence, scene=load("scene.json"))
    elif profile == "planar-rigid-v1":
        document = adapt_tracks(evidence, profile="planar", tracks=load("tracked-objects.json", required=True), scene=load("scene.json"))
    elif profile is None and "audit_episodes" in evidence:
        document = adapt_tracks(evidence, profile="rolling", graph=load("graph.json"))
    else:
        raise ValueError("No observations adapter for this input profile")
    document["provenance"]["input_artifacts"] = files
    return validate_observations(document)


def load_observations(path):
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Scene observations exceed 64 MB")
    return validate_observations(json.loads(path.read_text()))
