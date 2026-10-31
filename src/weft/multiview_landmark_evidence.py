"""Bound original image observations to conditional multi-view point estimates."""

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import re
import time

from weft.exercise_support_evidence import _json_size
from weft.camera_rotation_prior import CONTRACT as ROTATION_CONTRACT, PROFILE as ROTATION_PROFILE, prepare_camera_rotation_prior
from weft.io import canonical_hash
from weft.multiray_pose import CONTRACT as FIT_CONTRACT, triangulate_pixels
from weft.multiview_pose import CONTRACT as RAY_CONTRACT, _observation, _view
from weft.world_alignment import load_world_alignment


PROFILE = "calibrated-multiview-landmark-evidence-v1"
MAX_INPUT_BYTES, MAX_DOCUMENT_BYTES = 32 << 20, 8 << 20
MAX_OUTPUT_BYTES, MAX_ROW_BYTES, MAX_METADATA_BYTES = 8 << 20, 128 << 10, 128 << 10
MAX_SECONDS, MAX_FRAMES, MAX_POSES, MAX_FITS = 10., 1800, 900, 6000
CONSTRUCTION_FRACTION = .6
_IDENTITY_SPACE = {"scale": 1., "rotation_wxyz": [1., 0., 0., 0.], "translation": [0., 0., 0.]}
_IDENTITY_CLOCK = {"rate": 1., "source_origin_s": 0., "target_origin_s": 0.}


def _require(value, message):
    if not value:
        raise ValueError("Multiview landmark evidence: " + message)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Multiview landmark evidence exceeded its cooperative deadline")


def _number(value, low, high):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def _unique(pairs):
    value = {}
    for key, item in pairs:
        _require(key not in value, "duplicate JSON field")
        value[key] = item
    return value


def _invalid(_):
    raise ValueError("Multiview landmark evidence requires finite JSON")


def _decode(raw, deadline, *, camera=False, alignment=False):
    _check(deadline)
    _require(type(raw) is bytes and 0 < len(raw) <= (16384 if alignment else MAX_DOCUMENT_BYTES),
             "exact bounded input bytes required")
    try:
        value = json.loads(raw, object_pairs_hook=_unique, parse_constant=_invalid)
    except (UnicodeError, RecursionError, ValueError) as error:
        raise ValueError("Multiview landmark evidence requires strict UTF-8 JSON") from error
    _require(type(value) is dict, "input document must be an object")
    if not alignment:
        maximum = MAX_FRAMES if camera else MAX_POSES
        _require(type(value.get("frames")) is list and 1 <= len(value["frames"]) <= maximum,
                 "source row envelope exceeded before canonical hashing")
    size = _json_size(value, deadline)
    _require(size <= MAX_DOCUMENT_BYTES, "canonical document exceeds its byte bound")
    pin = {"sha256": sha256(raw).hexdigest(), "canonical_sha256": canonical_hash(value), "bytes": len(raw)}
    _check(deadline)
    return value, pin, size


def _encoded(value, maximum, deadline):
    chunks, size = [], 0
    for chunk in json.JSONEncoder(allow_nan=False, sort_keys=True, separators=(",", ":")).iterencode(value):
        _check(deadline)
        raw = chunk.encode()
        size += len(raw)
        _require(size <= maximum, "prepared output exceeds its byte bound; no slots were dropped")
        chunks.append(raw)
    return b"".join(chunks)


def _world(camera_pin):
    return {"id": "camera-world:" + camera_pin["canonical_sha256"],
            "length_unit": "m", "axes": "x_right_y_depth_z_up"}


def _binding(value, camera_pin, primary_pin, registry):
    if value is None:
        return "missing_alignment"
    alignment = load_world_alignment(value, evidence_ids_by_document=registry)
    if not alignment.ready:
        return "unknown_alignment"
    if (value["from_frame"] != _world(camera_pin) or value["to_frame"] != _world(primary_pin)
            or value["spatial"] != _IDENTITY_SPACE or value["clock"] != _IDENTITY_CLOCK):
        return "incompatible_same_world_clock_alignment"
    return "conditional_supplied_identity_alignment"


def _prepared_camera(value):
    matrix, radial, rotation, translation, width, height, _, _ = value
    return {"matrix": matrix, "radial": radial, "rotation": rotation, "translation": translation,
            "width": width, "height": height}


def _image_observation(view, frame, joint, construction):
    pose = view["poses"].get(frame)
    item = {"name": view["name"], "source_evidence_id": None if pose is None else pose["id"],
            "decoder_timestamp_s": None if pose is None else pose["decoder_timestamp_s"],
            "pose_status": None if pose is None else pose.get("status"),
            "pose_count": None if pose is None else pose.get("pose_count"),
            "identity_status": None if pose is None else pose.get("identity_status"),
            "model_visibility": None, "model_presence": None, "pixel": None}
    if frame >= view["decoded"]:
        status, pixel = "source_not_decoded", None
    elif not construction:
        status, pixel = "excluded_later_source", None
    else:
        pixel, status = _observation(pose, joint, view["camera"]["width"], view["camera"]["height"])
    item["status"] = status
    landmarks = pose.get("landmarks") if construction and pose is not None else None
    if type(landmarks) is list and len(landmarks) == 33 and type(landmarks[joint]) is dict:
        for source, target in (("visibility", "model_visibility"), ("presence", "model_presence")):
            value = landmarks[joint].get(source)
            if _number(value, 0., 1.):
                item[target] = value
    if pixel is not None:
        item["pixel"] = pixel.tolist()
    return item, pixel


@dataclass(frozen=True)
class MultiviewLandmarkEvidence:
    """Keep exact inputs once; inspect only a bounded cached row per query."""

    view_inputs: tuple
    alignment_inputs: tuple
    _metadata: bytes
    _frames: tuple

    @property
    def frame_count(self):
        return len(self._frames)

    def describe(self):
        return json.loads(self._metadata)

    def frame(self, index):
        _require(type(index) is int and 0 <= index < len(self._frames), "frame index outside original source slots")
        return json.loads(self._frames[index])

    def iter_frame_bytes(self):
        return iter(self._frames)


def prepare_multiview_landmark_evidence(views, *, landmark_indices, alignments=None, rotation_prior=None):
    """Estimate selected original image landmarks using every qualified supplied view.

    Each view contains name, evidence_bytes, camera_bytes and clock_provenance.
    Alignments correspond to views 2–4; missing mappings yield unavailable points.
    Only explicit identity world/clock mappings are supported, without refitting.
    The optional versioned rotation prior is an assumed numerical correction, not calibration.
    """
    deadline = time.monotonic() + MAX_SECONDS
    _require(rotation_prior is None or type(rotation_prior) is str and rotation_prior == ROTATION_PROFILE,
             "rotation prior must be explicitly versioned or absent for strict raw-camera use")
    _require(type(views) in (list, tuple) and 2 <= len(views) <= 4, "two to four explicit views required")
    _require(type(landmark_indices) in (list, tuple) and 1 <= len(landmark_indices) <= 33
             and all(type(index) is int and 0 <= index < 33 for index in landmark_indices)
             and len(set(landmark_indices)) == len(landmark_indices), "unique original landmark indices required")
    indices = tuple(landmark_indices)
    alignments = (None,) * (len(views) - 1) if alignments is None else alignments
    _require(type(alignments) in (list, tuple) and len(alignments) == len(views) - 1,
             "one alignment slot per additional view required")
    raws, names = [], []
    for view in views:
        _require(type(view) is dict and set(view) == {"name", "evidence_bytes", "camera_bytes", "clock_provenance"},
                 "view fields differ from the source contract")
        _require(type(view["name"]) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", view["name"]),
                 "view name must be bounded and distinct")
        _require(type(view["clock_provenance"]) is str and 1 <= len(view["clock_provenance"].strip()) <= 2000,
                 "original-frame synchronization assumption must be explicit")
        names.append(view["name"])
        raws.extend((view["evidence_bytes"], view["camera_bytes"]))
    raws.extend(raw for raw in alignments if raw is not None)
    _require(len(set(names)) == len(names) and all(type(raw) is bytes and 0 < len(raw) <= MAX_DOCUMENT_BYTES for raw in raws)
             and sum(map(len, raws)) <= MAX_INPUT_BYTES, "distinct views and at most 32 MiB exact input bytes required")
    prepared, canonical_size = [], 0
    for view in views:
        evidence, evidence_pin, size = _decode(view["evidence_bytes"], deadline)
        camera, camera_pin, camera_size = _decode(view["camera_bytes"], deadline, camera=True)
        canonical_size += size + camera_size
        _require(canonical_size <= MAX_INPUT_BYTES, "canonical inputs exceed aggregate byte bound")
        pose_rows, cameras = _view({"name": view["name"], "evidence": evidence, "camera": camera,
                                   "clock_provenance": view["clock_provenance"]})
        decoded = evidence.get("decoded_frames", evidence["frame_count"])
        _require(type(decoded) is int and 1 <= decoded <= camera["frame_count"]
                 and all(frame < decoded for frame in pose_rows), "pose rows exceed decoded original slots")
        model = evidence.get("model")
        _require(type(model) is dict and type(model.get("sha256")) is str
                 and re.fullmatch(r"[0-9a-f]{64}", model["sha256"])
                 and type(model.get("name")) is str and 1 <= len(model["name"]) <= 200,
                 "exact source model name and byte identity must remain present")
        _require(_json_size(evidence["model"], deadline) <= 16384, "model provenance exceeds bounded metadata")
        prepared.append({"name": view["name"], "camera": camera, "poses": pose_rows, "cameras": cameras,
                         "decoded": decoded, "model": evidence["model"], "evidence_pin": evidence_pin,
                         "camera_pin": camera_pin, "clock_provenance": view["clock_provenance"]})
        _check(deadline)
    primary = prepared[0]
    count, fps = (primary["camera"][key] for key in ("frame_count", "fps"))
    _require(2 <= count <= MAX_FRAMES and all(view["camera"]["frame_count"] == count and view["camera"]["fps"] == fps
             for view in prepared), "same original frame-count/fps required; no clock fitting or resampling")
    _require(len({view["camera"]["source_sha256"] for view in prepared}) == len(prepared),
             "each supplied view needs a distinct original source video")
    cutoff = CONSTRUCTION_FRACTION * (count - 1)
    construction_frames = sum(frame <= cutoff for frame in range(count))
    possible_fits = sum(sum(frame in view["poses"] for view in prepared) >= 2
                        for frame in range(construction_frames)) * len(indices)
    _require(possible_fits <= MAX_FITS, "potential fit count exceeds work bound before geometry evaluation")
    registry = {view["evidence_pin"]["canonical_sha256"]: [
        pose["id"] for frame, pose in view["poses"].items() if frame <= cutoff] for view in prepared}
    alignment_records, alignment_statuses = [], ["primary_world_definition"]
    for view, raw in zip(prepared[1:], alignments, strict=True):
        value, pin = None, None
        if raw is not None:
            value, pin, size = _decode(raw, deadline, alignment=True)
            canonical_size += size
            _require(canonical_size <= MAX_INPUT_BYTES, "canonical inputs exceed aggregate byte bound")
        status = _binding(value, view["camera_pin"], primary["camera_pin"], registry)
        alignment_records.append({"pin": pin, "document": value, "status": status})
        alignment_statuses.append(status)
    bound = all(status == "conditional_supplied_identity_alignment" for status in alignment_statuses[1:])
    frames, statuses, observation_statuses = [], Counter(), Counter()
    total_bytes, fits, prior_count = 0, 0, 0
    for frame in range(count):
        _check(deadline)
        construction = frame <= cutoff
        row = {"frame": frame, "time_s": frame / fps,
               "partition": "construction_development" if construction else "excluded_later_source", "landmarks": []}
        camera_cache, prior_records = {}, {}
        for joint in indices:
            observations, pixels, qualified, used = [], [], [], []
            for view in prepared:
                item, pixel = _image_observation(view, frame, joint, construction)
                observations.append(item)
                observation_statuses[item["status"]] += 1
                if pixel is not None:
                    pixels.append(pixel)
                    qualified.append(view)
                    used.append(view["name"])
            fit = None
            if not construction:
                status = "excluded_later_source"
            elif not bound:
                status = "world_clock_binding_unavailable"
            elif len(pixels) < 2:
                status = "insufficient_qualified_views"
            else:
                _check(deadline)
                for view in qualified:
                    name = view["name"]
                    if name not in camera_cache:
                        camera = _prepared_camera(view["cameras"][frame])
                        if rotation_prior is not None:
                            prior = prepare_camera_rotation_prior(view["camera"]["frames"][frame]["R"])
                            camera["rotation"] = prior.corrected_rotation
                            prior_records[name] = {"view_name": name, "source_frame": frame,
                                "source_sha256": view["camera"]["source_sha256"], "camera_pin": view["camera_pin"],
                                "prior": prior.describe()}
                            prior_count += 1
                        camera_cache[name] = camera
                cameras = [camera_cache[name] for name in used]
                _check(deadline)
                fits += 1
                fit = triangulate_pixels(pixels, cameras)
                _check(deadline)
                status = "conditional_estimate" if fit["status"] == "estimated" else fit["status"]
            statuses[status] += 1
            point = {"landmark_index": joint, "status": status,
                "world_point_m": None if fit is None else fit["world_point_m"],
                "qualified_view_names": used, "observations": observations, "solver": fit,
                "contact_load_grasp": "unknown"}
            if rotation_prior is not None:
                point["solver_camera_rotation_ids"] = [] if fit is None else [
                    {"view_name": name, "prior_id": prior_records[name]["prior"]["id"]} for name in used]
            row["landmarks"].append(point)
        if rotation_prior is not None:
            row["camera_rotation_priors"] = list(prior_records.values())
        raw = _encoded(row, MAX_ROW_BYTES, deadline)
        total_bytes += len(raw)
        _require(total_bytes <= MAX_OUTPUT_BYTES, "all-row cache exceeds 8 MiB; no source slots were dropped")
        frames.append(raw)
    summary = {"schema_version": 1, "profile": PROFILE, "physical_validation": "unverified",
        "geometry_installed": False, "world_frame": _world(primary["camera_pin"]),
        "world_clock_binding_available": bound, "alignment_statuses": alignment_statuses,
        "alignments": alignment_records, "landmark_indices": list(indices),
        "views": [{"name": view["name"], "source_sha256": view["camera"]["source_sha256"],
            "evidence_pin": view["evidence_pin"], "camera_pin": view["camera_pin"], "model": view["model"],
            "metadata_source_frames": count, "decoded_source_frames": view["decoded"],
            "sampled_rows": len(view["poses"]), "calibration_quality": view["camera"]["quality"],
            "clock_provenance": view["clock_provenance"], "role": "construction_development"} for view in prepared],
        "coverage": {"original_source_frames": count, "construction_source_frames": construction_frames,
            "excluded_later_source_frames": count - construction_frames, "landmark_slots": count * len(indices),
            "view_landmark_slots": count * len(indices) * len(prepared), "status_counts": dict(statuses),
            "observation_status_counts": dict(observation_statuses), "solver_calls": fits},
        "contract": {"solver": dict(FIT_CONTRACT), "observation": dict(RAY_CONTRACT),
            "construction_fraction": CONSTRUCTION_FRACTION, "maximum_solver_calls": MAX_FITS,
            "maximum_input_bytes": MAX_INPUT_BYTES, "maximum_output_bytes": MAX_OUTPUT_BYTES,
            "maximum_seconds": MAX_SECONDS, "view_selection": "every_qualified_supplied_view_no_outlier_removal",
            "alignment": "explicit_identity_shared_world_and_original_clock_only"},
        "uncertainty": {"image_confidence": "stored_model_visibility_presence_not_calibrated_accuracy",
            "pixel_sensitivity": "one_pixel_independent_noise_sensitivity_not_measured_uncertainty",
            "calibration_and_clock": "caller_supplied_claims_not_independently_verified",
            "body_scale_shape_sole_contact": "not_estimated_or_admitted",
            "comparison": "all_supplied_views_are_construction_not_independent_accuracy_evidence"},
        "input_bytes": sum(map(len, raws)), "cached_row_bytes": total_bytes}
    if rotation_prior is not None:
        summary["numerical_rotation_prior"] = {"profile": ROTATION_PROFILE, "contract": dict(ROTATION_CONTRACT),
            "assumption_status": "assumed", "prepared_view_frames": prior_count,
            "raw_camera_inputs_preserved": True, "translation_intrinsics_radial_clock_unchanged": True,
            "physical_validation": "unverified"}
    metadata = _encoded(summary, MAX_METADATA_BYTES, deadline)
    _require(total_bytes + len(metadata) <= MAX_OUTPUT_BYTES, "complete prepared output exceeds 8 MiB")
    _check(deadline)
    return MultiviewLandmarkEvidence(
        tuple((view["name"], view["evidence_bytes"], view["camera_bytes"]) for view in views),
        tuple(alignments), metadata, tuple(frames))
