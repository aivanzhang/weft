"""Collect explicit surface inputs against original observations; infer no geometry."""

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import time

from weft.camera_pose import validate_camera
from weft.source_surface_anchors import _annotations, _document, _sources
from weft.source_surface_envelopes import prepare_source_surface_envelopes
from weft.source_surface_fit_inputs import (
    CONFIG_MEMBER, MAX_CONFIG_BYTES, MAX_INPUT_BYTES, OPTIONAL_MEMBERS, PROFILE as CONFIG_PROFILE,
    _descriptor, parse_surface_fit_config, validate_surface_fit_snapshot,
)
from weft.source_surface_interval_construction import (
    BINDING_PROFILE, MAX_BINDING_BYTES, MAX_BODY_BYTES, _body, _encode,
    canonical_hash, prepare_surface_interval_objective,
)


PROFILE = "exercise-surface-input-collection-v1"
REQUEST_PROFILE = "exercise-surface-input-request-v1"
MAX_SECONDS, MAX_CACHE_BYTES, MAX_REQUEST_BYTES = 20., 8 << 20, (4 << 20) + (128 << 10)
MAX_EXTRA_BYTES, MAX_FRAMES, MAX_CHANNELS = 4 << 20, 1800, 12
_SCOPE = "Explicit conditional construction inputs, not recovered surfaces, contact, load, grasp or body admission"


def _require(value, message):
    if not value:
        raise ValueError("Exercise surface inputs: " + message)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Exercise surface inputs cooperative deadline exceeded")


def _json(raw, limit, deadline):
    _check(deadline)
    _require(type(raw) is bytes and 0 < len(raw) <= limit, "bounded exact JSON bytes required")
    value = _document(raw, deadline)
    _require(type(value) is dict, "JSON object required")
    return value


def _reason(frame, source, observed):
    if frame >= source["decoded_frames"]:
        return "metadata_only_not_decoded"
    if observed is None:
        return "missing_source_observation"
    if observed["partition"] == "unsampled":
        return "source_not_sampled"
    if observed["admission"] != "included" or observed["partition"] != "construction":
        return "excluded_evidence_partition"
    if observed["visibility"] not in {"detected", "partial"}:
        return "source_entity_unavailable"
    return None


@dataclass(frozen=True)
class QualifiedExerciseSurfaceInputs:
    """Immutable compiled files and bounded cached rows; no filesystem writes."""

    _metadata: bytes
    _rows: tuple[bytes, ...]
    _files: tuple[tuple[str, bytes], ...]

    def describe(self):
        return json.loads(self._metadata)

    def frame(self, index):
        _require(type(index) is int and 0 <= index < len(self._rows), "original frame outside range")
        return json.loads(self._rows[index])

    def files(self):
        return dict(self._files)


class ExerciseSurfaceInputs:
    """Prepare once; edit the existing schemas returned by request_document()."""

    def __init__(self, inputs, camera, body, metadata, request, rows):
        self._inputs, self._camera, self._body = inputs, camera, body
        self._metadata, self._request, self._rows = metadata, request, rows
        self._last = None

    def describe(self):
        return json.loads(self._metadata)

    def request_document(self):
        return self._request

    def frame(self, index):
        _require(type(index) is int and 0 <= index < len(self._rows), "original frame outside range")
        return json.loads(self._rows[index])

    def qualify(self, request_document):
        """Compile explicit edits into the existing fit config, preserving unknowns."""
        deadline = time.monotonic() + MAX_SECONDS
        _require(type(request_document) is bytes and 0 < len(request_document) <= MAX_REQUEST_BYTES,
                 "request exceeds its byte bound")
        pin = sha256(request_document).hexdigest()
        if self._last is not None and self._last[0] == pin:
            _check(deadline)
            return self._last[1]
        request = _json(request_document, MAX_REQUEST_BYTES, deadline)
        initial = json.loads(self._request)
        _require(set(request) == set(initial) and request["schema_version"] == 1
                 and type(request["schema_version"]) is int and request["profile"] == REQUEST_PROFILE
                 and request["input_bindings"] == initial["input_bindings"], "request differs from exact original inputs")
        annotation, surfaces, binding = (request[key] for key in ("annotations", "surface_intervals", "body_binding"))
        _require(type(annotation) is dict and annotation.get("channels") == initial["annotations"]["channels"],
                 "channel/entity selection changed; prepare a new explicit collection")
        _require(type(surfaces) is dict and type(surfaces.get("surfaces")) is list
                 and len(surfaces["surfaces"]) <= MAX_CHANNELS
                 and all(type(row) is dict for row in surfaces["surfaces"])
                 and [row.get("id") for row in surfaces["surfaces"]] == [row["id"] for row in initial["surface_intervals"]["surfaces"]],
                 "surface identity inventory differs")
        missing = []
        weight = request["weight"]
        if weight is None:
            missing.append("weight: supply an explicit authored fitting weight in (0,1000]")
        else:
            _require(type(weight) in (int, float) and math.isfinite(weight) and 0 < weight <= 1000,
                     "weight must be explicit and within (0,1000]")
        if self._camera is None:
            missing.append("camera_document: supply exact source-qualified calibration; no identity camera is inferred")
        if surfaces.get("world_frame") is None:
            missing.append("surface_intervals.world_frame: identify the supplied calibration world and metre units")
        for index, row in enumerate(surfaces["surfaces"]):
            _require(type(row) is dict, "surface must use the existing interval schema")
            kind = row.get("kind")
            _require(kind is None or kind in {"ground", "static_support"}, "moving/handheld or unknown-kind equipment is not a static surface")
            if kind is None:
                missing.append(f"surface_intervals.surfaces[{index}].kind: explicitly identify ground or a static-support hypothesis")
            if kind == "static_support" and row.get("bounds_xy_m") is None:
                missing.append(f"surface_intervals.surfaces[{index}].bounds_xy_m: supply finite static-support extents")
        _require(binding is None if self._body is None else type(binding) is dict,
                 "body binding requires the exact supplied original body")
        if binding is not None:
            _require(type(binding.get("channels")) is list and len(binding["channels"]) <= MAX_CHANNELS
                     and all(type(row) is dict for row in binding["channels"]), "bounded existing body-channel schema required")
        if binding is not None and binding.get("human_entity_ref") is None:
            missing.append("body_binding.human_entity_ref: select the exact observed person; equipment is not a body identity")
        actions = list(missing)
        for index, row in enumerate(surfaces["surfaces"]):
            if row.get("height_interval_m") is None:
                actions.append(f"surface_intervals.surfaces[{index}].height_interval_m: supply justified bounds or retain unknown; do not use a midpoint as a measurement")
        if not annotation.get("annotations"):
            actions.append("annotations.annotations: supply exact-frame pixel ranges or explicit occluded/unknown states using the cached source templates")
        if binding is not None:
            if binding.get("body_to_world") is None:
                actions.append("body_binding.body_to_world: supply the explicit unit/scale/rotation/translation and original-clock alignment, or retain unknown")
            for index, row in enumerate(binding.get("channels", [])):
                if row.get("body_point") is None:
                    actions.append(f"body_binding.channels[{index}].body_point: explicitly justify the named-body-endpoint correspondence; proximity is not contact")
        if missing:
            return self._finish(pin, request_document, missing, actions, None, None, {}, deadline)

        # Only foreign-key hashes are generated; all geometry and assumptions stay supplied.
        camera_pin = sha256(self._camera).hexdigest()
        _require(surfaces.get("calibration_sha256") in (None, camera_pin), "surface calibration bytes differ")
        surfaces["calibration_sha256"] = camera_pin
        annotation_raw = _encode(annotation, MAX_EXTRA_BYTES, deadline)
        surface_raw = _encode(surfaces, MAX_EXTRA_BYTES, deadline)
        _require(len(self._camera) + len(annotation_raw) + len(surface_raw) <= MAX_EXTRA_BYTES,
                 "camera/surface/annotation inputs exceed four MiB")
        envelopes = prepare_source_surface_envelopes(dict(self._inputs), source_sha256=initial["input_bindings"]["source_sha256"],
            camera_document=self._camera, surface_interval_document=surface_raw, annotation_document=annotation_raw)
        _check(deadline)
        envelope_meta = envelopes.describe()
        objective, binding_raw = None, None
        if binding is not None:
            _require(binding.get("envelope_bindings") in (None, envelope_meta["bindings"]),
                     "explicit body binding belongs to different envelope inputs")
            binding["envelope_bindings"] = envelope_meta["bindings"]
            binding_raw = _encode(binding, MAX_BINDING_BYTES, deadline)
            objective = prepare_surface_interval_objective(envelopes, body_bytes=self._body,
                binding_bytes=binding_raw, include_image_admission=True)
            _check(deadline)
        documents = {"camera": self._camera, "surface_intervals": surface_raw,
                     "annotations": annotation_raw, "body_binding": binding_raw}
        files, observations = {}, []
        for index, (_, raw) in enumerate(self._inputs):
            name = f"surface-observation-{index:02d}.json"
            files[name] = raw; observations.append({"path": name, "sha256": sha256(raw).hexdigest()})
        config = {"schema_version": 1, "profile": CONFIG_PROFILE,
            "source_sha256": initial["input_bindings"]["source_sha256"], "weight": weight, "observations": observations}
        for role, member in OPTIONAL_MEMBERS.items():
            raw = documents[role]; name = member.removeprefix("inputs/")
            config[role] = None if raw is None else {"path": name, "sha256": sha256(raw).hexdigest()}
            if raw is not None:
                files[name] = raw
        config_raw = _encode(config, MAX_CONFIG_BYTES, deadline)
        config = parse_surface_fit_config(config_raw)
        snapshot = {"inputs/" + name: raw for name, raw in files.items()} | {CONFIG_MEMBER: config_raw}
        validate_surface_fit_snapshot(snapshot, _descriptor(config, config_raw), source_sha256=config["source_sha256"])
        _check(deadline)
        files["surface-config.json"] = config_raw
        return self._finish(pin, request_document, [], actions, envelopes, objective, files, deadline)

    def _finish(self, pin, request_raw, missing, actions, envelopes, objective, files, deadline):
        rows, total, statuses, ready = [], 0, Counter(), 0
        for frame, raw in enumerate(self._rows):
            _check(deadline)
            row = {"source": json.loads(raw), "envelope": None, "body_join": None, "image_admission": None}
            if envelopes is not None:
                row["envelope"] = envelopes.frame(frame)
                statuses.update(anchor["reason"] or anchor["status"] for anchor in row["envelope"]["anchors"])
            if objective is not None:
                joined = objective.frame(frame)
                row["body_join"] = {key: value for key, value in joined.items() if key != "source_envelopes"}
                row["image_admission"] = objective.image_admission(frame)
                ready += sum(channel["status"] == "conditional_interval_ready" for channel in joined["channels"])
            encoded = _encode(row, 128 << 10, deadline); total += len(encoded)
            _require(total <= MAX_CACHE_BYTES, "complete qualification cache exceeds eight MiB; no frames dropped")
            rows.append(encoded)
        source_meta = self.describe()
        envelope_meta = envelopes.describe() if envelopes is not None else None
        objective_meta = objective.describe() if objective is not None else None
        conditional = 0 if envelope_meta is None else envelope_meta["counts"]["conditional_envelopes"]
        available = ready or self._body is None and conditional
        status = "incomplete_request" if missing else "conditional_inputs_available" if available else "qualified_but_unavailable"
        metadata = {"schema_version": 1, "profile": PROFILE, "status": status,
            "request_sha256": pin, "input_bindings": source_meta["input_bindings"], "source_counts": source_meta["source_counts"],
            "source_clocks": source_meta["source_clocks"],
            "missing_fields": missing, "required_explicit_inputs": actions,
            "conditional_envelopes": conditional, "conditional_body_channel_terms": ready,
            "envelope_status_counts": dict(statuses), "envelope_metadata": envelope_meta, "body_join_metadata": objective_meta,
            "config_member": "surface-config.json" if files else None,
            "config_schema_qualified": bool(files), "original_body_supplied": self._body is not None,
            "cached_row_bytes": total, "source_worksheet_cache_bytes": sum(map(len, self._rows)),
            "scope": _SCOPE, "contact": "unknown", "load": "unknown", "grasp": "unknown",
            "physical_validation": "unverified", "body_or_floor_installed": False,
            "body_geometry_evaluations": 0, "fit_or_simulator_admission": False,
            "next_action": "Supply the named missing fields, or inspect every unavailable source/channel reason; do not fabricate coordinates"}
        encoded_meta = _encode(metadata, 256 << 10, deadline)
        _require(total + len(encoded_meta) <= MAX_CACHE_BYTES, "qualification metadata/cache exceeds eight MiB")
        # Every output name is fixed locally; no edited request can select a path.
        files = dict(files)
        for index, (_, raw) in enumerate(self._inputs):
            files.setdefault(f"surface-observation-{index:02d}.json", raw)
        if self._camera is not None:
            files.setdefault("surface-camera.json", self._camera)
        if self._body is not None:
            files["original-human.json"] = self._body
        files.update({"collection-request.json": request_raw, "collection-report.json": encoded_meta,
                      "collection-frames.jsonl": b"\n".join(rows) + b"\n"})
        _require(len(files) <= 20 and sum(map(len, files.values())) <= (32 << 20) - (64 << 10),
                 "complete collection package exceeds its file/byte bound")
        files["collection-manifest.json"] = _encode({"schema_version": 1, "profile": PROFILE, "status": status,
            "files": {name: sha256(raw).hexdigest() for name, raw in sorted(files.items())},
            "scope": "Pure input collection snapshot; no source media read, fit, physical admission or external authentication"}, 64 << 10, deadline)
        _check(deadline)
        result = QualifiedExerciseSurfaceInputs(encoded_meta, tuple(rows), tuple(sorted(files.items())))
        self._last = (pin, result)
        return result


def prepare_exercise_surface_inputs(observation_documents, *, source_sha256, channels,
        camera_document=None, body_document=None, human_entity_ref=None):
    """Create an unqualified editable request and complete source annotation worksheet.

    Channels use the existing source-surface-annotations-v1 schema. All document
    inputs are exact bytes; source identity is checked, not freshly decoded.
    """
    deadline = time.monotonic() + MAX_SECONDS
    _require(type(observation_documents) is dict and 1 <= len(observation_documents) <= 8
             and all(type(raw) is bytes and 0 < len(raw) <= 8 << 20 for raw in observation_documents.values())
             and sum(map(len, observation_documents.values())) <= MAX_INPUT_BYTES, "one to eight bounded exact observation documents required")
    _require(camera_document is None or type(camera_document) is bytes and 0 < len(camera_document) <= MAX_EXTRA_BYTES,
             "camera exceeds the existing four-MiB optional-input bound")
    _require(sum(map(len, observation_documents.values())) + (len(camera_document) if camera_document else 0) <= MAX_INPUT_BYTES,
             "surface input documents exceed sixteen MiB")
    source, documents, observations, _, _, _ = _sources(observation_documents, source_sha256, deadline)
    count = max(source["decoded_frames"], source["metadata_frames"])
    _require(count <= MAX_FRAMES, "complete original source exceeds 1800 frames")
    annotation = {"schema_version": 1, "profile": "source-surface-annotations-v1", "source_sha256": source_sha256,
        "quality": "authored", "provenance": "Explicit collection request; unprovided annotations remain unknown",
        "channels": channels, "annotations": []}
    checked_channels, _ = _annotations(annotation, source, documents, observations, deadline)
    camera_hash, body_hash, body_frame, body_refs = None, None, None, {}
    if camera_document is not None:
        camera = _json(camera_document, MAX_EXTRA_BYTES, deadline)
        validate_camera(camera, {"source_sha256": source_sha256, "width": source["width_px"], "height": source["height_px"],
                                "frame_count": source["metadata_frames"], "fps": source.get("fps", camera.get("fps"))})
        camera_hash = sha256(camera_document).hexdigest()
    if body_document is not None:
        body = _json(body_document, MAX_BODY_BYTES, deadline)
        body_refs, reason = _body(body, source_sha256, deadline)
        _require(reason is None, "body source identity differs")
        body_hash = sha256(body_document).hexdigest()
        body_frame = {"id": "body:" + canonical_hash(body), "length_unit": "human_model_unit", "axes": "x_right_y_away_z_up"}
    if human_entity_ref is not None:
        _require(type(human_entity_ref) is dict and set(human_entity_ref) == {"document_sha256", "object_id"}
                 and (human_entity_ref["document_sha256"], human_entity_ref["object_id"], 0) in observations,
                 "explicit human entity must resolve in exact original observations")
    _require(body_document is not None or human_entity_ref is None, "human entity selection needs an original body")
    bindings = {"source_sha256": source_sha256, "observation_document_sha256s": sorted(documents),
                "camera_raw_sha256": camera_hash, "body_raw_sha256": body_hash}
    surface_ids = list(dict.fromkeys(channel["surface_id"] for channel in checked_channels))
    surface = {"schema_version": 1, "profile": "source-horizontal-surface-intervals-v1", "source_sha256": source_sha256,
        "calibration_sha256": camera_hash, "world_frame": None, "calibration_evidence_refs": [],
        "surfaces": [{"id": identity, "kind": None, "height_interval_m": None, "bounds_xy_m": None,
            "quality": "unknown", "provenance": "Not supplied; no ground or static-equipment classification inferred", "evidence_refs": []}
            for identity in surface_ids]}
    binding = None if body_document is None else {"schema_version": 1, "profile": BINDING_PROFILE,
        "source_sha256": source_sha256, "body_sha256": body_hash, "envelope_bindings": None,
        "human_entity_ref": human_entity_ref, "body_to_world": None,
        "channels": [{"channel_id": channel["id"], "body_point": None, "status": "unknown",
            "method": "Not supplied", "uncertainty": "An image point is not automatically a body endpoint or contact"}
            for channel in checked_channels]}
    request = {"schema_version": 1, "profile": REQUEST_PROFILE, "input_bindings": bindings,
               "weight": None, "surface_intervals": surface, "annotations": annotation, "body_binding": binding}
    rows, size, counts = [], 0, Counter()
    for frame in range(count):
        _check(deadline)
        row = {"source_frame": frame, "source_status": "decoded" if frame < source["decoded_frames"] else "metadata_only_not_decoded",
               "original_body_reference": body_refs.get(frame), "channels": []}
        for channel in checked_channels:
            ref = channel["entity_ref"]
            observed = observations.get((ref["document_sha256"], ref["object_id"], frame))
            reason = _reason(frame, source, observed); counts[reason or "construction_annotation_candidate"] += 1
            template = None if reason is not None else {"id": f"{source_sha256}:frame:{frame}:surface-anchor:{channel['id']}",
                "channel_id": channel["id"], "frame": frame, "time_s": observed["time_s"], "observation_id": observed["id"],
                "status": "unknown", "pixel": None, "uncertainty_px": None}
            row["channels"].append({"channel_id": channel["id"], "entity_ref": ref,
                "surface_id": channel["surface_id"], "reason": reason, "annotation_template": template,
                "source_observation_id": None if observed is None else observed["id"],
                "source_time_s": None if observed is None else observed["time_s"],
                "source_partition": None if observed is None else observed["partition"],
                "source_visibility": None if observed is None else observed["visibility"],
                "source_evidence_refs": [] if observed is None else observed["evidence_refs"]})
        raw = _encode(row, 128 << 10, deadline); size += len(raw)
        _require(size <= MAX_CACHE_BYTES, "complete source worksheet exceeds eight MiB; no slots dropped")
        rows.append(raw)
    request_raw = _encode(request, MAX_REQUEST_BYTES, deadline)
    metadata = {"schema_version": 1, "profile": PROFILE, "status": "unqualified_editable_request", "input_bindings": bindings,
        "source_clocks": {digest: value["time"] for digest, value in documents.items()},
        "source_counts": {"decoded_frames": source["decoded_frames"], "metadata_frames": source["metadata_frames"],
            "retained_source_frames": count, "channel_slots": count * len(checked_channels), "status_counts": dict(counts)},
        "body_coordinate_frame": body_frame, "cached_row_bytes": size, "request_sha256": sha256(request_raw).hexdigest(),
        "scope": _SCOPE, "contact": "unknown", "physical_validation": "unverified", "body_or_floor_installed": False,
        "source_media_read": False, "limits": {"cooperative_seconds_per_preparation_or_qualification": MAX_SECONDS,
            "source_worksheet_cache_bytes": MAX_CACHE_BYTES, "qualification_cache_bytes": MAX_CACHE_BYTES,
            "last_qualification_cache_entries": 1, "surface_input_bytes": MAX_INPUT_BYTES,
            "original_body_bytes": MAX_BODY_BYTES, "package_bytes": 32 << 20}}
    metadata_raw = _encode(metadata, 128 << 10, deadline)
    _require(size + len(metadata_raw) + len(request_raw) <= MAX_CACHE_BYTES, "complete prepared cache exceeds eight MiB")
    _check(deadline)
    return ExerciseSurfaceInputs(tuple(sorted(observation_documents.items())), camera_document, body_document,
                                 metadata_raw, request_raw, tuple(rows))
