"""Construction residuals for conditional surface ranges; never contact or a point target."""

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import time

import numpy as np

from weft.io import canonical_hash
from weft.source_surface_anchors import _document, _fields, _sources
from weft.source_surface_envelopes import SourceSurfaceEnvelopes
from weft.whole_body_reference_refinement import NAMES, POINTS
from weft.world_alignment import load_world_alignment


PROFILE = "source-surface-interval-objective-v1"
BINDING_PROFILE = "source-surface-interval-body-binding-v1"
MAX_BODY_BYTES, MAX_BINDING_BYTES = 8 << 20, 64 << 10
MAX_CACHE_BYTES, MAX_ROW_BYTES = 8 << 20, 128 << 10
MAX_SECONDS, MAX_QUERY_SECONDS = 5., .25
MAX_FRAMES, MAX_REFERENCES, MAX_CHANNELS = 1800, 900, 12


def _require(value, message):
    if not value:
        raise ValueError("Surface interval construction: " + message)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Surface interval construction cooperative deadline exceeded")


def _number(value, bound=3600.):
    return type(value) in (int, float) and math.isfinite(value) and abs(value) <= bound


def _encode(value, limit, deadline):
    pieces, size = [], 0
    for part in json.JSONEncoder(allow_nan=False, sort_keys=True, separators=(",", ":")).iterencode(value):
        _check(deadline)
        piece = part.encode()
        size += len(piece)
        _require(size <= limit, "complete output exceeds its bound; no slots were dropped")
        pieces.append(piece)
    return b"".join(pieces)


def _input(raw, limit, deadline):
    _require(type(raw) is bytes and 0 < len(raw) <= limit, "bounded exact input bytes required")
    return _document(raw, deadline)


def _array(value, shape, label):
    # Check dimensions before NumPy can materialize an unrestricted nested input.
    def check(item, dimensions):
        if dimensions:
            _require(isinstance(item, (list, tuple, np.ndarray)) and len(item) == dimensions[0], label + " shape differs")
            for child in item:
                check(child, dimensions[1:])
        else:
            _require(isinstance(item, (int, float, np.integer, np.floating)) and not isinstance(item, (bool, np.bool_))
                     and math.isfinite(float(item)) and abs(float(item)) <= 1e6, label + " requires bounded finite numbers")
    check(value, shape)
    return np.asarray(value, dtype=float)


def _body(body, source, deadline):
    _require(body.get("profile") == "exercise-v3" and type(body.get("schema_version")) is int
             and body["schema_version"] == 3, "source-linked exercise-v3 body required")
    lists = [body.get(key) for key in ("reference", "reference_evidence", "root_reference")]
    _require(all(type(rows) is list and 2 <= len(rows) <= MAX_REFERENCES for rows in lists)
             and len({len(rows) for rows in lists}) == 1, "complete bounded body/reference/source clock rows required")
    references, provenance, roots = lists
    result, last_frame, last_time = {}, -1, -math.inf
    for index, (pose, item, root) in enumerate(zip(references, provenance, roots, strict=True)):
        _check(deadline)
        _require(all(type(row) is dict for row in (pose, item, root)), "body reference rows must be objects")
        identifier, stamp = item.get("evidence_id"), item.get("time_s")
        prefix, separator, suffix = identifier.rpartition(":frame:") if type(identifier) is str else (None, None, "")
        _require(separator and prefix == body.get("source_sha256") and suffix.isascii() and suffix.isdigit()
                 and len(suffix) <= 8, "exact original body source-frame identity required")
        frame = int(suffix)
        _require(last_frame < frame < MAX_FRAMES and _number(stamp) and 0 <= stamp and last_time < stamp
                 and pose.get("time_s") == stamp == root.get("time_s"), "original body reference/root clocks differ")
        _require(type(pose.get("joints")) is dict and set(pose["joints"]) == set(NAMES), "all original named joint columns required")
        result[frame] = {"reference_index": index, "evidence_id": identifier, "body_time_s": stamp}
        last_frame, last_time = frame, stamp
    reason = None if body.get("source_sha256") == source else "body_source_identity_differs"
    return result, reason


def _binding(document, metadata, body_hash, body_frame, allowed_ids):
    if document is None:
        return {}, None, None, "missing_explicit_body_binding"
    _fields(document, ("schema_version", "profile", "source_sha256", "body_sha256", "envelope_bindings",
                       "human_entity_ref", "body_to_world", "channels"), "interval body binding")
    _require(type(document["schema_version"]) is int and document["schema_version"] == 1
             and document["profile"] == BINDING_PROFILE, "versioned interval body binding required")
    _fields(document["human_entity_ref"], ("document_sha256", "object_id"), "human entity reference")
    _require(type(document["channels"]) is list and len(document["channels"]) <= MAX_CHANNELS, "at most twelve channel correspondences")
    channel_ids = {row["id"] for row in metadata["channels"]}
    channels = {}
    for item in document["channels"]:
        _fields(item, ("channel_id", "body_point", "status", "method", "uncertainty"), "channel correspondence")
        _require(item["channel_id"] in channel_ids and item["channel_id"] not in channels, "foreign or duplicate channel correspondence")
        _require((item["body_point"] in POINTS and item["status"] == "assumed")
                 or (item["body_point"] is None and item["status"] == "unknown"),
                 "endpoint correspondence must be explicit assumed core-body point or unknown")
        _require(all(type(item[key]) is str and 1 <= len(item[key].strip()) <= 2000 for key in ("method", "uncertainty")),
                 "correspondence needs bounded method and uncertainty")
        channels[item["channel_id"]] = item
    reason = None
    if document["body_sha256"] != body_hash or document["source_sha256"] != metadata["source_sha256"]:
        reason = "binding_body_or_source_identity_differs"
    elif document["envelope_bindings"] != metadata["bindings"]:
        reason = "binding_envelope_inputs_differ"
    mapping = document["body_to_world"]
    alignment = load_world_alignment(mapping, evidence_ids_by_document=allowed_ids) if mapping is not None else None
    if alignment is None or not alignment.ready:
        reason = reason or "missing_or_unknown_body_world_alignment"
    elif mapping["from_frame"] != body_frame or mapping["to_frame"] != metadata["world_frame"]:
        reason = reason or "incompatible_body_world_coordinate_frames"
    return channels, alignment, mapping, reason


def _transform(alignment, mapping):
    zero = alignment.map_state(from_frame=mapping["from_frame"], to_frame=mapping["to_frame"], time_s=0., position=[0., 0., 0.])
    columns = []
    for index in range(3):
        basis = [float(i == index) for i in range(3)]
        state = alignment.map_state(from_frame=mapping["from_frame"], to_frame=mapping["to_frame"],
                                    time_s=0., position=[0., 0., 0.], velocity=basis)
        # Position derivatives use spatial scale, not the clock-scaled velocity units.
        columns.append(np.asarray(state["velocity"]) * mapping["clock"]["rate"])
    return tuple(tuple(row) for row in np.column_stack(columns)), tuple(zero["position"])


@dataclass(frozen=True)
class SurfaceIntervalObjective:
    """Immutable source joins; queries use one cached frame, never the whole evidence tree."""

    envelopes: SourceSurfaceEnvelopes
    body_bytes: bytes
    binding_bytes: bytes | None
    _metadata: bytes
    _rows: tuple[bytes, ...]
    _body_sha256: str
    _body_frame: tuple
    _matrix: tuple | None
    _translation: tuple | None

    @property
    def frame_count(self):
        return len(self._rows)

    def describe(self):
        return json.loads(self._metadata)

    @property
    def coordinate_frame(self):
        return dict(self._body_frame)

    def frame(self, index):
        _require(type(index) is int and 0 <= index < self.frame_count, "original source frame outside range")
        result = json.loads(self._rows[index])
        result["source_envelopes"] = self.envelopes.frame(index)
        return result

    def _query(self, frame, body_sha256, body_time_s, coordinate_frame):
        _require(type(frame) is int and 0 <= frame < self.frame_count, "original source frame outside range")
        _require(body_sha256 == self._body_sha256, "candidate belongs to another body")
        _require(coordinate_frame == dict(self._body_frame), "candidate coordinate identity, units or axes differ")
        row = json.loads(self._rows[frame])
        expected = row["body_reference"]
        _require((expected is None and body_time_s is None) or
                 (expected is not None and _number(body_time_s) and body_time_s == expected["body_time_s"]),
                 "candidate body/source clock differs")
        return row

    def evaluate(self, frame, *, body_sha256, body_time_s, coordinate_frame, positions, jacobians=None, parameter_names=None):
        """Evaluate absolute model-frame positions; derivatives are model units per named joint radian."""
        deadline = time.monotonic() + MAX_QUERY_SECONDS
        row = self._query(frame, body_sha256, body_time_s, coordinate_frame)
        active = [item for item in row["channels"] if item["status"] == "conditional_interval_ready"]
        values, derivatives, parameters = {}, {}, ()
        if active:
            _require(type(positions) is dict and len(positions) <= MAX_CHANNELS and set(positions) <= set(POINTS), "bounded named body positions required")
            if jacobians is not None:
                _require(type(jacobians) is dict and set(jacobians) <= set(positions), "Jacobian point identities differ")
                _require(type(parameter_names) in (list, tuple) and 1 <= len(parameter_names) <= len(NAMES)
                         and len(set(parameter_names)) == len(parameter_names) and set(parameter_names) <= set(NAMES),
                         "unique named articulated-joint derivative columns required")
                parameters = tuple(parameter_names)
            else:
                _require(parameter_names is None, "derivative names require supplied Jacobians")
            # Unused/excluded candidate payloads cannot influence this source term.
            for name in {item["body_point"] for item in active} & positions.keys():
                values[name] = _array(positions[name], (3,), "body position")
                if jacobians is not None and name in jacobians:
                    derivatives[name] = _array(jacobians[name], (3, len(parameters)), "body Jacobian")
        source = self.envelopes.frame(frame)
        matrix = np.asarray(self._matrix) if active else None
        translation = np.asarray(self._translation) if active else None
        outputs, losses, residuals, jac_rows = [], [], [], []
        for joined, anchor in zip(row["channels"], source["anchors"], strict=True):
            _check(deadline)
            item = dict(joined)
            item.update(signed_outside_m=None, world_position_m=None, distance_to_outer_envelope_m=None,
                        objective_m2=None, jacobian_m_per_rad=None, active_axes=None)
            name = joined["body_point"]
            if joined["status"] == "conditional_interval_ready":
                if name not in values:
                    item.update(status="unavailable", reason="missing_candidate_body_point")
                else:
                    world = matrix @ values[name] + translation
                    bounds = anchor["world_envelope_m"]
                    lo, hi = np.asarray(bounds["minimum"]), np.asarray(bounds["maximum"])
                    residual = world - np.clip(world, lo, hi)
                    mask = (world < lo) | (world > hi)
                    loss = .5 * float(residual @ residual)
                    item.update(status="conditional_interval_residual", reason=None, signed_outside_m=residual.tolist(),
                                world_position_m=world.tolist(), distance_to_outer_envelope_m=float(np.linalg.norm(residual)),
                                objective_m2=loss, active_axes=mask.tolist())
                    losses.append(loss); residuals.append(residual)
                    derivative = mask[:, None] * (matrix @ derivatives[name]) if name in derivatives else None
                    if derivative is not None:
                        item["jacobian_m_per_rad"] = derivative.tolist()
                    jac_rows.append(derivative)
            outputs.append(item)
        gradient = hessian = None
        if jac_rows and all(item is not None for item in jac_rows):
            jac = np.vstack(jac_rows); residual = np.concatenate(residuals)
            gradient, hessian = (jac.T @ residual).tolist(), (jac.T @ jac).tolist()
        result = {"schema_version": 1, "profile": PROFILE, "source_frame": frame,
            "body_sha256": self._body_sha256, "body_reference": row["body_reference"], "channels": outputs,
            "counts": {"retained_channels": len(outputs), "evaluated_channels": len(losses),
                       "unavailable_channels": len(outputs) - len(losses)},
            "objective_m2": sum(losses) if losses else None, "parameter_names": list(parameters),
            "gradient_m2_per_rad": gradient, "gauss_newton_m2_per_rad2": hessian,
            "derivatives_status": "available" if gradient is not None else "unavailable",
            "geometry_modified": False, "contact": "unknown", "physical_validation": "unverified"}
        _encode(result, 512 << 10, deadline)
        _check(deadline)
        return result

    def evaluate_kinematics(self, frame, *, body_sha256, body_time_s, coordinate_frame, point_names, joint_names,
                            relative_positions, pelvis_position, relative_jacobian=None, pelvis_jacobian=None):
        """Consume body_refinement._kinematics output, restoring its removed pelvis position and derivative."""
        row = self._query(frame, body_sha256, body_time_s, coordinate_frame)
        if not any(item["status"] == "conditional_interval_ready" for item in row["channels"]):
            return self.evaluate(frame, body_sha256=body_sha256, body_time_s=body_time_s, coordinate_frame=coordinate_frame, positions=None)
        _require(type(point_names) in (list, tuple) and tuple(point_names) == tuple(POINTS)
                 and type(joint_names) in (list, tuple) and tuple(joint_names) == tuple(NAMES),
                 "kinematic point/joint column order differs from the existing producer")
        positions = _array(relative_positions, (12, 3), "relative positions") + _array(pelvis_position, (3,), "pelvis position")
        jacobians, parameters = None, None
        if relative_jacobian is not None:
            _require(pelvis_jacobian is not None, "absolute derivative requires the supplied pelvis Jacobian; zero is not assumed")
            relative = _array(relative_jacobian, (36, 19), "relative Jacobian").reshape(12, 3, 19)
            absolute = relative + _array(pelvis_jacobian, (3, 19), "pelvis Jacobian")
            jacobians = {name: value for name, value in zip(POINTS, absolute, strict=True)}
            parameters = NAMES
        else:
            _require(pelvis_jacobian is None, "pelvis derivative needs matching endpoint derivatives")
        return self.evaluate(frame, body_sha256=body_sha256, body_time_s=body_time_s,
            coordinate_frame=coordinate_frame,
            positions={name: value for name, value in zip(POINTS, positions, strict=True)},
            jacobians=jacobians, parameter_names=parameters)


def prepare_surface_interval_objective(envelopes, *, body_bytes, binding_bytes=None):
    """Prepare a callable construction term, not an installed optimizer or physical admission.

    Accept the trusted envelope API's prepared object; do not reconstruct its geometry.
    Source custody is rejoined once. Full-body validity and candidate geometry remain
    the trusted producer's responsibility, not a consequence of a matching hash.
    """
    deadline = time.monotonic() + MAX_SECONDS
    _require(type(body_bytes) is bytes and 0 < len(body_bytes) <= MAX_BODY_BYTES
             and (binding_bytes is None or type(binding_bytes) is bytes and 0 < len(binding_bytes) <= MAX_BINDING_BYTES),
             "bounded exact input bytes required")
    _require(type(envelopes) is SourceSurfaceEnvelopes and 1 <= envelopes.frame_count <= MAX_FRAMES,
             "bounded prepared source-surface envelopes required")
    _require(type(envelopes._metadata) is bytes and len(envelopes._metadata) <= MAX_ROW_BYTES
             and all(type(raw) is bytes and 0 < len(raw) <= MAX_ROW_BYTES for raw in envelopes.iter_frame_bytes())
             and len(envelopes._metadata) + sum(map(len, envelopes.iter_frame_bytes())) <= MAX_CACHE_BYTES,
             "prepared envelope cache exceeds its bounds")
    metadata = envelopes.describe()
    _require(metadata.get("profile") == "source-surface-envelopes-v1" and metadata.get("schema_version") == 1
             and metadata.get("counts", {}).get("retained_source_frames") == envelopes.frame_count
             and type(metadata.get("channels")) is list and len(metadata["channels"]) <= MAX_CHANNELS,
             "prepared envelope profile or complete denominators differ")
    source, documents, observations, evidence, allowed_ids, _ = _sources(dict(envelopes.observation_inputs), metadata["source_sha256"], deadline)
    _require(sorted(documents) == metadata["bindings"]["observation_document_sha256s"]
             and max(source["metadata_frames"], source["decoded_frames"]) == envelopes.frame_count,
             "envelope source custody or complete clock slots differ")
    body = _input(body_bytes, MAX_BODY_BYTES, deadline)
    body_rows, body_reason = _body(body, metadata["source_sha256"], deadline)
    body_hash, body_model_hash = sha256(body_bytes).hexdigest(), canonical_hash(body)
    body_frame = {"id": "body:" + body_model_hash, "length_unit": "human_model_unit", "axes": "x_right_y_away_z_up"}
    binding = _input(binding_bytes, MAX_BINDING_BYTES, deadline) if binding_bytes is not None else None
    channels, alignment, mapping, binding_reason = _binding(binding, metadata, body_hash, body_frame, allowed_ids)
    global_reason = body_reason or binding_reason
    matrix, translation = _transform(alignment, mapping) if global_reason is None else (None, None)
    entity = binding["human_entity_ref"] if binding is not None else None
    if entity is not None and (entity["document_sha256"], entity["object_id"], 0) not in observations:
        global_reason = global_reason or "human_entity_not_in_exact_source"
    rows, counts, cache_bytes = [], Counter(), 0
    digest = sha256(b"source-surface-envelope-cache-v1\0")
    digest.update(len(envelopes._metadata).to_bytes(8, "big")); digest.update(envelopes._metadata)
    for index, raw in enumerate(envelopes.iter_frame_bytes()):
        _check(deadline)
        digest.update(len(raw).to_bytes(8, "big")); digest.update(raw)
        row = json.loads(raw)
        _require(row.get("source_frame") == index and type(row.get("anchors")) is list
                 and len(row["anchors"]) == len(metadata["channels"]), "envelope source slots or channel denominator differ")
        reference = body_rows.get(index)
        joined = []
        for anchor, declaration in zip(row["anchors"], metadata["channels"], strict=True):
            _require(anchor["channel_id"] == declaration["id"] and anchor["entity_ref"] == declaration["entity_ref"],
                     "envelope channel/entity order differs")
            observed = observations.get((anchor["entity_ref"]["document_sha256"], anchor["entity_ref"]["object_id"], index))
            _require(observed is None and anchor["observation_id"] is None or observed is not None
                     and anchor["observation_id"] == observed["id"] and anchor["source_time_s"] == observed["time_s"]
                     and anchor["partition"] == observed["partition"], "envelope observation identity or clock differs")
            correspondence = channels.get(anchor["channel_id"])
            point = None if correspondence is None else correspondence["body_point"]
            reason = anchor["reason"] if anchor["status"] == "unavailable" else global_reason
            if reason is None and (observed is None or observed["admission"] != "included" or observed["partition"] != "construction"):
                reason = "not_construction_source_evidence"
            if reason is None and anchor["entity_ref"] != entity:
                reason = "channel_entity_differs_from_body_entity"
            if reason is None and point is None:
                reason = "missing_or_unknown_endpoint_correspondence"
            if reason is None and reference is None:
                reason = "original_body_reference_unavailable"
            if reason is None:
                evidence_ref = {"document_sha256": entity["document_sha256"], "evidence_id": reference["evidence_id"]}
                record = evidence.get((entity["document_sha256"], reference["evidence_id"]))
                if evidence_ref not in observed["evidence_refs"] or record is None or record["object_id"] != entity["object_id"] or record["frame"] != index:
                    reason = "body_reference_not_in_source_entity_evidence"
                elif record["time_s"] != observed["time_s"]:
                    reason = "body_evidence_source_clock_differs"
                elif reference["body_time_s"] != observed["time_s"]:
                    reason = "body_reference_source_clock_differs"
                else:
                    stamp = alignment.map_state(from_frame=mapping["from_frame"], to_frame=mapping["to_frame"],
                                                time_s=reference["body_time_s"], position=[0., 0., 0.])["time_s"]
                    if stamp != observed["time_s"]:
                        reason = "body_world_source_clock_differs"
            if reason is None:
                bounds = anchor["world_envelope_m"]
                _require(anchor["status"] == "conditional_surface_envelope" and bounds["unit"] == "m"
                         and bounds["kind"] == "outer_axis_aligned_enclosure", "qualified outer enclosure required")
                lo = _array(bounds["minimum"], (3,), "envelope minimum")
                hi = _array(bounds["maximum"], (3,), "envelope maximum")
                _require((lo <= hi).all(), "ordered closed envelope required")
            status = "conditional_interval_ready" if reason is None else "unavailable"
            counts[reason or status] += 1
            joined.append({"channel_id": anchor["channel_id"], "entity_ref": anchor["entity_ref"], "body_point": point,
                           "source_observation_id": anchor["observation_id"], "source_time_s": anchor["source_time_s"],
                           "source_partition": anchor["partition"], "envelope_status": anchor["status"],
                           "envelope_reason": anchor["reason"], "status": status, "reason": reason})
        encoded = _encode({"source_frame": index, "body_reference": reference, "channels": joined}, MAX_ROW_BYTES, deadline)
        cache_bytes += len(encoded)
        _require(cache_bytes <= MAX_CACHE_BYTES, "complete join cache exceeds 8 MiB")
        rows.append(encoded)
    start = min(body_rows)
    description = {"schema_version": 1, "profile": PROFILE,
        "status": "conditional_objective_available" if counts["conditional_interval_ready"] else "unavailable",
        "source_sha256": metadata["source_sha256"], "body_sha256": body_hash, "body_model_sha256": body_model_hash,
        "body_frame": body_frame, "world_frame": metadata["world_frame"], "source_clocks": metadata["source_clocks"],
        "envelope_bindings": metadata["bindings"], "envelope_cache_sha256": digest.hexdigest(),
        "binding_sha256": sha256(binding_bytes).hexdigest() if binding_bytes is not None else None,
        "binding": binding, "binding_unavailable_reason": global_reason,
        "original_start_frame": start, "original_start_statuses": json.loads(rows[start])["channels"] if start < len(rows) else [],
        "counts": {"retained_source_frames": len(rows), "retained_channel_slots": len(rows) * len(metadata["channels"]),
                   "original_body_references": len(body_rows), "status_counts": dict(counts)},
        "units": {"residual": "supplied_world_metre_prior", "objective": "supplied_world_metre_prior_squared",
                  "input_position": "human_model_unit", "input_jacobian": "human_model_unit_per_joint_radian"},
        "derivative_convention": "Zero on closed-box boundaries and inside; active outside. Chained spatial scale/rotation, no clock-rate division.",
        "scope": "Distance to a conditional outer enclosure; inside does not establish an attainable source point or contact",
        "source_scope": "Trusted prepared envelope output; exact raw canonical source registry joined once, enclosure geometry not recomputed",
        "candidate_scope": "Caller-produced absolute model-frame positions; full body and candidate kinematics require trusted worker verification",
        "optimizer_installation": "not_installed_no_existing_fitter_callback", "confidence": None,
        "body_or_floor_installed": False, "geometry_modified": False, "contact": "unknown", "load": "unknown", "grasp": "unknown",
        "physical_validation": "unverified", "cached_join_bytes": cache_bytes,
        "bounds": {"preparation_seconds_cooperative": MAX_SECONDS, "query_seconds_cooperative": MAX_QUERY_SECONDS,
                   "body_bytes": MAX_BODY_BYTES, "binding_bytes": MAX_BINDING_BYTES, "cache_bytes": MAX_CACHE_BYTES,
                   "source_frames": MAX_FRAMES, "body_references": MAX_REFERENCES, "channels": MAX_CHANNELS,
                   "derivative_columns": len(NAMES)}}
    encoded = _encode(description, MAX_ROW_BYTES, deadline)
    _require(cache_bytes + len(encoded) <= MAX_CACHE_BYTES, "complete metadata/join cache exceeds 8 MiB")
    _check(deadline)
    return SurfaceIntervalObjective(envelopes, body_bytes, binding_bytes, encoded, tuple(rows), body_hash,
                                    tuple(body_frame.items()), matrix, translation)
