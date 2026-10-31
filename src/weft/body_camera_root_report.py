"""Pure offline inspection of saved camera/root proposals, never admission."""

from hashlib import sha256
from html import escape
import json
import math


MAX_INPUT_BYTES = 8 << 20
MAX_PROPOSAL_BYTES = 16 << 20
MAX_OUTPUT_BYTES = 8 << 20
_LINKS = frozenset(("geometry.json", "camera-root-input.json", "target-provenance.json",
                   "source-slots.json", "proposal.json", "geometry-check.json", "worker.json",
                   "manifest.json", "plan.json"))
_BINDINGS = ("source_sha256", "body_sha256", "evidence_sha256", "body_angles_sha256", "body_shape_sha256")
_GATES = {"first_root_unchanged": "First root unchanged", "bounds": "Camera/root change bounds",
          "image_pointwise_no_worse": "Every known image endpoint no worse",
          "saved_floor_no_worse": "Saved floor gaps no worse",
          "saved_floor_feasibility": "Saved floor limit (0.025 model units)"}
_GEOMETRY_GATES = {"fixed_plane": "Original floor plane unchanged", "floor_no_worse": "Native gaps no worse",
                   "floor_feasible": "Native floor limit (0.025 model units)",
                   "translation_consistency": "Fixed-body translation consistency"}
_SETTINGS = {"maximum_seconds": 10., "maximum_iterations": 8, "maximum_backtracks": 8,
             "maximum_camera_rotation_rad": .15, "maximum_root_translation_model_units": .08,
             "maximum_root_correction_speed_model_units_s": .3}
_REASONS = {"no_accepted_direction": "No update satisfied every required check",
            "iteration_limit": "Reached the iteration limit",
            "already_matches_available_pixels": "Available image points already match",
            "unidentified_or_ill_conditioned": "Available observations cannot distinguish this camera/body change",
            "time_limit": "Stopped at the time limit"}


def _require(okay, message):
    if not okay:
        raise ValueError("Camera/root report: " + message)


def _fields(value, fields, *, extra=False):
    _require(type(value) is dict and (set(fields) <= set(value) if extra else set(value) == set(fields)),
             "unsupported or missing document fields")


def _load(raw, limit):
    _require(type(raw) is bytes and 0 < len(raw) <= limit, "bounded JSON bytes required")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    try:
        result = json.loads(raw, object_pairs_hook=unique)
        json.dumps(result, allow_nan=False)
    except (ValueError, TypeError, RecursionError, OverflowError, UnicodeError) as error:
        raise ValueError("Camera/root report: invalid finite JSON") from error
    _require(type(result) is dict, "JSON object required")
    return result


def _hash(value):
    _require(type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
             "exact SHA-256 identity required")
    return value


def _canonical(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _number(value, low=-1e12, high=1e12):
    _require(type(value) in (int, float) and math.isfinite(value) and low <= value <= high,
             "finite bounded number required")
    return value


def _count(value, high=10000000):
    _require(type(value) is int and 0 <= value <= high, "bounded integer count required")
    return value


def _text(value, limit=2000):
    _require(type(value) is str and 0 < len(value) <= limit
             and all(ord(c) >= 32 or c in "\n\r\t" for c in value), "bounded text required")
    return value


def _vector(value, size, bound=1e4):
    _require(type(value) is list and len(value) == size, "fixed numeric vector required")
    return [_number(v, -bound, bound) for v in value]


def _near(left, right):
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-7)


def _flags(document):
    _require(document.get("admitted") is False and document.get("promoted") is False
             and document.get("physical_validation") == "unverified", "report cannot claim admission or physical validation")


def _camera(value):
    _fields(value, ("rotation", "center", "K", "radial_k1", "quality", "provenance"))
    for name, bound in (("rotation", 1.000000001), ("K", 1e6)):
        _require(type(value[name]) is list and len(value[name]) == 3, "three-row camera matrix required")
        for row in value[name]:
            _vector(row, 3, bound)
    _vector(value["center"], 3)
    rotation = value["rotation"]
    _require(all(abs(sum(rotation[k][i]*rotation[k][j] for k in range(3))-(1 if i == j else 0)) <= 1e-9
                 for i in range(3) for j in range(3)), "proper camera rotation required")
    determinant = sum(rotation[0][i] * (rotation[1][(i+1)%3]*rotation[2][(i+2)%3]
                                      - rotation[1][(i+2)%3]*rotation[2][(i+1)%3]) for i in range(3))
    _require(abs(determinant-1) <= 1e-9, "camera reflection is not a rotation")
    _number(value["radial_k1"], -1, 1)
    _require(value["K"][2] == [0, 0, 1] and value["K"][0][0] > 0
             and value["K"][1][1] > 0 and value["K"][1][0] == 0, "pinhole intrinsics differ")
    _require(value["quality"] == "authored", "this report requires the explicit authored-camera adapter")
    _text(value["provenance"])


def _project(point, camera):
    relative = [point[i] - camera["center"][i] for i in range(3)]
    x, y, z = [sum(row[j] * relative[j] for j in range(3)) for row in camera["rotation"]]
    _require(z >= .05 - 1e-10, "saved geometry is behind the projection limit")
    x, y = x / z, y / z
    radial = 1 + camera["radial_k1"] * (x * x + y * y)
    k = camera["K"]
    return [_number((k[0][0] * x + k[0][1] * y) * radial + k[0][2]),
            _number(k[1][1] * y * radial + k[1][2])]


def _gates(value, names, *, nullable=True):
    _fields(value, names)
    _require(all(type(v) is bool or (nullable and v is None) for v in value.values()), "explicit gate results required")
    return value


def _origin(row):
    source = row.get("landmark_source")
    _require(source in ("landmarks", "identity_raw.landmarks"), "unsupported landmark source")
    origin = row.get("target_origin")
    if origin is None:
        return ("Preserved identity-raw whole-frame estimate" if source == "identity_raw.landmarks"
                else "Whole-frame image estimate"), None
    _fields(origin, ("kind", "method", "original_observation_sha256", "proposal_sha256",
                     "decision_sha256", "anatomical_identity_verified"))
    _require(source == "landmarks" and origin["kind"] == "selected_crop_estimate_mapped_to_source"
             and origin["method"] == "single-view-selective-crop-v1"
             and origin["anatomical_identity_verified"] is False, "crop target origin differs")
    for key in ("original_observation_sha256", "proposal_sha256", "decision_sha256"):
        _hash(origin[key])
    return "Selected crop estimate, mapped to source image", origin


def _floor_rows(document, report, times):
    original, proposed = document["floor_checks"], report["floor_checks"]
    if original is None:
        _require(proposed is None and report["gates"]["saved_floor_no_worse"] is None
                 and report["gates"]["saved_floor_feasibility"] is None, "missing floor evidence cannot pass")
        return None, None
    _fields(original, ("bindings", "rows"))
    expected = {k: document["bindings"][k] for k in ("body_sha256", "body_angles_sha256", "body_shape_sha256")}
    expected.update(root_reference_sha256=_canonical([{"time_s": r["time_s"], "translation": r["root"]}
                                                       for r in document["samples"]]), floor_sha256=_canonical(document["floor"]))
    _require(original["bindings"] == expected, "floor evidence identities differ")
    grid = math.floor((times[-1] - times[0]) * 30 + 1e-9) + 1
    schedule = [("source_pose", t) for t in times] + [("control_grid", min(times[-1], times[0] + i / 30)) for i in range(grid)]
    _require(type(original["rows"]) is list and type(proposed) is list
             and len(original["rows"]) == len(proposed) == len(schedule), "complete saved floor schedule required")
    summary = {kind: {"checks": 0, "before_failures": 0, "after_failures": 0} for kind in ("source_pose", "control_grid")}
    no_worse, feasible, supports = True, True, None
    for old, new, (kind, stamp) in zip(original["rows"], proposed, schedule):
        _fields(old, ("kind", "time_s", "gaps")); _fields(new, ("kind", "time_s", "gaps_before", "gaps_after"))
        _require(old["kind"] == new["kind"] == kind and old["time_s"] == new["time_s"] == stamp
                 and new["gaps_before"] == old["gaps"], "saved floor row identity differs")
        before, after = old["gaps"], new["gaps_after"]
        _require(type(before) is dict and type(after) is dict and 1 <= len(before) <= 64
                 and set(before) == set(after) and (supports is None or set(before) == supports), "floor support slots differ")
        supports = set(before)
        for name in before:
            _text(name, 100); _number(before[name], -1e4, 1e4); _number(after[name], -1e4, 1e4)
            no_worse = no_worse and after[name] >= min(0., before[name]) - 1e-8
        failed = min(after.values()) < -.025
        summary[kind]["checks"] += 1
        summary[kind]["before_failures"] += min(before.values()) < -.025
        summary[kind]["after_failures"] += failed
        feasible = feasible and not failed
    _require(report["gates"]["saved_floor_no_worse"] == no_worse
             and report["gates"]["saved_floor_feasibility"] == feasible, "saved floor gates contradict gaps")
    return summary, proposed


def _geometry(check, document, proposal_raw, provenance, floor):
    _fields(check, ("schema_version", "profile", "status", "reason", "bindings", "source_pose_checks",
                    "control_grid_checks", "unique_evaluations", "rows", "maximum_point_translation_error_model_units",
                    "maximum_gap_translation_error_model_units", "gates", "physics_steps", "admitted", "promoted",
                    "physical_validation", "scope"))
    _require(type(check["schema_version"]) is int and check["schema_version"] == 1
             and check["profile"] == "fixed-original-body-geometry-check-v1", "unsupported geometry diagnostic")
    _flags(check); _text(check["scope"])
    expected = {k: document["bindings"][k] for k in ("body_sha256", "evidence_sha256")}
    expected.update(geometry_sha256=provenance["geometry_snapshot_sha256"], proposal_sha256=sha256(proposal_raw).hexdigest())
    _require(check["bindings"] == expected and type(check["physics_steps"]) is int
             and check["physics_steps"] == 0, "geometry diagnostic identities or physics scope differ")
    counts = [check[key] for key in ("source_pose_checks", "control_grid_checks", "unique_evaluations")]
    for count in counts:
        _count(count, 2700)
    if check["status"] == "not_required":
        skipped_gates = _gates(check["gates"], _GEOMETRY_GATES)
        _require(check["reason"] == "original_roots_unchanged" and counts == [0, 0, 0] and check["rows"] == []
                 and all(check[key] is None for key in ("maximum_point_translation_error_model_units",
                    "maximum_gap_translation_error_model_units")) and all(value is None for value in skipped_gates.values()),
                 "skipped geometry must remain unknown")
        return {"status": "not_required", "gates": None, "failures": [], "counts": counts}
    _require(check["status"] == "completed" and check["reason"] is None and floor is not None, "unsupported geometry stage")
    gates = _gates(check["gates"], _GEOMETRY_GATES, nullable=False)
    _require(type(check["rows"]) is list and len(check["rows"]) == len(floor), "native diagnostic schedule differs")
    _require(counts == [sum(r["kind"] == "source_pose" for r in floor), sum(r["kind"] == "control_grid" for r in floor),
                       len({r["time_s"] for r in floor})], "native diagnostic counts differ")
    failures, no_worse, feasible, max_error = [], True, True, 0.
    for row, saved in zip(check["rows"], floor):
        _fields(row, ("kind", "time_s", "gaps", "maximum_gap_translation_error_model_units"))
        _require(row["kind"] == saved["kind"] and row["time_s"] == saved["time_s"]
                 and type(row["gaps"]) is dict and set(row["gaps"]) == set(saved["gaps_before"]), "native diagnostic row differs")
        error = _number(row["maximum_gap_translation_error_model_units"], 0)
        observed_error = 0.
        for name, gap in row["gaps"].items():
            _number(gap, -1e4, 1e4)
            observed_error = max(observed_error, abs(gap - saved["gaps_after"][name]))
            no_worse = no_worse and gap >= min(0., saved["gaps_before"][name]) - 1e-8
        _require(_near(error, observed_error), "native gap translation diagnostic differs")
        max_error = max(max_error, error)
        if min(row["gaps"].values()) < -.025:
            feasible = False
            if len(failures) < 8:
                failures.append({"kind": row["kind"], "time_s": row["time_s"], "minimum_gap": min(row["gaps"].values())})
    _require(gates["floor_no_worse"] == no_worse and gates["floor_feasible"] == feasible, "native floor gates contradict gaps")
    point_error = _number(check["maximum_point_translation_error_model_units"], 0)
    gap_error = _number(check["maximum_gap_translation_error_model_units"], 0)
    _require(_near(gap_error, max_error), "native maximum gap error differs")
    _require(gates["translation_consistency"] == (max(point_error, gap_error) <= 1e-8),
             "translation consistency gate contradicts saved errors")
    return {"status": "completed", "gates": gates, "failures": failures, "counts": counts}


def _model(raw_input, raw_proposal, raw_geometry, raw_provenance):
    document = _load(raw_input, MAX_INPUT_BYTES)
    proposal = _load(raw_proposal, MAX_PROPOSAL_BYTES)
    check = _load(raw_geometry, MAX_INPUT_BYTES)
    provenance = _load(raw_provenance, MAX_INPUT_BYTES)
    _fields(document, ("schema_version", "profile", "bindings", "scale_model_units_per_unit", "camera", "floor", "samples", "floor_checks"))
    _require(type(document["schema_version"]) is int and document["schema_version"] == 1
             and document["profile"] == "fixed-body-camera-root-input-v1", "unsupported input profile")
    _fields(document["bindings"], _BINDINGS)
    for value in document["bindings"].values():
        _hash(value)
    _number(document["scale_model_units_per_unit"], 1e-6, 1e6)
    _fields(document["floor"], ("normal", "offset_model_units"))
    normal = _vector(document["floor"]["normal"], 3, 1)
    _require(_near(sum(v*v for v in normal), 1), "unit floor normal required")
    _number(document["floor"]["offset_model_units"], -1e4, 1e4)
    _camera(document["camera"])
    _fields(proposal, ("status", "camera_candidate", "root_reference_candidate", "report"))
    _require(proposal["status"] in ("proposed", "unchanged", "unavailable", "partial"), "unsupported proposal status")
    _camera(proposal["camera_candidate"])
    _require(all(document["camera"][k] == proposal["camera_candidate"][k] for k in document["camera"] if k != "rotation"),
             "fixed camera fields changed")
    report = proposal["report"]
    _fields(report, ("schema_version", "profile", "reason", "contract", "settings", "input_sha256", "bindings", "binding_scope",
                    "fixed_input_sha256", "initial_identifiability", "final_identifiability", "counts", "known_endpoints",
                    "unknown_endpoints", "image_rmse_before_px", "image_rmse_after_px", "maximum_camera_rotation_rad",
                    "maximum_root_translation_model_units", "maximum_root_correction_speed_model_units_s", "gates", "samples",
                    "floor_checks", "elapsed_seconds", "geometry_admission", "source_integration", "geometry_qualification",
                    "source_qualified", "admitted", "physical_validation", "promoted"))
    _require(type(report["schema_version"]) is int and report["schema_version"] == 1
             and report["profile"] == "fixed-body-camera-root-proposal-v1", "unsupported proposal profile")
    _flags(report)
    _require(report["source_qualified"] is False and report["geometry_admission"] == "unverified_not_executed"
             and report["source_integration"] == "not_performed", "foundation scope changed")
    _require(report["input_sha256"] == sha256(raw_input).hexdigest() and report["bindings"] == document["bindings"], "proposal input identities differ")
    fixed = {"bindings": document["bindings"], "scale": document["scale_model_units_per_unit"], "floor": document["floor"],
             "first_root": document["samples"][0]["root"],
             "camera_except_rotation": {k: v for k, v in document["camera"].items() if k != "rotation"},
             "samples": document["samples"], "floor_checks": document["floor_checks"]}
    _require(report["fixed_input_sha256"] == _canonical(fixed), "fixed input identity differs")
    _text(report["binding_scope"]); _text(report["reason"], 100)
    _require(type(report["contract"]) is dict and report["contract"].get("floor_penetration_limit_model_units") == .025,
             "saved floor threshold changed")
    _fields(report["settings"], _SETTINGS)
    for key, high in _SETTINGS.items():
        if key in ("maximum_iterations", "maximum_backtracks"):
            _require(1 <= _count(report["settings"][key], high), "invalid iteration setting")
        else:
            _number(report["settings"][key], .001, high)
    for key in ("initial_identifiability", "final_identifiability"):
        value = report[key]
        _fields(value, ("status", "rank", "variables", "regularization_used_for_rank", "global_uniqueness_claim"), extra=True)
        _require(value["status"] in ("locally_identifiable", "unidentified_or_ill_conditioned")
                 and value["regularization_used_for_rank"] is False and value["global_uniqueness_claim"] is False,
                 "local conditioning scope changed")
        _require(_count(value["rank"], 2700) <= _count(value["variables"], 2700), "conditioning rank exceeds variables")
    _gates(report["gates"], _GATES)
    _fields(provenance, ("schema_version", "profile", "status", "bindings", "input_sha256", "camera_hypothesis_sha256",
                        "geometry_snapshot_sha256", "body_manifest_sha256", "image", "camera_scope", "geometry_scope",
                        "source_scope", "source_rows", "counts", "runtime", "source_qualified", "admitted", "promoted", "physical_validation"), extra=True)
    _require(type(provenance["schema_version"]) is int and provenance["schema_version"] == 1
             and provenance["profile"] == "fixed-body-camera-root-inputs-v1"
             and provenance["status"] == "prepared_metadata_bound", "unsupported provenance profile")
    _flags(provenance)
    _require(provenance["source_qualified"] is False and provenance["input_sha256"] == sha256(raw_input).hexdigest(), "provenance input scope differs")
    _fields(provenance["bindings"], (*_BINDINGS, "root_reference_sha256", "floor_sha256"))
    _require(all(provenance["bindings"][k] == document["bindings"][k] for k in _BINDINGS), "provenance body/source identities differ")
    for key in ("camera_hypothesis_sha256", "geometry_snapshot_sha256", "body_manifest_sha256"):
        _hash(provenance[key])
    for key in ("camera_scope", "geometry_scope", "source_scope"):
        _text(provenance[key])
    image = provenance["image"]
    _fields(image, ("width", "height", "fps", "frame_count", "decoded_frames", "inference_stride", "dimension_provenance"), extra=True)
    _require(1 <= _count(image["width"], 16384) and 1 <= _count(image["height"], 16384), "source dimensions unavailable")
    _number(image["fps"], .01, 1000)
    for key in ("frame_count", "decoded_frames", "inference_stride"):
        _count(image[key], 100000)
    samples, rows, origins = document["samples"], report["samples"], provenance["source_rows"]
    _require(type(samples) is list and 2 <= len(samples) <= 900 and type(rows) is list and type(origins) is list
             and type(proposal["root_reference_candidate"]) is list
             and len(samples) == len(rows) == len(origins) == len(proposal["root_reference_candidate"]), "source-knot counts differ")
    packed, names, times, errors_before, errors_after = [], None, [], [], []
    previous_frame, known, unknown = -1, 0, 0
    for index, (sample, row, origin, candidate) in enumerate(zip(samples, rows, origins, proposal["root_reference_candidate"])):
        _fields(sample, ("evidence_id", "frame", "time_s", "decoder_timestamp_s", "root", "points"))
        _fields(row, ("evidence_id", "frame", "time_s", "decoder_timestamp_s", "root_before", "root_after", "root_delta", "fit_status", "endpoints"))
        _fields(origin, ("evidence_id", "frame", "time_s", "decoder_timestamp_s", "root_status", "target_origin", "landmark_source", "points"), extra=True)
        _fields(candidate, ("time_s", "translation"))
        frame, stamp = _count(sample["frame"], 1799), _number(sample["time_s"], 0, 60)
        _require(frame > previous_frame and (not times or 1e-4 <= stamp-times[-1] <= .25+1e-9)
                 and sample["evidence_id"] == document["bindings"]["source_sha256"] + f":frame:{frame}", "source clock/identity differs")
        previous_frame = frame; times.append(stamp)
        if sample["decoder_timestamp_s"] is not None:
            _number(sample["decoder_timestamp_s"], 0, 60)
        _require(all(sample[k] == row[k] == origin[k] for k in ("evidence_id", "frame", "time_s", "decoder_timestamp_s")), "sample evidence or clocks differ")
        before, after, delta = _vector(row["root_before"], 3), _vector(row["root_after"], 3), _vector(row["root_delta"], 3, .080000001)
        _require(before == sample["root"] and candidate == {"time_s": stamp, "translation": after}
                 and all(_near(after[i]-before[i], delta[i]) for i in range(3)), "root candidate differs")
        _require(index != 0 or delta == [0, 0, 0], "first root changed")
        _text(row["fit_status"], 100); _text(origin["root_status"], 100)
        label, descriptor = _origin(origin)
        points, endpoints, sources = sample["points"], row["endpoints"], origin["points"]
        _require(type(points) is list and 1 <= len(points) <= 12 and type(endpoints) is list and type(sources) is list
                 and len(points) == len(endpoints) == len(sources), "endpoint slots differ")
        identifiers, values = [], []
        for point, endpoint, source in zip(points, endpoints, sources):
            _fields(point, ("name", "world", "pixel", "weight")); _fields(endpoint, ("name", "pixel", "weight", "error_before_px", "error_after_px"))
            _fields(source, ("name", "status", "source_confidence", "weight", "pixel"), extra=True)
            name = _text(point["name"], 80); identifiers.append(name)
            _require(all(point[k] == endpoint[k] == source[k] for k in ("name", "pixel", "weight")), "endpoint target differs")
            xyz = _vector(point["world"], 3); weight = _number(point["weight"], 0, 1)
            _require(weight == 0 or weight >= .5, "weak point has fit weight")
            if source["source_confidence"] is not None:
                _number(source["source_confidence"], 0, 1)
            status = _text(source["status"], 100)
            observed = None if point["pixel"] is None else _vector(point["pixel"], 2, 16384)
            first = _project(xyz, document["camera"])
            second = _project([xyz[i]+delta[i] for i in range(3)], proposal["camera_candidate"])
            if weight > 0:
                _require(observed is not None, "missing target has fit weight")
                for key, projected in (("error_before_px", first), ("error_after_px", second)):
                    _require(_near(_number(endpoint[key], 0), math.hypot(*(projected[i]-observed[i] for i in range(2)))), "saved image error differs from projection")
                errors_before.append(endpoint["error_before_px"]); errors_after.append(endpoint["error_after_px"]); known += 1
            else:
                _require(endpoint["error_before_px"] is None and endpoint["error_after_px"] is None, "unavailable endpoint error must stay null")
                unknown += 1
            values.append({"name": name, "status": status, "weight": weight, "observed": observed,
                           "before": first, "after": second, "error_before": endpoint["error_before_px"], "error_after": endpoint["error_after_px"]})
        _require(len(set(identifiers)) == len(identifiers) and (names is None or names == identifiers), "endpoint identities differ")
        names = identifiers
        _require(any(p["weight"] > 0 for p in points) or delta == [0, 0, 0], "unknown root changed")
        packed.append({"evidence_id": row["evidence_id"], "frame": frame, "time_s": stamp, "decoder_timestamp_s": row["decoder_timestamp_s"],
                       "root_before": before, "root_after": after, "root_delta": delta, "fit_status": row["fit_status"],
                       "origin_label": label, "target_origin": descriptor, "landmark_source": origin["landmark_source"], "points": values})
    expected_bindings = {"root_reference_sha256": _canonical([{"time_s": r["time_s"], "translation": r["root"]} for r in samples]),
                         "floor_sha256": _canonical(document["floor"])}
    _require(all(provenance["bindings"][k] == v for k, v in expected_bindings.items()), "provenance root/floor identities differ")
    for key, expected in (("known_endpoints", known), ("unknown_endpoints", unknown)):
        _require(type(report[key]) is int and report[key] == expected and provenance["counts"].get(key) == expected, "endpoint counts differ")
    for key, values in (("image_rmse_before_px", errors_before), ("image_rmse_after_px", errors_after)):
        _require(report[key] is None if not values else _near(_number(report[key], 0), math.sqrt(sum(v*v for v in values)/len(values))), "image aggregate differs")
    no_worse = all(b <= a+1e-8 for a, b in zip(errors_before, errors_after)) if known else None
    _require(report["gates"]["image_pointwise_no_worse"] == no_worse and report["gates"]["first_root_unchanged"] is True, "image/root gates contradict samples")
    counts = provenance["counts"]
    count_keys = ("source_rows", "control_grid_rows", "floor_rows", "retained_source_slots", "known_endpoints", "unknown_endpoints")
    _fields(counts, count_keys, extra=True)
    for key in count_keys:
        _count(counts[key], 100000)
    grid_count = math.floor((times[-1]-times[0])*30+1e-9)+1
    _require(counts["source_rows"] == len(samples) and counts["retained_source_slots"] >= len(samples)
             and counts["control_grid_rows"] == grid_count, "source coverage counts differ")
    floor_summary, floor = _floor_rows(document, report, times)
    _require(counts["floor_rows"] == (0 if floor is None else len(floor)), "floor row count differs")
    diagnostic = _geometry(check, document, raw_proposal, provenance, floor)
    if diagnostic["status"] == "not_required":
        _require(all(r["root_delta"] == [0, 0, 0] for r in packed), "changed roots require geometry diagnostic")
    _fields(report["counts"], ("iterations", "evaluated_trials", "accepted_updates", "rejected_bounds", "rejected_projection",
                               "rejected_image", "rejected_floor", "rejected_conditioning", "model_constructions", "physics_steps", "inference_calls"))
    for value in report["counts"].values():
        _count(value)
    _require(all(report["counts"][k] == 0 for k in ("model_constructions", "physics_steps", "inference_calls")), "foundation execution scope differs")
    work = report["counts"]
    _require(work["iterations"] <= report["settings"]["maximum_iterations"]
             and work["accepted_updates"] <= min(work["iterations"], work["evaluated_trials"])
             and work["evaluated_trials"] <= work["iterations"]*report["settings"]["maximum_backtracks"], "proposal work counts contradict bounds")
    expected_status = ("partial" if report["reason"] == "time_limit" else "unavailable"
                       if report["initial_identifiability"]["status"] != "locally_identifiable"
                       else "proposed" if work["accepted_updates"] else "unchanged")
    _require(proposal["status"] == expected_status, "proposal status contradicts saved work")
    for key in ("maximum_camera_rotation_rad", "maximum_root_translation_model_units", "maximum_root_correction_speed_model_units_s", "elapsed_seconds"):
        _number(report[key], 0)
    root_max = max(math.sqrt(sum(v*v for v in row["root_delta"])) for row in packed)
    speed_max = max(math.sqrt(sum((b["root_delta"][i]-a["root_delta"][i])**2 for i in range(3)))
                    /(b["time_s"]-a["time_s"]) for a, b in zip(packed, packed[1:]))
    original, proposed = document["camera"]["rotation"], proposal["camera_candidate"]["rotation"]
    angle = math.acos(max(-1., min(1., (sum(original[i][j]*proposed[i][j] for i in range(3) for j in range(3))-1)/2)))
    magnitudes = {"maximum_camera_rotation_rad": angle, "maximum_root_translation_model_units": root_max,
                  "maximum_root_correction_speed_model_units_s": speed_max}
    _require(all(_near(report[key], value) for key, value in magnitudes.items()), "displayed camera/root magnitude differs")
    _require(report["gates"]["bounds"] == all(value <= report["settings"][key]+1e-12 for key, value in magnitudes.items()),
             "saved bounds gate differs")
    return {"samples": packed, "image": {k: image[k] for k in ("width", "height", "frame_count", "decoded_frames")},
            "counts": counts, "status": proposal["status"], "report": report, "provenance": provenance,
            "camera_before": document["camera"], "camera_after": proposal["camera_candidate"],
            "floor": floor_summary, "geometry": diagnostic,
            "hashes": {name: sha256(raw).hexdigest() for name, raw in (("camera-root-input.json", raw_input),
                ("proposal.json", raw_proposal), ("geometry-check.json", raw_geometry), ("target-provenance.json", raw_provenance))}}


_SCRIPT = r"""'use strict';
const data=JSON.parse(document.getElementById('data').textContent), el=id=>document.getElementById(id);
const num=v=>v===null?'Unavailable':Number(v).toFixed(5), vector=v=>v.map(num).join(', ');
function node(tag,text){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;return n;}
function mark(tag,attributes){const n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [key,value] of Object.entries(attributes))n.setAttribute(key,String(value));return n;}
function draw(){let index=Number(el('sample').value);if(!Number.isInteger(index))index=0;index=Math.max(0,Math.min(data.samples.length-1,index));
 el('sample').value=String(index);el('sample-number').value=String(index+1);el('previous').disabled=index===0;el('next').disabled=index===data.samples.length-1;
 const row=data.samples[index];el('identity').textContent=`Source knot ${index+1}/${data.samples.length} · frame ${row.frame} · nominal ${num(row.time_s)} s · decoder ${row.decoder_timestamp_s===null?'Unavailable':num(row.decoder_timestamp_s)+' s'}`;
 el('source-id').textContent=row.evidence_id;el('roots').textContent=`Original [${vector(row.root_before)}] → proposed [${vector(row.root_after)}]; correction [${vector(row.root_delta)}] model units. ${row.fit_status}`;
 el('origin').textContent=row.origin_label+' · '+row.landmark_source;el('origin-details').textContent=row.target_origin===null?'No selected-crop descriptor.':JSON.stringify(row.target_origin,null,2);
 el('points').replaceChildren();el('projection').replaceChildren();
 let outside=0;const inside=p=>p!==null&&p[0]>=0&&p[0]<=data.image.width&&p[1]>=0&&p[1]<=data.image.height;
 for(const point of row.points){const tr=node('tr');for(const value of [point.name,point.status,num(point.weight),point.observed===null?'Unavailable':vector(point.observed),num(point.error_before),num(point.error_after)])tr.append(node('td',value));el('points').append(tr);
  for(const [kind,p] of [['observed',point.observed],['before',point.before],['after',point.after]]){if(p===null)continue;if(!inside(p)){outside++;continue;}const size=Math.max(data.image.width,data.image.height)/150;
   const attrs={class:kind};let n;if(kind==='before')n=mark('rect',{...attrs,x:p[0]-size,y:p[1]-size,width:size*2,height:size*2});else n=mark('circle',{...attrs,cx:p[0],cy:p[1],r:kind==='after'?size*.65:size});
   n.append(mark('title',{}));n.children[0].textContent=point.name+' · '+kind;el('projection').append(n);}}
 el('plot-note').textContent=`Source pixel coordinates only, not a video or photograph. ${outside} projected/target markers outside the image are not clipped onto its edge. Missing targets remain unavailable.`;
}
el('sample').max=String(data.samples.length-1);el('sample').value='0';el('sample-number').max=String(data.samples.length);el('sample-number').value='1';
el('sample').addEventListener('input',draw);el('sample-number').addEventListener('change',()=>{el('sample').value=String(Number(el('sample-number').value)-1);draw();});
el('previous').addEventListener('click',()=>{el('sample').value=String(Number(el('sample').value)-1);draw();});
el('next').addEventListener('click',()=>{el('sample').value=String(Number(el('sample').value)+1);draw();});draw();
"""


def render_camera_root_report(input_document_bytes, proposal_bytes, geometry_check_bytes, provenance_bytes, *, artifact_links):
    """Return bounded offline HTML; callers authenticate file custody and publish it."""
    _require(type(artifact_links) is dict and len(artifact_links) <= len(_LINKS), "bounded local artifact links required")
    _require(all(type(name) is str and name in _LINKS and type(path) is str and path == name
                 for name, path in artifact_links.items()), "only allowlisted same-directory links are supported")
    try:
        model = _model(input_document_bytes, proposal_bytes, geometry_check_bytes, provenance_bytes)
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError, RecursionError) as error:
        raise ValueError("Camera/root report: malformed consumed metadata") from error
    report, provenance, geometry = model["report"], model["provenance"], model["geometry"]
    esc = lambda value: escape(str(value), quote=True)
    fmt = lambda value: "Unavailable" if value is None else f"{value:.6g}"
    status = {"proposed": "Camera/root proposal saved", "unchanged": "No candidate change", "partial": "Time-limited partial proposal",
              "unavailable": "Inputs do not identify this proposal"}[model["status"]]
    gates = [("Saved proposal", _GATES, report["gates"]), ("Native geometry diagnostic", _GEOMETRY_GATES, geometry["gates"])]
    gate_rows = "".join(f'<tr><td>{esc(stage)}</td><td>{esc(label)}</td><td>{"Unavailable" if values is None or values[key] is None else "Pass" if values[key] else "Fail — preserved"}</td></tr>'
                        for stage, labels, values in gates for key, label in labels.items())
    failed = [label for _, labels, values in gates if values is not None for key, label in labels.items() if values[key] is False]
    failure = '<p class="notice" role="alert">Saved failed checks: '+esc("; ".join(failed))+'. No failed check is repaired or discarded by this page.</p>' if failed else '<p class="notice">No displayed failed gate is not an admission or task-success result. Unavailable checks are not passes.</p>'
    floor = '<p>Saved floor evidence: unavailable. No zero failures or pass is inferred.</p>' if model["floor"] is None else '<table><thead><tr><th>Saved samples</th><th>Checks</th><th>Original floor failures</th><th>Proposed floor failures</th></tr></thead><tbody>'+"".join(
        f'<tr><td>{esc(kind)}</td><td>{row["checks"]}</td><td>{row["before_failures"]}</td><td>{row["after_failures"]}</td></tr>' for kind, row in model["floor"].items())+'</tbody></table>'
    geometry_text = ("Skipped: roots are unchanged. No second model or new geometry pass is claimed." if geometry["status"] == "not_required"
                     else f'Saved diagnostic completed: {geometry["counts"][0]} source and {geometry["counts"][1]} grid checks, {geometry["counts"][2]} unique evaluations. Zero physics steps; not lab admission.')
    links = " · ".join(f'<a href="{name}">{esc(name)}</a>' for name in sorted(artifact_links)) or "No artifact links supplied."
    hashes = "".join(f'<dt>{esc(name)}</dt><dd><code>{digest}</code></dd>' for name, digest in model["hashes"].items())
    identities = "".join(f'<dt>{esc(name)} (declared source binding)</dt><dd><code>{digest}</code></dd>' for name, digest in provenance["bindings"].items())
    payload = {"samples": model["samples"], "image": model["image"]}
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=True, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'none'; img-src 'none'; base-uri 'none'; form-action 'none'">
<title>Weft — camera/root inspection</title><style>
:root{{color-scheme:light}}body{{margin:0;background:#F5EADC;color:#214F50;font:16px/1.5 system-ui,sans-serif}}main{{max-width:1100px;margin:auto;padding:24px}}
h1,h2{{line-height:1.2}}section,.notice{{border:1px solid #214F5044;border-radius:12px;padding:18px;margin:18px 0;background:#fff8}}
a{{color:#965644}}button,input{{font:inherit}}button{{padding:8px 14px;background:#965644;color:white;border:0;border-radius:6px}}button:disabled{{opacity:.45}}
:focus-visible{{outline:3px solid #965644;outline-offset:3px}}table{{border-collapse:collapse;width:100%;text-align:left}}td,th{{padding:8px;border-bottom:1px solid #214F5033;vertical-align:top}}
.scroll{{overflow:auto}}code,pre{{overflow-wrap:anywhere;white-space:pre-wrap;font-size:.85em}}dl{{display:grid;grid-template-columns:minmax(180px,1fr) 2fr;gap:8px}}dd{{margin:0}}
#chart{{width:100%;max-height:460px;background:#fff;border:1px solid #214F5055}}.observed{{fill:none;stroke:#214F50;stroke-width:2}}.before{{fill:none;stroke:#965644;stroke-width:2}}.after{{fill:#247c91}}
.controls{{display:flex;gap:12px;align-items:center;flex-wrap:wrap}}input[type=range]{{flex:1;min-width:140px}}input[type=number]{{width:5em}}
</style></head><body><main><p>Weft · saved camera/root inspection</p><h1>{status}</h1>
<p><strong>Not installed · Not admitted · Physical validation unverified.</strong> This is an authored camera hypothesis and a conditional root proposal, not recovered anatomy, measured camera calibration or task success.</p>
<ol><li>Source metadata prepared: exact input document and cross-document bindings checked here. Original source files are not read by this page.</li><li>Proposal: <strong>{esc(model['status'])}</strong> — {_REASONS.get(report['reason'], 'See the saved proposal for its stop reason')}.</li><li>Geometry stage: {esc(geometry_text)}</li></ol>{failure}
<section><h2>Original versus proposed</h2><dl><dt>Known / unavailable endpoints</dt><dd>{model['counts']['known_endpoints']} / {model['counts']['unknown_endpoints']}</dd>
<dt>Image RMS error</dt><dd>{fmt(report['image_rmse_before_px'])} → {fmt(report['image_rmse_after_px'])} pixels</dd>
<dt>Maximum root correction</dt><dd>{fmt(report['maximum_root_translation_model_units'])} model units</dd><dt>Maximum correction speed</dt><dd>{fmt(report['maximum_root_correction_speed_model_units_s'])} model units/second</dd>
<dt>Camera rotation change</dt><dd>{fmt(report['maximum_camera_rotation_rad'])} radians</dd></dl>
<p>Local data sensitivity: {esc(report['initial_identifiability']['status'])} → {esc(report['final_identifiability']['status'])}. Rank is local, not evidence of a unique body/camera reconstruction.</p>
<p>Camera center, intrinsics, distortion, scale, body shape and joint angles stay fixed. Local numerical identifiability is not global uniqueness or physical accuracy.</p>
<details><summary>Authored camera and saved work counters</summary><pre>{esc(json.dumps({'original_camera':model['camera_before'],'proposed_camera':model['camera_after'],'counts':report['counts']},indent=2))}</pre></details></section>
<section><h2>Preserved checks and failures</h2><div class="scroll"><table><thead><tr><th>Stage</th><th>Check</th><th>Result</th></tr></thead><tbody>{gate_rows}</tbody></table>{floor}</div>
<p>Floor threshold: 0.025 model units. Source and grid samples can overlap in time; neither establishes continuous-time contact or a physical trial. See geometry-check.json for complete saved native diagnostics.</p></section>
<section><h2>Inspect one source knot</h2><p>{len(model['samples'])} source knots from {model['counts']['retained_source_slots']} retained source slots. Decoded count: {model['image']['decoded_frames']}; original metadata declaration: {model['image']['frame_count']}. All slots remain in source-slots.json; this view does not fill missing observations.</p>
<div class="controls"><button id="previous" type="button">Previous</button><label for="sample">Source knot</label><input id="sample" type="range" min="0" step="1"><label for="sample-number">Number</label><input id="sample-number" type="number" min="1" step="1"><button id="next" type="button">Next</button></div>
<p id="identity" aria-live="polite"></p><code id="source-id"></code><p id="roots"></p><p id="origin"></p><details><summary>Selected target provenance</summary><pre id="origin-details"></pre></details>
<p>Target: outlined circle · original projection: square · proposed projection: filled circle. Image coordinates increase rightward and downward.</p>
<svg id="chart" viewBox="0 0 {model['image']['width']} {model['image']['height']}" role="img" aria-label="Endpoint projections in source image pixel coordinates, not a photograph"><g id="projection"></g></svg><p id="plot-note"></p>
<div class="scroll"><table><thead><tr><th>Endpoint</th><th>Source status</th><th>Fit weight</th><th>Target pixels</th><th>Original error (px)</th><th>Proposed error (px)</th></tr></thead><tbody id="points"></tbody></table></div>
<noscript>Enable local JavaScript to select samples; the JSON artifact links and all summary results remain available. No network is used.</noscript></section>
<section><h2>Evidence and limits</h2><p>{esc(provenance['source_scope'])}</p><p>{esc(provenance['camera_scope'])}</p><p>{esc(provenance['geometry_scope'])}</p>
<p>Metadata bindings are distinct from observed physical truth. Selected-crop and identity-raw targets retain their original origin; missing errors remain unavailable, not zero. No physics, inference, video decode, installation or background request runs in this page.</p>
<p>{links}</p><details><summary>Exact supplied byte identities</summary><dl>{hashes}{identities}</dl></details></section>
<script type="application/json" id="data">{data}</script><script>{_SCRIPT}</script></main></body></html>'''
    output = page.encode("utf-8")
    _require(len(output) <= MAX_OUTPUT_BYTES, "HTML output exceeds its bound")
    return output
