"""Construction-only depth hypotheses; neither source depth nor executable bodies are recovered here."""

from copy import deepcopy
import hashlib
import json
import math
import time

import mujoco
import numpy as np

from weft.arm_reference_alignment import _construction_source, _endpoint, _vector, perspective_points
from weft.body_refinement import JOINT_LANDMARKS, _kinematics
from weft.human import HumanEnv, human_xml, validate_human
from weft.io import canonical_hash
from weft.world_evidence import _check_tree, _nonfinite, _unique_object


PROFILE = "perspective-root-depth-proposal-v1"
MAX_HUMAN_BYTES = 8 * 1024 * 1024
MAX_OUTPUT_BYTES = 8 * 1024 * 1024
MAX_REFERENCES = 900
DEFAULTS = {
    "maximum_depth_change_model_units": .15, "minimum_confidence": .5,
    "image_sigma_px": 6., "old_depth_sigma_model_units": .15,
    "temporal_velocity_sigma_model_units_s": .3,
    "maximum_seconds": 30., "maximum_iterations": 12, "maximum_backtracks": 8,
}
CONTRACT = {
    "ray_policy": "baseline_projected_pelvis_ray", "first_root": "unchanged",
    "minimum_confident_landmarks": 6, "minimum_camera_depth_model_units": .05,
    "maximum_temporal_gap_s": .25, "maximum_reference_poses": MAX_REFERENCES,
    "minimum_reference_interval_s": 1e-4,
    "maximum_output_bytes": MAX_OUTPUT_BYTES, "minimum_rmse_improvement_px": 1e-6,
    "numeric_tolerance": 1e-8, "linear_system_ridge": 1e-8,
    "objective_excludes": ["contacts", "equipment", "floor_distance", "task_reward", "policy_return", "held_out_evidence"],
    "identifiability": "Data-only local image sensitivity under fixed geometry/camera and an authored image sigma; not measured depth uncertainty",
    "unavailable_rows": "Retained unchanged; no image values or fit inferred through withheld/weak rows",
    "admission": "Unadmitted root array only; current mobile-body depth contract is not bypassed",
}
POINT_NAMES = tuple(JOINT_LANDMARKS)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Perspective root-depth proposal exceeded its cooperative bound")


def _settings(settings):
    if settings is not None and (not isinstance(settings, dict) or set(settings) - set(DEFAULTS)):
        raise ValueError("Unknown perspective-root settings")
    result = {**DEFAULTS, **(settings or {})}
    bounds = {
        "maximum_depth_change_model_units": (.001, .15), "minimum_confidence": (.5, 1.),
        "image_sigma_px": (1., 20.), "old_depth_sigma_model_units": (.01, .15),
        "temporal_velocity_sigma_model_units_s": (.01, 1.), "maximum_seconds": (.01, 30.),
        "maximum_iterations": (1, 12), "maximum_backtracks": (1, 8),
    }
    for name, (low, high) in bounds.items():
        value = result[name]
        if (type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high
                or name in ("maximum_iterations", "maximum_backtracks") and type(value) is not int):
            raise ValueError(f"Invalid bounded root-depth setting: {name}")
    return result


def _human(raw):
    if type(raw) is not bytes or not 1 <= len(raw) <= MAX_HUMAN_BYTES:
        raise ValueError("Root-depth input requires exact bounded human JSON bytes")
    try:
        document = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_nonfinite)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("Invalid root-depth human JSON") from error
    _check_tree(document)
    model = validate_human(document)
    if model["profile"] != "exercise-v3" or any(name in model for name in (
            "camera_reference", "arm_reference_alignment", "arm_equipment_refinement", "reference_refinement")):
        raise ValueError("Root-depth proposals require the original uncalibrated mobile body")
    references = model["reference"]
    if not 3 <= len(references) <= MAX_REFERENCES:
        raise ValueError("Root-depth proposals require 3–900 complete reference poses")
    if not isinstance(model.get("reference_evidence"), list) or len(model["reference_evidence"]) != len(references):
        raise ValueError("Each retained reference needs its original evidence identity")
    if min(np.diff([row["time_s"] for row in references])) < CONTRACT["minimum_reference_interval_s"]:
        raise ValueError("Reference intervals are too small for the bounded temporal fit")
    return model


def _camera(camera, model):
    keys = {"distance_model_units", "focal_px", "original_scale_at_anchor_model_units_per_pixel",
            "pixel_origin", "world_anchor_model_units", "world_axes"}
    if not isinstance(camera, dict) or set(camera) != keys:
        raise ValueError("Root-depth proposals require an explicit fixed perspective-camera hypothesis")
    result = deepcopy(camera)
    for name, low, high in (("distance_model_units", .1, 50.), ("focal_px", 1., 100000.),
                           ("original_scale_at_anchor_model_units_per_pixel", 1e-6, 1.)):
        value = result[name]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"Invalid fixed camera field: {name}")
    anchor = _vector(result["world_anchor_model_units"], 3, "Camera anchor")
    _vector(result["pixel_origin"], 2, "Pixel origin")
    if (result["world_axes"] != ["x_right", "y_depth", "z_up"]
            or not np.array_equal(anchor, [0., 0., model["mobility"]["base_height"]])
            or not math.isclose(result["focal_px"], result["distance_model_units"] /
                                result["original_scale_at_anchor_model_units_per_pixel"], rel_tol=0, abs_tol=1e-9)):
        raise ValueError("Camera anchor, axes or original scale differs; no implicit recalibration")
    return result


def _rows(model, source, observations, entity, confidence):
    result = []
    width, height = source["image"]["width_px"], source["image"]["height_px"]
    for index, (reference, meta) in enumerate(zip(model["reference"], model["reference_evidence"])):
        identifier = meta.get("evidence_id", "")
        prefix, separator, suffix = identifier.rpartition(":frame:")
        if not separator or prefix != model["source_sha256"] or not suffix.isdigit():
            raise ValueError("Reference must retain its exact source-frame identity")
        frame = int(suffix)
        row = observations.get(frame)
        if (row is None or meta.get("time_s") != reference["time_s"] or row["time_s"] != reference["time_s"]
                or model["root_reference"][index]["time_s"] != reference["time_s"]):
            raise ValueError("Root, body and observation clocks differ; no retiming is allowed")
        if row["admission"] == "included" and {"document_sha256": entity["document_sha256"],
                                                "evidence_id": identifier} not in row["evidence_refs"]:
            raise ValueError("Body reference does not resolve to its qualified construction evidence")
        points = [_endpoint(row, name) for name in POINT_NAMES]
        for point in points:
            pixel = point["image_px"]
            if pixel is not None and not (0 <= pixel[0] <= width and 0 <= pixel[1] <= height):
                point["status"] = "outside_source_image"
            elif point["status"] == "available" and point["confidence"] < confidence:
                point["status"] = "low_confidence"
            point.pop("raw_model_relative_depth", None)
        result.append({"reference_index": index, "source_frame": frame, "evidence_id": identifier,
                       "time_s": reference["time_s"], "source_admission": row["admission"], "points": points,
                       "status": "frozen_first_reference" if index == 0 else "pending"})
    return result


def _project(world, directions, delta, camera):
    moved = world + directions[:, None, :] * delta[:, None, None]
    depth = camera["distance_model_units"] + moved[:, :, 1]
    if (depth <= CONTRACT["minimum_camera_depth_model_units"]).any():
        raise ValueError("The complete root-depth envelope must remain in front of the fixed camera")
    focal, height = camera["focal_px"], camera["world_anchor_model_units"][2]
    xy = np.stack((moved[:, :, 0], height - moved[:, :, 2]), axis=2)
    pixels = camera["pixel_origin"] + focal * xy / depth[:, :, None]
    motion_xy = np.stack((directions[:, 0], -directions[:, 2]), axis=1)[:, None, :]
    derivative = focal * (motion_xy * depth[:, :, None] - xy) / depth[:, :, None] ** 2
    return pixels, derivative


def _tridiagonal(diagonal, off_diagonal, rhs):
    """Temporal neighbors yield a banded system; never allocate an N×N pose matrix."""
    d, b = diagonal.copy(), rhs.copy()
    for index in range(1, len(d)):
        factor = off_diagonal[index - 1] / d[index - 1]
        d[index] -= factor * off_diagonal[index - 1]
        b[index] -= factor * b[index - 1]
    solution = np.empty(len(d))
    solution[-1] = b[-1] / d[-1]
    for index in range(len(d) - 2, -1, -1):
        solution[index] = (b[index] - off_diagonal[index] * solution[index + 1]) / d[index]
    return solution


def _rmse(error, mask):
    return float(np.sqrt(np.mean(np.sum(error[mask] ** 2, axis=-1)))) if np.any(mask) else None


def propose_perspective_root_depth(human_document: bytes, observation_document: bytes, *, human_entity_ref,
                                   camera_hypothesis, settings=None):
    """Return one source-pinned, unadmitted root proposal without changing any input or running physics."""
    started = time.monotonic()
    settings = _settings(settings)
    deadline = started + settings["maximum_seconds"]
    model = _human(human_document)
    camera = _camera(camera_hypothesis, model)
    try:
        source, observations = _construction_source(observation_document, human_entity_ref, model["source_sha256"])
        rows = _rows(model, source, observations, human_entity_ref, settings["minimum_confidence"])
    except (KeyError, TypeError, IndexError, RecursionError) as error:
        raise ValueError("Malformed qualified construction evidence for root depth") from error
    original = deepcopy(model["root_reference"])
    report = {"schema_version": 1, "profile": PROFILE, "status": "preparing", "promoted": False,
              "physical_validation": "unverified", "geometry_admission": "unverified_not_executed",
              "human_sha256": hashlib.sha256(human_document).hexdigest(),
              "construction_sha256": hashlib.sha256(observation_document).hexdigest(),
              "human_entity_ref": deepcopy(human_entity_ref), "camera_hypothesis": camera,
              "camera_sha256": canonical_hash(camera), "settings": settings, "contract": deepcopy(CONTRACT),
              "source_clock": deepcopy(source["time"]), "source": deepcopy(source["source"]),
              "coordinates": "Original x-right/y-depth/z-up model units; inferred camera and body scale remain uncalibrated",
              "references": rows, "source_slots": [{key: row[key] for key in (
                  "id", "frame", "time_s", "partition", "admission", "visibility", "evidence_refs")}
                  for row in sorted(observations.values(), key=lambda item: item["frame"])],
              "fixed_field_sha256": {name: canonical_hash(value) for name, value in model.items() if name != "root_reference"},
              "iterations": 0, "backtracks": 0, "physics_steps": 0, "failures": [], "image_summary": None}
    result = {"status": "preparing", "root_reference_candidate": original, "report": report}
    env = None
    try:
        _check(deadline)
        env = HumanEnv(model)
        report["human_xml_sha256"] = hashlib.sha256(human_xml(model).encode()).hexdigest()
        data = mujoco.MjData(env.model)
        world, pelvis = [], []
        for index, reference in enumerate(model["reference"]):
            _check(deadline)
            env.mobility.reference(data, index)
            relative, _ = _kinematics(env, data, [reference["joints"][name] for name in env.joint_names])
            center = data.site_xpos[env._site_ids["pelvis"]].copy()
            world.append(relative + center)
            pelvis.append(center)
        world, pelvis = np.asarray(world), np.asarray(pelvis)
        times = np.asarray([row["time_s"] for row in original])
        directions = np.column_stack((pelvis[:, 0], np.ones(len(pelvis)),
                                      pelvis[:, 2] - camera["world_anchor_model_units"][2]))
        directions[:, (0, 2)] /= (camera["distance_model_units"] + pelvis[:, 1])[:, None]
        delta = np.zeros(len(rows))
        bound = settings["maximum_depth_change_model_units"]
        before, jacobian = _project(world, directions, delta, camera)
        # Reject the whole declared envelope instead of clipping a fit to the near-camera boundary.
        _project(world, directions, delta - bound, camera)
        targets, weights, available = before.copy(), np.zeros(before.shape[:2]), np.zeros(before.shape[:2], bool)
        for index, row in enumerate(rows):
            for point_index, point in enumerate(row["points"]):
                if point["status"] == "available":
                    targets[index, point_index] = point["image_px"]
                    weights[index, point_index] = point["confidence"]
                    available[index, point_index] = True
        sigma = settings["image_sigma_px"]
        information = np.sum(weights[:, :, None] * jacobian ** 2, axis=(1, 2)) / sigma ** 2
        conditional_scale = np.divide(1., np.sqrt(information), out=np.full(len(rows), np.inf), where=information > 0)
        eligible = (available.sum(axis=1) >= CONTRACT["minimum_confident_landmarks"]) & (conditional_scale <= bound)
        eligible[0] = False
        for index, row in enumerate(rows):
            row.update(confident_landmarks=int(available[index].sum()), data_information_per_model_unit_squared=float(information[index]),
                       conditional_noise_scale_model_units=float(conditional_scale[index]) if np.isfinite(conditional_scale[index]) else None,
                       locally_identifiable_under_fixed_hypothesis=bool(conditional_scale[index] <= bound), fit_enabled=bool(eligible[index]))
            if index:
                row["status"] = ("bounded_fit" if eligible[index] else "unchanged_insufficient_landmarks"
                                 if available[index].sum() < CONTRACT["minimum_confident_landmarks"] else "unchanged_flat_likelihood")
        prior = 1 / settings["old_depth_sigma_model_units"] ** 2
        intervals = np.diff(times)
        smooth = np.where(intervals <= CONTRACT["maximum_temporal_gap_s"],
                          1 / (settings["temporal_velocity_sigma_model_units_s"] * intervals) ** 2, 0.)

        def objective(change):
            pixels, _ = _project(world, directions, change, camera)
            return float(.5 * np.sum(weights[:, :, None] * ((pixels - targets) / sigma) ** 2)
                         + .5 * prior * (change @ change) + .5 * np.sum(smooth * np.diff(change) ** 2))

        report["objective_before"] = objective(delta)
        exhausted = False
        if np.any(eligible):
            for iteration in range(settings["maximum_iterations"]):
                _check(deadline)
                pixels, jacobian = _project(world, directions, delta, camera)
                gradient = np.sum(weights[:, :, None] * jacobian * (pixels - targets), axis=(1, 2)) / sigma ** 2 + prior * delta
                diagonal = np.sum(weights[:, :, None] * jacobian ** 2, axis=(1, 2)) / sigma ** 2 + prior
                difference = np.diff(delta) * smooth
                gradient[:-1] -= difference; gradient[1:] += difference
                diagonal[:-1] += smooth; diagonal[1:] += smooth
                off = -smooth.copy()
                off[~(eligible[:-1] & eligible[1:])] = 0.
                diagonal[~eligible], gradient[~eligible] = 1., 0.
                direction = _tridiagonal(diagonal + CONTRACT["linear_system_ridge"], off, -gradient)
                old_objective, accepted = objective(delta), False
                for backtrack in range(settings["maximum_backtracks"]):
                    _check(deadline)
                    candidate = np.clip(delta + direction * 2. ** -backtrack, -bound, bound)
                    candidate[~eligible] = 0.
                    report["backtracks"] += 1
                    if objective(candidate) < old_objective - 1e-12:
                        accepted = True
                        break
                report["iterations"] = iteration + 1
                if not accepted or np.max(np.abs(candidate - delta)) <= 1e-10:
                    break
                delta = candidate
            else:
                exhausted = True
        after, final_jacobian = _project(world, directions, delta, camera)
        candidate_roots = deepcopy(original)
        for index, root in enumerate(candidate_roots):
            if delta[index] != 0:
                root["translation"] = (np.asarray(root["translation"]) + directions[index] * delta[index]).tolist()
        before_error, after_error = before - targets, after - targets
        per_joint = {name: {"samples": int(available[:, index].sum()),
                           "before_rmse_px": _rmse(before_error[:, index], available[:, index]),
                           "after_rmse_px": _rmse(after_error[:, index], available[:, index])}
                     for index, name in enumerate(POINT_NAMES)}
        summary = {"confident_samples": int(available.sum()), "before_rmse_px": _rmse(before_error, available),
                   "after_rmse_px": _rmse(after_error, available), "per_joint": per_joint,
                   "unchanged_first_and_unsupported_rows_included": True}
        failures = []
        if not np.any(eligible):
            failures.append("no_locally_identifiable_supported_depth_rows")
        if (summary["before_rmse_px"] is None or summary["after_rmse_px"] >=
                summary["before_rmse_px"] - CONTRACT["minimum_rmse_improvement_px"]):
            failures.append("no_confident_image_improvement")
        if any(row["samples"] and row["after_rmse_px"] > row["before_rmse_px"] + CONTRACT["numeric_tolerance"]
               for row in per_joint.values()):
            failures.append("per_joint_confident_image_regression")
        if np.any(np.abs(delta) >= bound - 1e-8):
            failures.append("authored_depth_bound_active")
        if exhausted:
            failures.append("iteration_limit_reached")
        for index, row in enumerate(rows):
            row["depth_change_model_units"] = float(delta[index])
            row["root_delta_model_units"] = (directions[index] * delta[index]).tolist()
            row["data_information_after_per_model_unit_squared"] = float(np.sum(
                weights[index, :, None] * final_jacobian[index] ** 2) / sigma ** 2)
            for point_index, point in enumerate(row["points"]):
                point.update(projected_before_px=before[index, point_index].tolist(), projected_after_px=after[index, point_index].tolist(),
                             fit_weight=float(weights[index, point_index]) if eligible[index] else 0.)
                if point["image_px"] is not None:
                    point["error_before_px"] = float(np.linalg.norm(before[index, point_index] - point["image_px"]))
                    point["error_after_px"] = float(np.linalg.norm(after[index, point_index] - point["image_px"]))
            hip_points = [row["points"][POINT_NAMES.index(name)] for name in ("left_hip", "right_hip")]
            pelvis_pixel, _ = perspective_points(pelvis[index:index + 1], camera)
            moved_pelvis_pixel, _ = perspective_points(pelvis[index:index + 1] + directions[index:index + 1] * delta[index], camera)
            row["pelvis_ray_error_px"] = float(np.max(np.abs(pelvis_pixel - moved_pelvis_pixel)))
            row["observed_hip_midpoint_residual_px"] = (float(np.linalg.norm(pelvis_pixel[0] - np.mean(
                [point["image_px"] for point in hip_points], axis=0))) if all(point["status"] == "available" for point in hip_points) else None)
        status = ("unsupported" if not np.any(eligible) else "rejected_image_hypothesis" if failures
                  else "image_hypothesis_unadmitted")
        report.update(status=status, image_summary=summary, failures=failures, objective_after=objective(delta),
                      changed_reference_poses=int(np.count_nonzero(delta)), maximum_depth_change_model_units=float(np.max(np.abs(delta))),
                      maximum_depth_correction_rate_model_units_s=float(np.max(np.abs(np.diff(delta) / intervals))),
                      first_root_unchanged=candidate_roots[0] == original[0],
                      maximum_pelvis_ray_error_px=max(row["pelvis_ray_error_px"] for row in rows))
        result.update(status=status, root_reference_candidate=candidate_roots)
    except TimeoutError as error:
        report.update(status="partial_budget_exhausted", failures=[str(error)])
        result["status"] = report["status"]
    finally:
        if env is not None:
            env.close()
    report["seconds"] = time.monotonic() - started
    report["root_reference_sha256"] = canonical_hash(result["root_reference_candidate"])
    if len(json.dumps(result, allow_nan=False, separators=(",", ":")).encode()) > MAX_OUTPUT_BYTES:
        raise ValueError("Root-depth proposal exceeds its output bound")
    return result
