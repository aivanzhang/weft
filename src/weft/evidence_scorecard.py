"""Bounded evidence summaries, not grades or reconstruction-accuracy claims.

The caller verifies source manifests and normalizes selected documents first.
This module performs no I/O and never promotes build readiness into test evidence.
"""

import json
import math
from pathlib import PurePosixPath
import re


PROFILE = "weft-evidence-scorecard-v1"
MAX_BYTES = 64 * 1024
MAX_METRICS = 16
MAX_EXAMPLES = 8
MAX_POINTERS = 8
_SHA = re.compile(r"[0-9a-f]{64}")
_PARTITIONS = {"construction", "exposed_development", "unknown"}
DIMENSIONS = (
    ("execution_actions", "Execution and action validity", "seeded_execution_trials", "trials",
     "Build readiness does not verify legal actions, finite states, termination or seeded replay."),
    ("perception_identity", "Perception and identity", "independent_identity_errors", "errors",
     "Model-reported coverage is not identity accuracy against independent annotations."),
    ("geometry_camera", "Geometry and camera", "independent_3d_error", "unknown",
     "Calibrated geometry and camera error require independent measurements; model confidence is not accuracy."),
    ("temporal_dynamics", "Temporal dynamics", "independent_trajectory_error", "unknown",
     "Source-reference gaps do not measure simulated dynamics, event order or contact timing."),
    ("physical_relationships", "Physical relationships", "measured_contact_error", "unknown",
     "Static reference overlap is not dynamic stability, measured support, force or grasp."),
    ("hidden_state_continuity", "Hidden-state continuity", "independent_hidden_state_error", "unknown",
     "Missing or interpolated states are not observations or verified persistent identities."),
    ("parameter_validity", "Parameter validity", "independently_measured_parameters", "parameters",
     "Body scale, depth, masses, friction and support forces remain priors without independent measurements."),
    ("causal_response", "Causal response", "bound_intervention_trials", "trials",
     "Changed-action and appearance-invariance checks are unavailable in this build-summary adapter."),
    ("long_run_stability", "Long-run stability", "long_rollout_stress_trials", "trials",
     "Completing a recorded task interval does not establish longer-rollout, reset or timestep robustness."),
    ("robustness_coverage", "Robustness and coverage", "fresh_final_sources", "sources",
     "One build and exposed development starts cannot establish diverse-source or fresh-final generalization."),
    ("uncertainty_abstention", "Uncertainty and abstention", "independent_unsupported_claim_rate", "fraction",
     "Reported uncertainty is not a measured unsupported-claim rate or a calibrated confidence interval."),
    ("task_learning_utility", "Task and learning utility", "physical_transfer_trials", "trials",
     "Saved simulator outcomes are not independently replayed here and do not establish physical transfer."),
    ("reproducibility_integrity", "Reproducibility and integrity", "independent_physical_authenticity", "checks",
     "Only selected manifest members were verified by the caller; hashes do not authenticate physical truth."),
    ("practical_efficiency", "Practical efficiency", "total_run_cost", "USD",
     "Peak memory, human intervention and aggregate failed/successful-run cost need separate bound receipts."),
)


def _object(value, name):
    if type(value) is not dict:
        raise ValueError(f"Scorecard {name} must be an object")
    return value


def _number(value, name, *, maximum=1e12, integer=False, minimum=0):
    types = (int,) if integer else (int, float)
    if type(value) not in types or not minimum <= value <= maximum or not math.isfinite(value):
        raise ValueError(f"Scorecard {name} is invalid or outside its bound")
    return value


def _text(value, name, maximum=512):
    if (not isinstance(value, str) or len(value) > maximum
            or any(ord(char) < 32 and char not in "\t\n\r" for char in value)):
        raise ValueError(f"Scorecard {name} is invalid or oversized")
    return value


def _hash(value, name):
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError(f"Scorecard {name} requires an exact SHA-256")
    return value


def _path(value, name):
    value = _text(value, name)
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or "\\" in value or ":" in value
            or any(ord(char) < 32 for char in value)
            or any(part in ("", ".", "..") for part in value.split("/"))):
        raise ValueError(f"Scorecard {name} must be a local relative artifact path")
    return value


def _pointers(summary):
    artifacts = summary.get("artifacts", [])
    if not isinstance(artifacts, list) or len(artifacts) > 128:
        raise ValueError("Scorecard artifact count exceeds the report contract")
    result = {}
    for item in artifacts:
        item = _object(item, "artifact")
        name = _path(item.get("source_name"), "artifact identity")
        if name in result:
            raise ValueError("Scorecard artifact identities must be unique")
        result[name] = {"source_name": name, "sha256": _hash(item.get("sha256"), "artifact hash"),
                        "path": _path(item.get("path"), "artifact link")}
    return result


def _metric(identifier, label, value=None, unit="unknown", *, numerator=None, denominator=None, gate=None):
    if numerator is not None or denominator is not None:
        _number(numerator, identifier + " numerator", integer=True)
        _number(denominator, identifier + " denominator", integer=True)
        if numerator > denominator:
            raise ValueError("Scorecard numerator exceeds denominator")
    return {"id": identifier, "label": label, "value": value, "unit": unit,
            "numerator": numerator, "denominator": denominator, "recorded_gate": gate,
            "evidence_kind": "unknown", "partition": "unknown", "evidence": []}


def _attach(row, metrics, pointers, *, kind, partition, failed=False):
    row["metrics"].extend(metrics)
    if metrics:
        row["status"] = "failure_reported" if failed or row["status"] == "failure_reported" else "evidence_available"
    for pointer in pointers:
        if pointer not in row["evidence"]:
            row["evidence"].append(dict(pointer))
    for metric in metrics:
        metric["evidence_kind"], metric["partition"] = kind, partition
        metric["evidence"] = [dict(pointer) for pointer in pointers]
    observed = [metric for metric in row["metrics"] if metric["value"] is not None]
    for key in ("evidence_kind", "partition"):
        values = {metric[key] for metric in observed}
        row[key] = next(iter(values)) if len(values) == 1 else "unknown"


def _coverage(summary, source_sha):
    coverage = _object(summary.get("coverage") or {}, "coverage")
    if coverage.get("status") != "available":
        return None
    observed_source = _hash(coverage.get("source_sha256"), "coverage source")
    if source_sha is not None and observed_source != source_sha:
        raise ValueError("Scorecard coverage belongs to a different source")
    names = ("decoded_frames", "sampled_frames", "unsampled_frames", "usable_samples",
             "partial_samples", "unusable_samples", "competing_samples")
    counts = {key: _number(coverage.get(key), key, integer=True, maximum=1800) for key in names}
    if (counts["sampled_frames"] + counts["unsampled_frames"] != counts["decoded_frames"]
            or counts["usable_samples"] + counts["unusable_samples"] != counts["sampled_frames"]
            or counts["partial_samples"] > counts["usable_samples"]
            or counts["competing_samples"] > counts["unusable_samples"]):
        raise ValueError("Scorecard coverage counts contradict each other")
    issues = coverage.get("issues", [])
    if not isinstance(issues, list) or len(issues) > 1800:
        raise ValueError("Scorecard coverage issues exceed the source bound")
    examples, seen = [], set()
    for item in issues:
        item = _object(item, "coverage issue")
        kind = _text(item.get("kind"), "coverage issue kind", 80)
        first = _number(item.get("first_frame"), "first frame", integer=True, maximum=1799)
        last = _number(item.get("last_frame"), "last frame", integer=True, maximum=1799)
        stamp = _number(item.get("seek_s"), "source seek time", maximum=60)
        if first > last or last >= counts["decoded_frames"] or (kind, first, last) in seen:
            raise ValueError("Scorecard coverage issue identities conflict")
        seen.add((kind, first, last))
        if len(examples) < MAX_EXAMPLES:
            examples.append({"kind": kind, "first_frame": first, "last_frame": last,
                             "time_s": stamp, "clock": "saved_source_seek_clock"})
    movement = _object(coverage.get("movement", {}), "movement")
    gaps = {}
    for key in ("long_gaps", "unfilled_gaps", "maximum_gap_s"):
        if movement.get(key) is not None:
            gaps[key] = _number(movement[key], key, maximum=60 if key.endswith("_s") else 1800,
                                integer=not key.endswith("_s"))
    return counts, examples, len(issues), gaps


def _feasibility(summary, pointers, source_sha, scene_sha):
    value = _object(summary.get("reference_feasibility") or {}, "reference feasibility")
    if value.get("status") not in ("feasible", "infeasible"):
        return None
    for key, expected in (("source_sha256", source_sha), ("scene_sha256", scene_sha)):
        if key == "source_sha256" and value.get("source_binding_mode") == "analytic_fixture":
            if value.get(key) is not None or expected is not None:
                raise ValueError("Scorecard analytic feasibility contradicts source evidence")
            continue
        actual = _hash(value.get(key), "feasibility " + key)
        if expected is not None and actual != expected:
            raise ValueError("Scorecard feasibility identities conflict")
    name = _path(value.get("audit_artifact"), "feasibility artifact")
    digest = _hash(value.get("audit_sha256"), "feasibility artifact hash")
    if name not in pointers or pointers[name]["sha256"] != digest:
        raise ValueError("Scorecard feasibility lacks its exact copied artifact")
    parts = {}
    for key, bound in (("source_pose", 900), ("control_grid", 1801)):
        row = _object(value.get(key), key)
        parts[key] = {field: _number(row.get(field), key + " " + field, integer=True, maximum=bound)
                      for field in ("checks", "floor_failures", "failed_preflight")}
        if not parts[key]["floor_failures"] <= parts[key]["failed_preflight"] <= parts[key]["checks"]:
            raise ValueError("Scorecard feasibility counts conflict")
    total = sum(row["checks"] for row in parts.values())
    static = _number(value.get("static_conflict_poses"), "static conflicts", integer=True, maximum=total)
    initial = _number(value.get("initial_cross_entity_intersections"), "initial intersections", integer=True, maximum=4096)
    limit = _number(value.get("floor_limit_model_units"), "floor gate", maximum=1e6)
    peak = _number(value.get("max_penetration_model_units"), "floor penetration", maximum=1e6)
    if type(value.get("physics_steps")) is not int or value["physics_steps"] != 0:
        raise ValueError("Scorecard reference feasibility must remain kinematic")
    failed = any(row["failed_preflight"] for row in parts.values()) or bool(initial)
    floor_failed = any(row["floor_failures"] for row in parts.values())
    if ((value["status"] == "infeasible") != failed or floor_failed != (peak > limit)
            or static > sum(row["failed_preflight"] for row in parts.values())):
        raise ValueError("Scorecard feasibility status contradicts its saved counts")
    examples = value.get("failure_examples", [])
    if not isinstance(examples, list) or len(examples) > MAX_EXAMPLES:
        raise ValueError("Scorecard feasibility examples exceed the bound")
    omitted = _number(value.get("failure_examples_omitted", 0), "omitted failures", integer=True, maximum=2701)
    if len(examples) + omitted != sum(row["failed_preflight"] for row in parts.values()):
        raise ValueError("Scorecard feasibility examples contradict failure counts")
    result, seen = [], set()
    for item in examples:
        item = _object(item, "feasibility example")
        kind = item.get("kind")
        if kind not in parts:
            raise ValueError("Scorecard feasibility example kind is unknown")
        stamp = _number(item.get("time_s"), "reference time", maximum=3600)
        index = item.get("source_reference_index")
        if index is not None:
            _number(index, "source reference index", integer=True, maximum=899)
        if (kind, stamp, index) in seen:
            raise ValueError("Scorecard feasibility examples have duplicate identities")
        seen.add((kind, stamp, index))
        penetration = _number(item.get("max_penetration_model_units"), "example penetration", maximum=peak)
        static_count = _number(item.get("static_conflict_count"), "example static conflicts", integer=True, maximum=4096)
        if (item.get("passes_floor_limit") is not (penetration <= limit)
                or penetration <= limit and not static_count):
            raise ValueError("Scorecard failure example contradicts its floor gate")
        result.append({"kind": kind, "time_s": stamp, "source_reference_index": index,
                       "clock": "saved_reference_clock", "max_penetration_model_units": penetration,
                       "floor_limit_model_units": limit, "passes_floor_limit": penetration <= limit,
                       "static_conflict_count": static_count})
    return parts, static, initial, limit, peak, failed, result, omitted, pointers[name]


def _learning(summary, scene_sha, task_sha):
    value = _object(summary.get("learning") or {}, "learning")
    if not value:
        return None, "No saved learning/evaluation result is available."
    identities = {key: _hash(value[key], "learning " + key) for key in ("scene_sha256", "task_sha256") if key in value}
    for key, expected in (("scene_sha256", scene_sha), ("task_sha256", task_sha)):
        if expected is not None and key in identities and identities[key] != expected:
            raise ValueError("Scorecard learning and build identities conflict")
    if scene_sha is None or task_sha is None or len(identities) != 2:
        return None, "Saved learning summary is not bound to both build scene and task; counts are unavailable here."
    training = _object(value.get("training") or {}, "training")
    selection = _object(value.get("selection") or {}, "selection")
    evaluation = _object(value.get("evaluation") or {}, "evaluation")
    train = {key: _number(training[key], "training " + key, integer=True, maximum=10**9)
             for key in ("evaluated_episodes", "control_steps") if key in training}
    selected = {key: _number(selection[key], "selection " + key, integer=True, maximum=10**6)
                for key in ("feasible_candidates", "unique_candidates") if key in selection}
    if (set(selected) == {"feasible_candidates", "unique_candidates"}
            and selected["feasible_candidates"] > selected["unique_candidates"]):
        raise ValueError("Scorecard feasible candidates exceed unique candidates")
    selection_note = ""
    if selection.get("selection_status") is not None:
        selection_note = " Saved selection: " + _text(selection["selection_status"], "selection status", 128) + "."
    if selection.get("evaluation_used") is not None and type(selection["evaluation_used"]) is not bool:
        raise ValueError("Scorecard selection evaluation use must be a boolean")
    if selection.get("evaluation_used"):
        selection_note += " Evaluation was used for selection; this is exposed development evidence."
    if not evaluation:
        return {"training": train, "selection": selected, "controllers": {}, "trials": None,
                "starts": None, "failed_resets": None, "evaluation_steps": None}, "No saved evaluation trials are available." + selection_note
    trials = _number(evaluation.get("trials_per_controller"), "evaluation trials", integer=True, minimum=1, maximum=2000)
    if evaluation.get("comparison_complete") is not None and type(evaluation["comparison_complete"]) is not bool:
        raise ValueError("Scorecard comparison completeness must be a boolean")
    if evaluation.get("comparison_complete") is not True:
        selection_note += " Comparison completeness is not established by the saved summary."
    seeds = evaluation.get("evaluation_seeds")
    if seeds is not None:
        if not isinstance(seeds, list) or len(seeds) != trials:
            raise ValueError("Scorecard evaluation seeds differ from trial count")
        for seed in seeds:
            _number(seed, "evaluation seed", integer=True, maximum=2**32 - 1)
        if len(set(seeds)) != len(seeds):
            raise ValueError("Scorecard evaluation seed identities must be unique")
    starts = evaluation.get("independent_changed_states")
    if starts is not None:
        _number(starts, "changed starts", integer=True, maximum=trials)
    steps = evaluation.get("control_steps")
    if steps is not None:
        _number(steps, "evaluation controls", integer=True, maximum=10**9)
    controllers = _object(evaluation.get("controllers", {}), "controllers")
    if len(controllers) > 4:
        raise ValueError("Scorecard supports at most four saved controller summaries")
    records = {}
    for name, row in sorted(controllers.items()):
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", name):
            raise ValueError("Scorecard controller identity is invalid")
        row = _object(row, "controller")
        counts = {key: _number(row.get(key), name + " " + key, integer=True, maximum=trials)
                  for key in ("successes", "completed_intervals", "reset_failures")}
        if counts["successes"] > counts["completed_intervals"] or counts["completed_intervals"] + counts["reset_failures"] > trials:
            raise ValueError("Scorecard controller counts contradict each other")
        counts["mean_score"] = _number(row["mean_score"], name + " score", minimum=-1e12) if row.get("mean_score") is not None else None
        records[name] = counts
    return {"training": train, "selection": selected, "controllers": records, "trials": trials,
            "starts": starts, "failed_resets": sum(row["reset_failures"] for row in records.values()),
            "evaluation_steps": steps}, "Saved simulation summaries only; matching identities do not independently replay policy or trials." + selection_note


def _stage_time(summary):
    history = summary.get("stage_history", [])
    if not isinstance(history, list) or len(history) > 2048:
        raise ValueError("Scorecard stage history exceeds the report bound")
    previous = None
    for row in history:
        row = _object(row, "stage event")
        seconds = _number(row.get("elapsed_seconds"), "stage elapsed time", maximum=86400)
        if previous is not None and seconds < previous:
            raise ValueError("Scorecard saved stage clock is reversed")
        previous = seconds
    return previous


def build_evidence_scorecard(summary):
    """Summarize caller-verified report facts; unknown or unbound measures stay null."""
    summary = _object(summary, "summary")
    receipt_only = summary.get("source_seal_scope") == "failure_receipt_only"
    if receipt_only:
        # A failed seal covers the receipt, not even plausible-looking stage metadata.
        summary = {key: summary[key] for key in ("source_seal_scope", "source_manifest_sha256", "build_status",
                   "evidence_partition", "artifacts", "stage_history", "reasons") if key in summary}
    partition = summary.get("evidence_partition", "unknown")
    if partition not in _PARTITIONS:
        raise ValueError("This build adapter cannot establish fresh-final evidence")
    source = _object(summary.get("source") or {}, "source")
    source_sha = _hash(source["sha256"], "source identity") if "sha256" in source else None
    scene_sha = _hash(summary["scene_sha256"], "scene identity") if summary.get("scene_sha256") is not None else None
    task_sha = _hash(summary["task_sha256"], "task identity") if summary.get("task_sha256") is not None else None
    manifest_sha = _hash(summary["source_manifest_sha256"], "source manifest") if summary.get("source_manifest_sha256") is not None else None
    pointers = _pointers(summary)
    if receipt_only and any(name != "result.json" for name in pointers):
        raise ValueError("Scorecard failure receipt includes unsealed artifact pointers")
    manifest_pointer = ([{"source_name": "manifest.json", "sha256": manifest_sha, "path": None}]
                        if manifest_sha is not None else [])
    rows = {}
    for identifier, label, metric, unit, limit in DIMENSIONS:
        rows[identifier] = {"id": identifier, "label": label, "status": "unavailable", "evidence_kind": "unknown",
                            "partition": "unknown", "metrics": [_metric(metric, metric.replace("_", " "), unit=unit)],
                            "failure_examples": [], "failure_examples_omitted": 0,
                            "evidence": [], "uncertainty_and_missing_measurements": [limit]}
    coverage = _coverage(summary, source_sha)
    if coverage:
        counts, examples, issue_count, gaps = coverage
        metrics = [_metric(key, key.replace("_", " "), counts[key], "frames") for key in
                   ("decoded_frames", "sampled_frames", "unsampled_frames", "unusable_samples", "partial_samples", "competing_samples")]
        metrics.append(_metric("usable_sample_count", "Model-reported usable samples", counts["usable_samples"], "samples",
                               numerator=counts["usable_samples"], denominator=counts["sampled_frames"]))
        _attach(rows["perception_identity"], metrics, manifest_pointer, kind="construction", partition=partition,
                failed=counts["unusable_samples"] > 0)
        rows["perception_identity"]["failure_examples"] = examples
        rows["perception_identity"]["failure_examples_omitted"] = issue_count - len(examples)
        _attach(rows["hidden_state_continuity"], [_metric("unobserved_slots", "Unsampled or unusable source slots",
                counts["unsampled_frames"] + counts["unusable_samples"], "frames")], manifest_pointer,
                kind="construction", partition=partition)
        _attach(rows["uncertainty_abstention"], [_metric("reported_issue_spans", "Saved source issue spans", issue_count, "spans")],
                manifest_pointer, kind="construction", partition=partition, failed=counts["unusable_samples"] > 0)
        rows["uncertainty_abstention"]["failure_examples"] = [dict(row) for row in examples]
        rows["uncertainty_abstention"]["failure_examples_omitted"] = issue_count - len(examples)
        _attach(rows["temporal_dynamics"], [_metric(key, key.replace("_", " "), value, "seconds" if key.endswith("_s") else "gaps")
                for key, value in gaps.items()], manifest_pointer, kind="construction", partition=partition,
                failed=bool(gaps.get("long_gaps") or gaps.get("unfilled_gaps")))
    feasibility = _feasibility(summary, pointers, source_sha, scene_sha)
    if feasibility:
        parts, static, initial, limit, peak, failed, examples, omitted, pointer = feasibility
        metrics = [_metric(key + "_floor_failures", key.replace("_", " ") + " floor-limit failures",
                           value["floor_failures"], "poses", numerator=value["floor_failures"], denominator=value["checks"])
                   for key, value in parts.items()]
        metrics.extend([_metric("peak_floor_penetration", "Saved peak floor penetration", peak, "model_units",
                                gate={"name": "max_floor_penetration_model_units", "operator": "<=", "value": limit, "unit": "model_units"}),
                        _metric("static_conflict_poses", "Reference poses with static conflicts", static, "poses"),
                        _metric("initial_intersections", "Initial cross-entity intersections", initial, "intersections")])
        _attach(rows["physical_relationships"], metrics, [pointer], kind="software_check", partition=partition, failed=failed)
        rows["physical_relationships"]["failure_examples"] = examples
        rows["physical_relationships"]["failure_examples_omitted"] = omitted
        _attach(rows["geometry_camera"], [_metric("kinematic_geometry_checks", "Saved reference/grid geometry checks",
                sum(value["checks"] for value in parts.values()), "poses")], [pointer], kind="software_check", partition=partition, failed=failed)
    options = _object(summary.get("requested_options") or {}, "requested options")
    if options.get("body_scale") is not None:
        scale = _number(options["body_scale"], "body scale prior", maximum=1e6)
        _attach(rows["parameter_validity"], [_metric("authored_body_scale_prior", "Authored body-to-world scale prior", scale,
                "model_unit_scale")], manifest_pointer, kind="construction", partition=partition)
    learning, learning_note = _learning(summary, scene_sha, task_sha)
    rows["task_learning_utility"]["uncertainty_and_missing_measurements"].append(learning_note)
    if "root_assistance" in options:
        if type(options["root_assistance"]) is not bool:
            raise ValueError("Scorecard root assistance must be an explicit boolean")
        rows["task_learning_utility"]["uncertainty_and_missing_measurements"].append(
            "Three virtual root controls were requested; this is not unassisted balance."
            if options["root_assistance"] else "Root assistance was not requested; this is not a verified balance result.")
    if learning:
        rows["task_learning_utility"]["uncertainty_and_missing_measurements"].append(
            "Training or unchanged selection is not evidence of learned improvement over the authored controller.")
        proof = [pointers[name] for name in ("result.json", "learning_workflow/result.json") if name in pointers] or manifest_pointer
        metrics = []
        for name, value in learning["controllers"].items():
            metrics.extend([_metric(name + "_successes", name + " saved simulator successes", value["successes"], "trials",
                                    numerator=value["successes"], denominator=learning["trials"]),
                            _metric(name + "_completed_intervals", name + " completed task intervals", value["completed_intervals"], "trials",
                                    numerator=value["completed_intervals"], denominator=learning["trials"]),
                            _metric(name + "_mean_reward", name + " saved mean reward", value["mean_score"], "reward")])
        metrics.extend(_metric("training_" + key, "Training " + key.replace("_", " "), value,
                               "episodes" if key == "evaluated_episodes" else "actions") for key, value in learning["training"].items())
        # Keep source summaries bounded without conflating candidate feasibility with task evaluation.
        if "feasible_candidates" in learning["selection"]:
            selected = learning["selection"]
            fraction = ({"numerator": selected["feasible_candidates"], "denominator": selected["unique_candidates"]}
                        if "unique_candidates" in selected else {})
            metrics.append(_metric("training_feasible_candidates", "Training-feasible candidates", selected["feasible_candidates"],
                                   "candidates", **fraction))
        _attach(rows["task_learning_utility"], metrics, proof, kind="simulated_development", partition="exposed_development",
                failed=any(row["successes"] < learning["trials"] for row in learning["controllers"].values()))
        if learning["controllers"]:
            total = learning["trials"] * len(learning["controllers"])
            completed = sum(row["completed_intervals"] for row in learning["controllers"].values())
            _attach(rows["long_run_stability"], [_metric("completed_task_intervals", "Saved completed task intervals across controllers",
                    completed, "trials", numerator=completed, denominator=total)], proof,
                    kind="simulated_development", partition="exposed_development", failed=completed < total)
            if learning["failed_resets"]:
                _attach(rows["execution_actions"], [_metric("saved_reset_failures", "Reported simulator reset failures",
                        learning["failed_resets"], "trials", numerator=learning["failed_resets"], denominator=total)], proof,
                        kind="simulated_development", partition="exposed_development", failed=True)
        if learning["starts"] is not None:
            _attach(rows["robustness_coverage"], [_metric("saved_changed_starts", "Saved development changed starts",
                    learning["starts"], "starts")], proof, kind="simulated_development", partition="exposed_development")
        if learning["evaluation_steps"] is not None:
            _attach(rows["practical_efficiency"], [_metric("evaluation_actions", "Saved evaluation control actions",
                    learning["evaluation_steps"], "actions")], proof, kind="simulated_development", partition="exposed_development")
        if "control_steps" in learning["training"]:
            _attach(rows["practical_efficiency"], [_metric("training_actions", "Saved training control actions",
                    learning["training"]["control_steps"], "actions")], proof,
                    kind="simulated_development", partition="exposed_development")
    if source_sha is not None:
        _attach(rows["robustness_coverage"], [_metric("source_count", "Source videos represented by this build", 1, "sources")],
                manifest_pointer, kind="construction", partition=partition)
    if manifest_sha is not None:
        _attach(rows["reproducibility_integrity"], [_metric("copied_artifact_identities", "Selected copied artifact identities",
                len(pointers), "artifacts")], manifest_pointer, kind="software_check", partition=partition)
        rows["reproducibility_integrity"]["uncertainty_and_missing_measurements"].append(
            "Build state: " + _text(summary.get("build_status", "unknown"), "build state", 80) + "; readiness is not execution or accuracy.")
    reasons = summary.get("reasons", [])
    if not isinstance(reasons, list) or len(reasons) > 34:
        raise ValueError("Scorecard saved reasons exceed the report bound")
    if reasons:
        reason_examples = []
        for item in reasons:
            item = _object(item, "saved reason")
            example = {key: _text(item[key], "saved reason " + key, 2000)[:512]
                       for key in ("stage", "code", "error_type", "type", "message") if key in item}
            if len(reason_examples) < MAX_EXAMPLES:
                reason_examples.append(example)
        row = rows["uncertainty_abstention"]
        _attach(row, [_metric("saved_build_reasons", "Saved build failure reasons", len(reasons), "reasons")],
                [pointers["result.json"]] if "result.json" in pointers else manifest_pointer,
                kind="software_check", partition=partition, failed=True)
        combined = row["failure_examples"] + reason_examples
        row["failure_examples_omitted"] += len(reasons) - min(len(reasons), MAX_EXAMPLES - len(row["failure_examples"]))
        row["failure_examples"] = combined[:MAX_EXAMPLES]
    if receipt_only:
        rows["uncertainty_abstention"]["status"] = "failure_reported"
        rows["uncertainty_abstention"]["uncertainty_and_missing_measurements"].append(
            "Only the terminal failure receipt was verified; unsealed stage outputs are not evidence.")
    elapsed = _stage_time(summary)
    if elapsed is not None:
        _attach(rows["practical_efficiency"], [_metric("last_recorded_stage_elapsed", "Last recorded cumulative build-stage time",
                elapsed, "seconds")], [pointers["result.json"]] if "result.json" in pointers else manifest_pointer,
                kind="construction", partition=partition)
        rows["practical_efficiency"]["uncertainty_and_missing_measurements"].append(
            "The saved stage clock is not summed and may exclude final packaging, reporting or failed attempts.")
    result = {"schema_version": 1, "profile": PROFILE, "source_manifest_sha256": manifest_sha,
              "scope": "Evidence availability and saved failures, not a completion percentage or independent physical grade.",
              "identity_scope": "Matching sealed summary identities only; no policy/trial replay or full-package admission.",
              "identities": {"source_sha256": source_sha, "scene_sha256": scene_sha, "task_sha256": task_sha},
              "rows": list(rows.values())}
    for row in result["rows"]:
        if len(row["metrics"]) > MAX_METRICS or len(row["evidence"]) > MAX_POINTERS or len(row["failure_examples"]) > MAX_EXAMPLES:
            raise ValueError("Scorecard row exceeds its bounded envelope")
        if len({metric["id"] for metric in row["metrics"]}) != len(row["metrics"]):
            raise ValueError("Scorecard metric identities must be unique")
    if len(json.dumps(result, sort_keys=True, allow_nan=False).encode()) > MAX_BYTES:
        raise ValueError("Scorecard exceeds its 64-KiB envelope")
    return result
