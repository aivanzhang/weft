"""Experimental one-decision joint search; endpoint checks are not task success."""

from dataclasses import dataclass
import json
import math
import threading
import time

import numpy as np

from weft.shared_exercise_action_prediction import _identity, prepare_action_predictor
from weft.shared_exercise_task import GATES, GATES_SHA256


MAX_PREDICTIONS = 24
MAX_RELINEARIZED_PREDICTIONS = 48
MAX_RESULT_BYTES = 2 << 20
SCOPE = ("Authored, privileged, experimental one-control search. Equipment and human tracking are "
         "endpoint-only; unseen substep peaks, future contacts and whole-task success are not certified. "
         "Caller must serialize live environment commands and model changes. No public/default routing.")
_COUNTERS = ("native_substeps", "native_substeps_attempted", "boundary_forwards", "boundary_forwards_attempted",
             "copies", "copies_attempted", "data_allocations", "data_allocations_attempted")


@dataclass(frozen=True)
class ProjectionConfig:
    maximum_predictions: int = MAX_PREDICTIONS
    maximum_seconds: float = 5.
    difference: float = .1
    trust_radius: float = .25
    maximum_sweeps: int = 64
    candidate_strategy: str = "single_round"

    def __post_init__(self):
        if type(self.candidate_strategy) is not str or self.candidate_strategy not in ("single_round", "two_round_relinearization"):
            raise ValueError("Unknown bounded projection strategy")
        maximum = MAX_PREDICTIONS if self.candidate_strategy == "single_round" else MAX_RELINEARIZED_PREDICTIONS
        if type(self.maximum_predictions) is not int or not 1 <= self.maximum_predictions <= maximum:
            message = ("Projection permits one to twenty-four predictions over its lifetime"
                       if self.candidate_strategy == "single_round" else "Relinearization permits at most forty-eight predictions")
            raise ValueError(message)
        if type(self.maximum_sweeps) is not int or not 1 <= self.maximum_sweeps <= 64:
            raise ValueError("Projection permits at most sixty-four fixed solver sweeps")
        for name, low, high in (("maximum_seconds", .001, 5.), ("difference", 1e-4, .25),
                                ("trust_radius", 1e-4, .25)):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError("Invalid bounded projection setting: " + name)
        if self.difference > self.trust_radius:
            raise ValueError("Finite differences must remain inside the action trust region")


def _linear_projection(matrix, bound, lower, upper, maximum_sweeps, check):
    """Dykstra projects zero onto a local halfspace/box intersection, with fixed work."""
    if not np.isfinite(matrix).all() or np.max(abs(matrix)) > 1e12:
        raise RuntimeError("linear_response_envelope")
    minimum = np.sum(matrix * np.where(matrix >= 0, lower, upper), axis=1)
    conflict = np.flatnonzero(minimum > bound + 1e-10)
    if len(conflict):
        return None, {"status": "row_box_conflict", "constraint_index": int(conflict[0]), "sweeps": 0}
    norms = np.linalg.norm(matrix, axis=1)
    active = (norms > 1e-14) & (np.sum(matrix * np.where(matrix >= 0, upper, lower), axis=1) > bound)
    normals, offsets = matrix[active] / norms[active, None], bound[active] / norms[active]
    delta, box_correction = np.zeros(19), np.zeros(19)
    corrections = np.zeros(len(normals))
    for sweep in range(maximum_sweeps):
        check()
        previous = delta.copy()
        for index, (normal, offset) in enumerate(zip(normals, offsets)):
            shifted = delta + corrections[index] * normal
            corrections[index] = max(0., float(normal @ shifted - offset))
            delta = shifted - corrections[index] * normal
        shifted = delta + box_correction
        delta = np.clip(shifted, lower, upper)
        box_correction = shifted - delta
        violation = max(0., float(np.max(matrix @ delta - bound)))
        if violation <= 1e-8 and np.max(abs(delta - previous)) <= 1e-9:
            return delta, {"status": "converged", "sweeps": sweep + 1, "maximum_linear_violation": violation}
    # An unconverged bounded iterate can propose a test, never certify an action.
    return delta, {"status": "iteration_limit", "proposal_only": True,
                   "sweeps": maximum_sweeps, "maximum_linear_violation": violation}


def _bounded_trial(base, delta, radius):
    lower, upper = np.maximum(-1., base[:19].astype(float) - radius), np.minimum(1., base[:19].astype(float) + radius)
    trial = base.copy()
    trial[:19] = np.clip(base[:19].astype(float) + delta, lower, upper)
    # Round toward the interior when float32 conversion crosses a physical/trust bound.
    below, above = trial[:19] < lower, trial[:19] > upper
    trial[:19][below] = np.nextafter(trial[:19][below], np.array(np.inf, dtype=base.dtype))
    trial[:19][above] = np.nextafter(trial[:19][above], np.array(-np.inf, dtype=base.dtype))
    if (np.any(abs(trial) > 1) or np.any(abs(trial[:19].astype(float) - base[:19]) > radius)
            or trial[19:].tobytes() != base[19:].tobytes()):
        raise RuntimeError("rounded_action_outside_original_bounds")
    return trial


def _center_difference(base, center, joint, difference, radius):
    lower, upper = max(-1., float(base[joint]) - radius), min(1., float(base[joint]) + radius)
    value = float(center[joint])
    negative, positive = value - lower, upper - value
    sign = 1 if positive > negative or positive == negative and value <= 0 else -1
    room = positive if sign > 0 else negative
    trial = center.copy()
    trial[joint] = value + sign * min(difference, room)
    if float(trial[joint]) < lower or float(trial[joint]) > upper:
        direction = np.inf if float(trial[joint]) < lower else -np.inf
        trial[joint] = np.nextafter(trial[joint], np.array(direction, dtype=base.dtype))
    actual = float(trial[joint]) - value
    if (not math.isfinite(actual) or actual == 0 or actual * sign <= 0
            or np.any(abs(trial) > 1) or np.any(abs(trial[:19].astype(float) - base[:19]) > radius)
            or trial[19:].tobytes() != base[19:].tobytes()):
        raise RuntimeError("unavailable_relinearization_difference")
    return trial, actual


class ActionProjector:
    def __init__(self, env, config):
        if not isinstance(config, ProjectionConfig):
            raise ValueError("Projection requires immutable ProjectionConfig")
        task, engine = env.task, env._engine
        if (task.get("profile") != "shared-exercise-task-v1" or task.get("gates_sha256") != GATES_SHA256
                or task.get("contact_requirements") != []
                or task.get("control", {}).get("object_forces") != "forbidden"):
            raise ValueError("Projection requires frozen equipment gates with no required-contact windows")
        accumulator = env._accumulator
        if accumulator is None or env._done or not engine._initialized or engine._done:
            raise ValueError("Prepare projection after reset and before completion")
        identifiers = tuple(task["equipment_ids"])
        if not 1 <= len(identifiers) <= 12 or set(identifiers) != set(accumulator.initial):
            raise ValueError("Projection requires complete reset-relative equipment states")
        self._env, self._engine, self._config, self._accumulator = env, engine, config, accumulator
        self._model, self._live = engine.model, engine.data
        self._scale, self._substeps, self._dt = float(engine.scale), engine.substeps, float(engine.dt)
        self._q = np.asarray(engine._q, dtype=int).copy()
        if not .5 <= self._scale <= 2. or self._q.shape != (22,) or len(set(self._q)) != 22:
            raise ValueError("Projection requires the original twenty-two human state coordinates")
        if np.any(self._q < 0) or np.any(self._q >= self._model.nq):
            raise ValueError("Human coordinate map exceeds the model")
        self._q.flags.writeable = False
        self._objects, self._constraints = [], []
        for name in identifiers:
            qadr, vadr = engine._object_free_addresses[name]
            initial = accumulator.initial[name]
            position, quaternion = np.asarray(initial.position, float).copy(), np.asarray(initial.quaternion, float).copy()
            if (not 0 <= qadr <= self._model.nq - 7 or not 0 <= vadr <= self._model.nv - 6
                    or position.shape != (3,) or quaternion.shape != (4,)
                    or not np.isfinite(position).all() or np.max(abs(position)) > 1e6
                    or not np.isfinite(quaternion).all() or abs(np.linalg.norm(quaternion) - 1) > 1e-6):
                raise ValueError("Invalid reset equipment pose or free-joint map")
            position.flags.writeable = quaternion.flags.writeable = False
            self._objects.append((name, int(qadr), int(vadr), position, quaternion))
            for metric in ("displacement_model_units", "orientation_rad", "speed_model_units_s", "angular_speed_rad_s"):
                self._constraints.append({"name": name + ":" + metric, "limit": GATES["equipment_" + metric],
                    "sampling": "integrated_control_endpoint", "true_substep_equivalent": False})
        for name, gate in (("human_angular_rmse_rad", "peak_angular_rmse_rad"),
                           ("human_root_rmse_model_units", "peak_root_rmse_model_units")):
            self._constraints.append({"name": name, "limit": GATES[gate],
                "sampling": "integrated_control_endpoint", "true_substep_equivalent": False})
        for name, sampling, equivalent in (("floor_sample_max_model_units", "solver_start_cached_contact_samples", True),
                                           ("floor_boundary_model_units", "boundary_forward_contacts", False)):
            self._constraints.append({"name": name, "limit": GATES["max_floor_penetration_model_units"],
                                      "sampling": sampling, "true_substep_equivalent": equivalent})
        self._limits = np.array([row["limit"] for row in self._constraints])
        self._limits.flags.writeable = False
        self._predictor = prepare_action_predictor(env, maximum_predictions=config.maximum_predictions,
                                                  maximum_seconds=config.maximum_seconds)
        self._used, self._prediction_calls, self._seconds = False, 0, 0.
        self._counts = dict.fromkeys(_COUNTERS, 0)
        self._counts_complete = True
        self._lock = threading.Lock()

    def describe(self):
        return {"scope": SCOPE, "maximum_decisions": 1, "decisions_used": int(self._used),
                "maximum_predictions": self._config.maximum_predictions, "prediction_calls": self._prediction_calls,
                "maximum_native_substeps": self._config.maximum_predictions * self._substeps,
                "maximum_active_seconds": self._config.maximum_seconds, "active_seconds": self._seconds,
                "maximum_result_bytes": MAX_RESULT_BYTES, "native_counts": dict(self._counts),
                "native_counts_complete": self._counts_complete}

    def _measure(self, prediction, initial_time, target):
        endpoint = prediction["endpoint"]
        q, v = np.asarray(endpoint["qpos"], float), np.asarray(endpoint["qvel"], float)
        if (q.shape != (self._model.nq,) or v.shape != (self._model.nv,)
                or not np.isfinite(q).all() or not np.isfinite(v).all()
                or np.max(abs(q)) > 1e6 or np.max(abs(v)) > 1e6
                or not math.isclose(endpoint["time_s"], initial_time + self._dt, rel_tol=0, abs_tol=1e-10)):
            raise RuntimeError("unsupported_predicted_endpoint")
        values = []
        for _, qadr, vadr, position, quaternion in self._objects:
            orientation = q[qadr + 3:qadr + 7]
            if abs(np.linalg.norm(orientation) - 1) > 1e-6:
                raise RuntimeError("nonunit_equipment_quaternion")
            # Angular speed norm is rotation-invariant; no extra native velocity query is needed.
            values.extend((math.dist(q[qadr:qadr + 3], position) / self._scale,
                           2 * math.acos(float(np.clip(abs(orientation @ quaternion), 0, 1))),
                           math.hypot(*v[vadr:vadr + 3]) / self._scale, math.hypot(*v[vadr + 3:vadr + 6])))
        error = q[self._q] - target
        values.extend((math.hypot(*error[:19]) / math.sqrt(19),
                       math.hypot(*error[19:]) / (math.sqrt(3) * self._scale)))
        samples = prediction["substep_contacts"]
        if len(samples) != self._substeps:
            raise RuntimeError("incomplete_floor_contact_samples")
        floor = []
        for index, sample in enumerate(samples):
            if (sample["substep"] != index + 1
                    or not math.isclose(sample["solver_time_s"], initial_time + index * self._dt / self._substeps,
                                        rel_tol=0, abs_tol=1e-10)
                    or not math.isclose(sample["integrated_time_s"], initial_time + (index + 1) * self._dt / self._substeps,
                                        rel_tol=0, abs_tol=1e-10)):
                raise RuntimeError("unsupported_contact_sample_clock")
            floor.append(float(sample["contacts"]["max_human_floor_penetration_model_units"]))
        floor.append(float(endpoint["contacts"]["max_human_floor_penetration_model_units"]))
        if any(not math.isfinite(value) or not 0 <= value <= 1e6 for value in floor):
            raise RuntimeError("unsupported_floor_contact_metric")
        values.extend((max(floor[:-1]), floor[-1]))
        values = np.asarray(values)
        if not np.isfinite(values).all():
            raise RuntimeError("nonfinite_measured_constraints")
        return values

    def project(self, base_action):
        if not self._lock.acquire(blocking=False):
            return None, {"status": "unavailable", "reason": "projection_already_running"}
        try:
            return self._project(base_action)
        finally:
            self._lock.release()

    def _second_round(self, base, center, center_values, evaluate, solver_check, remaining_sweeps, report, summary):
        matrix = np.zeros((len(self._limits), 19))
        best, best_prediction, best_distance = None, None, math.inf
        for joint in range(19):
            trial, difference = _center_difference(base, center, joint, self._config.difference, self._config.trust_radius)
            prediction, values, passes = evaluate(trial, "finite_difference", joint=joint, difference=difference)
            matrix[:, joint] = (values - center_values) / (self._limits * difference)
            distance = float(np.sum((trial[:19].astype(float) - base[:19]) ** 2))
            if passes and distance < best_distance:
                best, best_prediction, best_distance = trial, prediction, distance
        original_lower = np.maximum(-1., base[:19].astype(float) - self._config.trust_radius)
        original_upper = np.minimum(1., base[:19].astype(float) + self._config.trust_radius)
        lower, upper = original_lower - center[:19], original_upper - center[:19]
        bound = 1 - 1e-6 - center_values / self._limits
        delta, linear = _linear_projection(matrix, bound, lower, upper, remaining_sweeps, solver_check)
        summary["linear_model"] = linear
        if delta is not None:
            for scale in (1., .5, .25):
                correction = center[:19].astype(float) - base[:19] + scale * delta
                trial = _bounded_trial(base, correction, self._config.trust_radius)
                prediction, values, passes = evaluate(trial, "candidate", correction_scale=scale)
                actual_delta = trial[:19].astype(float) - center[:19]
                report["evaluations"][-1]["linear_fit_max_ratio_error"] = float(np.max(abs(
                    values / self._limits - center_values / self._limits - matrix @ actual_delta)))
                distance = float(np.sum((trial[:19].astype(float) - base[:19]) ** 2))
                if passes and distance < best_distance:
                    best, best_prediction, best_distance = trial, prediction, distance
        return best, best_prediction, best_distance, delta, linear

    def _project(self, base_action):
        if self._used:
            return None, {"status": "unavailable", "reason": "decision_lifetime_budget", **self.describe()}
        self._used = True
        started, before, selected = time.monotonic(), None, None
        report = {"status": "unavailable", "reason": None, "scope": SCOPE,
                  "constraints": self._constraints, "evaluations": [], "verified_endpoint": None,
                  "joint_variables": 19, "root_commands_changed": False,
                  "linearized_feasibility_is_not_physical_feasibility": True}
        refinement, measured_center, current_round, pending_evaluation = None, None, 1, None
        if self._config.candidate_strategy == "two_round_relinearization":
            refinement = {"strategy": self._config.candidate_strategy, "maximum_rounds": 2,
                          "maximum_solver_sweeps": self._config.maximum_sweeps, "solver_sweeps_used": 0,
                          "stop_reason": "not_started", "rounds": []}
            report["refinement"] = refinement

        def check():
            if time.monotonic() - started >= self._config.maximum_seconds:
                raise TimeoutError("projection_active_compute_budget")
            if (self._engine.data is not self._live or self._engine.model is not self._model
                    or self._env._accumulator is not self._accumulator or self._env._done
                    or self._engine._done or not np.array_equal(self._engine._q, self._q)):
                raise RuntimeError("environment_contract_changed")
            if before is not None and (_identity(self._live) != before or self._engine.steps != initial_step):
                raise RuntimeError("live_state_changed_during_projection")

        def evaluate(action, kind, **details):
            nonlocal measured_center, pending_evaluation
            if refinement is not None:
                pending_evaluation = {"round": current_round, "kind": kind, **details,
                                      "action": action.tolist(), "prediction_attempted": False}
            check()
            if self._prediction_calls >= self._config.maximum_predictions:
                raise RuntimeError("projection_prediction_budget")
            self._prediction_calls += 1
            if pending_evaluation is not None:
                pending_evaluation["prediction_attempted"] = True
            try:
                prediction = self._predictor.predict(action)
            except Exception as error:
                self._counts_complete = False
                raise RuntimeError("predictor_raised:" + type(error).__name__) from error
            for key in _COUNTERS:
                if key not in prediction:
                    self._counts_complete = False
                self._counts[key] += int(prediction.get(key, 0))
            check()
            if prediction.get("status") != "predicted":
                raise RuntimeError("prediction_unavailable:" + str(prediction.get("reason")))
            values = self._measure(prediction, initial_time, target)
            passes = bool(np.all(values <= self._limits + 1e-12))
            if refinement is not None:
                details["round"] = current_round
            report["evaluations"].append({"kind": kind, **details, "action": action.tolist(),
                                          "values": values.tolist(), "passes_measured_constraints": passes})
            if refinement is not None and not passes:
                violation = max(0., float(np.max(values / self._limits)) - 1.)
                delta = action[:19].astype(float) - base[:19]
                index = len(report["evaluations"]) - 1
                key = (violation, float(delta @ delta), index)
                if measured_center is None or key < measured_center[0]:
                    measured_center = (key, action.copy(), values.copy())
            pending_evaluation = None
            return prediction, values, passes

        def solver_check():
            check()
            if refinement["solver_sweeps_used"] >= self._config.maximum_sweeps:
                raise RuntimeError("aggregate_solver_sweep_budget")
            refinement["solver_sweeps_used"] += 1

        try:
            check()
            base = np.asarray(base_action)
            if (base.shape != (self._env.action_count,) or base.dtype not in (np.dtype("float32"), np.dtype("float64"))
                    or not np.isfinite(base).all() or np.any(abs(base) > 1)
                    or isinstance(base_action, (list, tuple)) and any(isinstance(value, (bool, np.bool_)) for value in base_action)):
                raise ValueError("projection_requires_bounded_float32_or_float64_action")
            base = base.copy()
            before, initial_step, initial_time = _identity(self._live), self._engine.steps, float(self._live.time)
            target = np.asarray(self._engine._target_at(initial_time + self._dt)[0], float).copy()
            if target.shape != (22,) or not np.isfinite(target).all() or np.max(abs(target)) > 1e6:
                raise RuntimeError("unsupported_endpoint_reference")
            baseline, baseline_values, passes = evaluate(base, "base")
            if refinement is not None:
                baseline_violation = max(0., float(np.max(baseline_values / self._limits)) - 1.)
                refinement["rounds"].append({"round": 1, "center_evaluation_index": 0,
                    "center_action": base.tolist(), "center_normalized_violation": baseline_violation,
                    "prediction_start": 0, "solver_sweep_start": 0, "search_completed": False})
            if passes:
                selected = base
                report.update(status="unchanged", reason="base_passes_measured_constraints",
                              verified_endpoint=baseline["endpoint"])
                if refinement is not None:
                    refinement["stop_reason"] = "base_passes_measured_constraints"
                    refinement["rounds"][0]["search_completed"] = True
            else:
                # Reserve the complete finite-difference/search/repeat work before spending it.
                if self._config.maximum_predictions - self._prediction_calls < 23:
                    raise RuntimeError("insufficient_budget_for_complete_search")
                matrix = np.zeros((len(self._limits), 19))
                best, best_prediction, best_distance = None, None, math.inf
                for joint in range(19):
                    change = np.zeros(19)
                    change[joint] = self._config.difference * (1 if base[joint] <= 0 else -1)
                    trial = _bounded_trial(base, change, self._config.trust_radius)
                    difference = float(trial[joint]) - float(base[joint])
                    prediction, values, passes = evaluate(trial, "finite_difference", joint=joint, difference=difference)
                    matrix[:, joint] = (values - baseline_values) / (self._limits * difference)
                    if passes and difference ** 2 < best_distance:
                        best, best_prediction, best_distance = trial, prediction, difference ** 2
                lower = np.maximum(-1 - base[:19].astype(float), -self._config.trust_radius)
                upper = np.minimum(1 - base[:19].astype(float), self._config.trust_radius)
                # A tiny stricter target absorbs solver/float rounding; no task limit is relaxed.
                bound = 1 - 1e-6 - baseline_values / self._limits
                delta, linear = _linear_projection(matrix, bound, lower, upper, self._config.maximum_sweeps,
                                                   solver_check if refinement is not None else check)
                report["linear_model"] = linear
                if delta is not None:
                    for scale in (1., .5, .25):
                        trial = _bounded_trial(base, scale * delta, self._config.trust_radius)
                        prediction, values, passes = evaluate(trial, "candidate", correction_scale=scale)
                        actual_delta = trial[:19].astype(float) - base[:19]
                        report["evaluations"][-1]["linear_fit_max_ratio_error"] = float(np.max(abs(
                            values / self._limits - baseline_values / self._limits - matrix @ actual_delta)))
                        distance = float(actual_delta @ actual_delta)
                        if passes and distance < best_distance:
                            best, best_prediction, best_distance = trial, prediction, distance
                if refinement is not None:
                    first_round = refinement["rounds"][0]
                    first_round.update(linear_model=linear, search_completed=True)
                    if best is not None:
                        refinement["stop_reason"] = "passing_candidate_available"
                    elif measured_center is None or measured_center[0][0] >= baseline_violation:
                        refinement["stop_reason"] = "no_strict_measured_improvement"
                    elif self._config.maximum_predictions - self._prediction_calls < 23:
                        refinement["stop_reason"] = "insufficient_budget_for_complete_second_round"
                    elif refinement["solver_sweeps_used"] >= self._config.maximum_sweeps:
                        refinement["stop_reason"] = "aggregate_solver_sweep_budget"
                    else:
                        key, center, center_values = measured_center
                        current_round = 2
                        second_round = {"round": 2, "center_evaluation_index": key[2],
                            "center_action": center.tolist(), "center_normalized_violation": key[0],
                            "strict_measured_improvement": True, "prediction_start": self._prediction_calls,
                            "solver_sweep_start": refinement["solver_sweeps_used"], "search_completed": False}
                        refinement["rounds"].append(second_round)
                        best, best_prediction, best_distance, delta, linear = self._second_round(
                            base, center, center_values, evaluate, solver_check,
                            self._config.maximum_sweeps - refinement["solver_sweeps_used"], report, second_round)
                        second_round["search_completed"] = True
                        refinement["stop_reason"] = "maximum_rounds_completed"
                if best is None:
                    report.update(status="no_verified_candidate", reason=("linearized_" + linear["status"] if delta is None
                                  else "nonlinear_candidates_fail_measured_constraints"))
                else:
                    verified, _, passes = evaluate(best, "selected_repeat")
                    if (not passes or verified["endpoint"] != best_prediction["endpoint"]
                            or verified["substep_contacts"] != best_prediction["substep_contacts"]):
                        raise RuntimeError("selected_prediction_not_repeatable")
                    if best[19:].tobytes() != base[19:].tobytes():
                        raise RuntimeError("root_command_identity_mismatch")
                    selected = best
                    report.update(status="selected", reason="nonlinear_candidate_repeated",
                                  correction_squared_norm=best_distance, verified_endpoint=verified["endpoint"])
            check()
        except (RuntimeError, ValueError, KeyError, TypeError, OverflowError, TimeoutError) as error:
            selected = None
            report.update(status="unavailable", reason=str(error)[:300], verified_endpoint=None)
            if refinement is not None:
                refinement.update(stop_reason=report["reason"], stopped_round=current_round)
                if pending_evaluation is not None:
                    report["failed_evaluation"] = {**pending_evaluation, "status": "unavailable", "reason": report["reason"]}
        finally:
            if refinement is not None:
                rounds = refinement["rounds"]
                for index, item in enumerate(rounds):
                    following = rounds[index + 1] if index + 1 < len(rounds) else None
                    item["prediction_calls"] = (following["prediction_start"] if following else self._prediction_calls) - item["prediction_start"]
                    item["solver_sweeps"] = (following["solver_sweep_start"] if following else refinement["solver_sweeps_used"]) - item["solver_sweep_start"]
            if before is not None and (_identity(self._live) != before or self._engine.steps != initial_step):
                selected = None
                report.update(status="unavailable", reason="live_state_changed_during_projection", verified_endpoint=None)
                if refinement is not None:
                    refinement.update(stop_reason=report["reason"], stopped_round=current_round)
        try:
            if len(json.dumps(report, allow_nan=False, separators=(",", ":")).encode()) > MAX_RESULT_BYTES - 1024:
                raise ValueError("projection_result_byte_budget")
        except (ValueError, TypeError) as error:
            selected = None
            report.update(status="unavailable", reason=str(error)[:300], verified_endpoint=None)
            if refinement is None:
                report["evaluations"] = []
            else:
                refinement.update(stop_reason=report["reason"], stopped_round=current_round)
                # The unaccepted endpoint is expendable; retain measured failures within the same cap.
                try:
                    fits = len(json.dumps(report, allow_nan=False, separators=(",", ":")).encode()) <= MAX_RESULT_BYTES - 1024
                except (ValueError, TypeError):
                    fits = False
                if not fits:
                    omitted = {"reason": "diagnostics_exceed_result_budget", "evaluations": len(report["evaluations"]),
                               "rounds": len(refinement["rounds"])}
                    report = {"status": "unavailable", "reason": report["reason"], "scope": SCOPE,
                              "verified_endpoint": None, "evaluations": [], "diagnostic_omission": omitted,
                              "refinement": {"strategy": self._config.candidate_strategy,
                                  "stop_reason": report["reason"], "stopped_round": current_round,
                                  "solver_sweeps_used": refinement["solver_sweeps_used"]}}
                    refinement = report["refinement"]
        self._seconds = time.monotonic() - started
        report.update(self.describe())
        if self._seconds >= self._config.maximum_seconds:
            selected = None
            report.update(status="unavailable", reason="projection_active_compute_budget", verified_endpoint=None)
            if refinement is not None:
                refinement.update(stop_reason=report["reason"], stopped_round=current_round)
        return selected, report


def prepare_action_projector(env, *, config=ProjectionConfig()):
    return ActionProjector(env, config)
