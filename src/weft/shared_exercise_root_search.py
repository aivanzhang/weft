"""Experimental nonlinear polls of existing root assistance on fixed action sequences."""

from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import threading
import time

import numpy as np

from weft.shared_exercise_action_prediction import _identity
from weft.shared_exercise_action_projection import ActionProjector, ProjectionConfig, _COUNTERS
from weft.shared_exercise_sequence_prediction import prepare_action_sequence_predictor


MAX_RESULT_BYTES = 2 << 20
SCOPE = ("Experimental measured nonlinear search over existing three-axis root assistance in one or two "
         "explicit fixed actions. Joint commands and physical force limits stay unchanged; equipment forces "
         "remain forbidden. Sparse coordinate polls are not an infeasibility proof. Repeated endpoint/floor "
         "measurements are not accumulated task-gate coverage, policy feedback, task success or a safe-control "
         "guarantee. Caller owns actual stepping, independent task checks and aggregate episode budgets, and "
         "must serialize environment commands/model changes. No public/default routing.")


@dataclass(frozen=True)
class RootSearchConfig:
    maximum_sequences: int = 26
    maximum_controls: int = 52
    maximum_seconds: float = 5.
    root_residual_bound: float = .25

    def __post_init__(self):
        for name, maximum in (("maximum_sequences", 26), ("maximum_controls", 52)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer from one to {maximum}")
        if (type(self.maximum_seconds) not in (int, float) or not math.isfinite(self.maximum_seconds)
                or not .001 <= self.maximum_seconds <= 5.):
            raise ValueError("Root search active budget must be .001–5 seconds")
        if type(self.root_residual_bound) not in (int, float) or self.root_residual_bound != .25:
            raise ValueError("Root search fixes the original-base residual and poll step at .25")


def _hash(value):
    return sha256(json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def _key(actions):
    return tuple((action.dtype.str, action.tobytes()) for action in actions)


def _root_trial(bases, center, control, axis, direction, bound):
    trial = tuple(action.copy() for action in center)
    base, value = bases[control], trial[control]
    index = 19 + axis
    lower, upper = max(-1., float(base[index]) - bound), min(1., float(base[index]) + bound)
    rounded = value.dtype.type(np.clip(float(value[index]) + direction * bound, lower, upper))
    # Casting at a trust boundary can round outside the original action's box.
    if float(rounded) < lower:
        rounded = np.nextafter(rounded, value.dtype.type(math.inf))
    if float(rounded) > upper:
        rounded = np.nextafter(rounded, value.dtype.type(-math.inf))
    value[index] = rounded
    return trial


class RootAssistanceSearch:
    def __init__(self, env, config):
        if type(config) is not RootSearchConfig:
            raise ValueError("Root search requires immutable RootSearchConfig")
        if (env.action_count != 22 or tuple(env.action_names[19:]) != ("root_x", "root_y", "root_z")
                or env.task.get("control", {}).get("root_assistance") != "explicit_3_axis"):
            raise ValueError("Root search requires the task's existing explicit three-axis assistance")
        # This frozen context validates and measures only; its predictor is never called.
        self._measurement = ActionProjector(env, ProjectionConfig(maximum_predictions=1,
            maximum_seconds=config.maximum_seconds))
        self._predictor = prepare_action_sequence_predictor(env, maximum_sequences=config.maximum_sequences,
            maximum_controls=config.maximum_controls, maximum_sequence_controls=2, maximum_seconds=config.maximum_seconds)
        self._env, self._engine, self._config = env, env._engine, config
        self._live, self._model, self._accumulator = env._engine.data, env._engine.model, env._accumulator
        self._dt, self._substeps = self._measurement._dt, self._measurement._substeps
        self._used, self._seconds, self._calls, self._reserved = False, 0., 0, 0
        self._counts, self._counts_complete = dict.fromkeys(_COUNTERS, 0), True
        self._control_counts = dict.fromkeys(("controls_reserved", "controls_attempted", "controls_completed"), 0)
        self._lock = threading.Lock()

    def describe(self):
        return {"scope": SCOPE, "maximum_decisions": 1, "decisions_used": int(self._used),
            "configuration": asdict(self._config), "sequence_calls": self._calls,
            "requested_forecast_controls_reserved": self._reserved, "forecast_control_counts": dict(self._control_counts),
            "maximum_native_substeps": self._config.maximum_controls * self._substeps,
            "maximum_result_bytes": MAX_RESULT_BYTES, "active_seconds": self._seconds,
            "native_counts": dict(self._counts), "native_counts_complete": self._counts_complete}

    def search(self, base_actions):
        if not self._lock.acquire(blocking=False):
            return None, {"status": "unavailable", "reason": "root_search_already_running"}
        try:
            return self._search(base_actions)
        finally:
            self._lock.release()

    def _actions(self, actions):
        if (not isinstance(actions, (list, tuple, np.ndarray))
                or isinstance(actions, np.ndarray) and (actions.ndim != 2 or actions.shape[1:] != (22,))
                or not 1 <= len(actions) <= 2):
            raise ValueError("root_search_requires_one_or_two_materialized_actions")
        result = []
        for action in actions:
            if (not isinstance(action, (np.ndarray, list, tuple))
                    or isinstance(action, np.ndarray) and action.shape != (22,)
                    or isinstance(action, (list, tuple)) and (len(action) != 22 or any(
                        not isinstance(value, (int, float, np.integer, np.floating))
                        or isinstance(value, (bool, np.bool_)) for value in action))):
                raise ValueError("root_search_requires_bounded_float32_or_float64_actions")
            value = np.asarray(action)
            if (value.dtype not in (np.dtype("float32"), np.dtype("float64"))
                    or not np.isfinite(value).all() or np.any(abs(value) > 1)):
                raise ValueError("root_search_requires_bounded_float32_or_float64_actions")
            result.append(value.copy())
        return tuple(result)

    def _search(self, base_actions):
        if self._used:
            return None, {"status": "unavailable", "reason": "decision_lifetime_budget", **self.describe()}
        self._used = True
        started, before, before_metrics, selected, pending = time.monotonic(), None, None, None, None
        report = {"status": "unavailable", "reason": None, "scope": SCOPE, "evaluations": [],
            "skipped_candidates": [], "rounds": [], "verified_sequence": None,
            "joint_commands_changed": False, "measured_boundary_proposal_only": True,
            "task_metrics_forecast": False, "live_state_unchanged": None, "task_metrics_unchanged": None}

        def check():
            if time.monotonic() - started >= self._config.maximum_seconds:
                raise TimeoutError("root_search_active_budget")
            if (self._env._engine is not self._engine or self._engine.model is not self._model
                    or self._engine.data is not self._live or self._env._accumulator is not self._accumulator
                    or self._env._done or self._engine._done or self._engine.dt != self._dt
                    or self._engine.substeps != self._substeps
                    or not np.array_equal(self._engine._q, self._measurement._q)):
                raise RuntimeError("environment_contract_changed")
            if before is not None and (_identity(self._live) != before or self._engine.steps != initial_step):
                raise RuntimeError("live_state_changed_during_root_search")

        def admit(calls):
            check()
            if (self._calls + calls > self._config.maximum_sequences
                    or self._reserved + calls * len(bases) > self._config.maximum_controls):
                raise RuntimeError("root_search_prediction_budget")

        def evaluate(actions, kind, **details):
            nonlocal pending
            pending = {"kind": kind, "actions": [action.tolist() for action in actions],
                       **details, "prediction_attempted": False}
            admit(1)
            if any(action.dtype != base.dtype or action[:19].tobytes() != base[:19].tobytes()
                    or np.any(abs(action[19:].astype(float) - base[19:]) > self._config.root_residual_bound)
                    or not np.isfinite(action).all() or np.any(abs(action) > 1)
                    for action, base in zip(actions, bases)):
                raise RuntimeError("joint_or_original_root_box_identity_changed")
            self._calls += 1
            self._reserved += len(bases)
            pending["prediction_attempted"] = True
            try:
                prediction = self._predictor.predict(tuple(action.copy() for action in actions))
            except Exception as error:
                self._counts_complete = False
                raise RuntimeError("sequence_predictor_raised:" + type(error).__name__) from error
            for group in (self._counts, self._control_counts):
                for key in group:
                    value = prediction.get(key)
                    if type(value) is not int or value < 0:
                        self._counts_complete = False
                    else:
                        group[key] += value
            report["failed_sequence"] = prediction
            check()
            if not self._counts_complete:
                raise RuntimeError("incomplete_sequence_native_counts")
            if (prediction.get("status") != "predicted" or prediction.get("sequence_complete") is not True
                    or any(prediction.get(key) != len(bases) for key in
                           ("controls_completed", "requested_controls", "controls_reserved", "controls_attempted"))
                    or len(prediction.get("controls", [])) != len(bases) or prediction.get("omitted_detail")):
                raise RuntimeError("sequence_prediction_unavailable:" + str(prediction.get("reason")))
            values = []
            for index, row in enumerate(prediction["controls"]):
                expected_time = initial_time + index * self._dt
                if (row.get("complete") is not True or row.get("control") != index + 1
                        or not math.isclose(row["initial_time_s"], expected_time, rel_tol=0, abs_tol=1e-10)
                        or row.get("human_action") != actions[index].astype(float).tolist()):
                    raise RuntimeError("incomplete_mistimed_or_wrong_action_boundary")
                values.extend(self._measurement._measure(row, expected_time, targets[index]))
            values = np.asarray(values)
            if values.shape != limits.shape or not np.isfinite(values).all() or np.any(values < 0):
                raise RuntimeError("unavailable_root_search_measurement")
            excess = np.where(values > limits + 1e-12, (values - limits) / limits, 0.)
            distance = sum(float(np.sum((action[19:].astype(float) - base[19:]) ** 2))
                           for action, base in zip(actions, bases))
            rank = (float(np.max(excess)), float(np.mean(excess)), distance)
            passes = rank[0] == 0.
            report["evaluations"].append({"kind": kind, **details, "actions": [action.tolist() for action in actions],
                "values": values.tolist(), "passes_measured_constraints": passes,
                "rank": {"maximum_normalized_excess": rank[0], "mean_normalized_excess": rank[1],
                         "correction_squared_norm": rank[2]},
                "endpoint_sha256": [_hash(row["endpoint"]) for row in prediction["controls"]],
                "substep_contacts_sha256": [_hash(row["substep_contacts"]) for row in prediction["controls"]]})
            report.pop("failed_sequence")
            pending = None
            check()
            return {"actions": actions, "prediction": prediction, "rank": rank, "passes": passes,
                    "evaluation": len(report["evaluations"]) - 1, "values": values}

        def poll(center, round_number, seen):
            candidates = []
            for control in range(len(bases)):
                for axis in range(3):
                    for direction in (-1, 1):
                        details = {"round": round_number, "control": control + 1,
                                   "axis": ("root_x", "root_y", "root_z")[axis], "direction": direction}
                        trial = _root_trial(bases, center, control, axis, direction, self._config.root_residual_bound)
                        key = _key(trial)
                        if key in seen:
                            report["skipped_candidates"].append({**details, "reason": "duplicate_or_clipped_unchanged"})
                        else:
                            seen.add(key)
                            candidates.append((trial, details))
            return candidates

        try:
            check()
            bases = self._actions(base_actions)
            initial_step, initial_time = self._engine.steps, float(self._live.time)
            if initial_step + len(bases) > min(self._env.horizon_steps, self._engine.horizon_steps):
                raise RuntimeError("explicit_sequence_exceeds_remaining_horizon")
            before, before_metrics = _identity(self._live), deepcopy(self._env.metrics())
            report.update(base_actions=[action.tolist() for action in bases], action_dtypes=[str(action.dtype) for action in bases],
                supplied_controls=len(bases), joint_bytes_sha256=[sha256(action[:19].tobytes()).hexdigest() for action in bases])
            targets = [np.asarray(self._engine._target_at(initial_time + step * self._dt)[0], float).copy()
                       for step in range(1, len(bases) + 1)]
            if any(target.shape != (22,) or not np.isfinite(target).all() or np.max(abs(target)) > 1e6 for target in targets):
                raise RuntimeError("unsupported_sequence_endpoint_reference")
            limits = np.tile(self._measurement._limits, len(bases))
            report["constraints"] = [{**row, "control": index + 1, "time_s": initial_time + (index + 1) * self._dt}
                for index in range(len(bases)) for row in self._measurement._constraints]
            best = evaluate(bases, "base")
            seen = {_key(bases)}
            if not best["passes"]:
                for round_number in (1, 2):
                    center_rank, center_evaluation = best["rank"], best["evaluation"]
                    candidates = poll(best["actions"], round_number, seen)
                    round_row = {"round": round_number, "center_evaluation": center_evaluation,
                                 "candidate_count": len(candidates), "completed_candidates": 0, "status": "pending"}
                    report["rounds"].append(round_row)
                    # Reserve the entire unique poll plus verification, never an unreported shortened search.
                    admit(len(candidates) + 1)
                    for trial, details in candidates:
                        current = evaluate(trial, "root_poll", **details)
                        round_row["completed_candidates"] += 1
                        if current["rank"] < best["rank"]:
                            best = current
                        del current
                    improved = best["rank"][:2] < center_rank[:2]
                    round_row.update(status="completed", best_evaluation=best["evaluation"],
                                     strictly_improved_measured_violation=improved)
                    if not improved:
                        break
            report["best_evaluation"] = best["evaluation"]
            if best["passes"]:
                repeated = evaluate(best["actions"], "selected_repeat")
                if (not repeated["passes"] or not np.array_equal(repeated["values"], best["values"])
                        or repeated["prediction"]["controls"] != best["prediction"]["controls"]):
                    raise RuntimeError("selected_sequence_not_repeatable")
                selected = tuple(action.copy() for action in best["actions"])
                report.update(status="unchanged" if best["rank"][2] == 0 else "selected",
                    reason="measured_boundary_sequence_repeated", correction_squared_norm=best["rank"][2],
                    verified_sequence={"initial_time_s": initial_time, "controls": repeated["prediction"]["controls"]})
            else:
                report.update(status="no_verified_candidate", reason="nonlinear_root_candidates_fail_measured_constraints")
            check()
        except Exception as error:
            selected = None
            report.update(status="unavailable", reason=str(error)[:300], verified_sequence=None)
            if pending is not None:
                report["failed_evaluation"] = {**pending, "reason": report["reason"]}
            if report["rounds"] and report["rounds"][-1]["status"] == "pending":
                report["rounds"][-1].update(status="interrupted", reason=report["reason"])
        try:
            if before is not None:
                report["live_state_unchanged"] = _identity(self._live) == before and self._engine.steps == initial_step
                report["task_metrics_unchanged"] = self._env.metrics() == before_metrics
                if not report["live_state_unchanged"] or not report["task_metrics_unchanged"]:
                    raise RuntimeError("live_state_or_task_metrics_changed_during_root_search")
            check()
        except Exception as error:
            selected = None
            report.update(status="unavailable", reason=str(error)[:300], verified_sequence=None)
        report.update(self.describe())
        try:
            if len(json.dumps(report, allow_nan=False, separators=(",", ":")).encode()) > MAX_RESULT_BYTES - 8192:
                raise ValueError("root_search_result_byte_budget")
        except (ValueError, TypeError) as error:
            selected = None
            if report["verified_sequence"] is not None:
                report["omitted_verified_controls"] = len(report["verified_sequence"]["controls"])
            report.update(status="unavailable", reason=str(error)[:300], verified_sequence=None)
            failed = report.pop("failed_sequence", None)
            if failed is not None:
                report["omitted_failed_sequence"] = {"status": failed.get("status"), "reason": str(failed.get("reason"))[:300],
                    "controls": len(failed.get("controls", [])),
                    "substep_contacts": sum(len(row.get("substep_contacts", [])) for row in failed.get("controls", []))}
        try:
            check()
        except Exception as error:
            selected = None
            report.update(status="unavailable", reason=str(error)[:300], verified_sequence=None)
        self._seconds = time.monotonic() - started
        if self._seconds >= self._config.maximum_seconds:
            selected = None
            report.update(status="unavailable", reason="root_search_active_budget", verified_sequence=None)
        report.update(self.describe())
        return selected, report


def prepare_root_assistance_search(env, *, config=RootSearchConfig()):
    return RootAssistanceSearch(env, config)
