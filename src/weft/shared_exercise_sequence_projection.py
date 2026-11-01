"""Experimental first-action correction for an explicitly fixed two-action continuation."""

from dataclasses import dataclass, asdict
from hashlib import sha256
import json
import math
import threading
import time

import numpy as np

from weft.shared_exercise_action_prediction import _identity
from weft.shared_exercise_action_projection import ActionProjector, ProjectionConfig, _COUNTERS, _bounded_trial, _linear_projection
from weft.shared_exercise_sequence_prediction import prepare_action_sequence_predictor


MAX_RESULT_BYTES = 2 << 20
SCOPE = ("Experimental open-loop two-action proposal. Only the first action's joint commands may change; "
         "the supplied second action is fixed, not recomputed feedback. Repeated endpoint/floor measurements "
         "are not accumulated task-gate coverage, scorer success or a safe-control guarantee. No public/default "
         "routing. Caller must serialize live environment commands and model changes.")


@dataclass(frozen=True)
class SequenceProjectionConfig:
    maximum_sequences: int = 24
    maximum_controls: int = 48
    maximum_sweeps: int = 64
    maximum_seconds: float = 5.
    difference: float = .1
    trust_radius: float = .25

    def __post_init__(self):
        for name, maximum in (("maximum_sequences", 24), ("maximum_controls", 48), ("maximum_sweeps", 64)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer from one to {maximum}")
        if (type(self.maximum_seconds) not in (int, float) or not math.isfinite(self.maximum_seconds)
                or not .001 <= self.maximum_seconds <= 5.):
            raise ValueError("Sequence projection active budget must be .001–5 seconds")
        if (type(self.difference) not in (int, float) or self.difference != .1
                or type(self.trust_radius) not in (int, float) or self.trust_radius != .25):
            raise ValueError("This experiment fixes difference=.1 and original-base trust_radius=.25")


def _hash(value):
    return sha256(json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


class ActionSequenceProjector:
    def __init__(self, env, config):
        if not isinstance(config, SequenceProjectionConfig):
            raise ValueError("Sequence projection requires immutable SequenceProjectionConfig")
        # This context validates and measures only; its one-control predictor is never called.
        self._measurement = ActionProjector(env, ProjectionConfig(maximum_predictions=1,
            maximum_seconds=config.maximum_seconds))
        self._predictor = prepare_action_sequence_predictor(env, maximum_sequences=config.maximum_sequences,
            maximum_controls=config.maximum_controls, maximum_sequence_controls=2, maximum_seconds=config.maximum_seconds)
        self._env, self._engine, self._config = env, env._engine, config
        self._live, self._model, self._accumulator = env._engine.data, env._engine.model, env._accumulator
        self._dt, self._substeps = self._measurement._dt, self._measurement._substeps
        self._limits = np.tile(self._measurement._limits, 2)
        self._used, self._seconds, self._calls, self._sweeps = False, 0., 0, 0
        self._counts, self._counts_complete = dict.fromkeys(_COUNTERS, 0), True
        self._control_counts = dict.fromkeys(("controls_reserved", "controls_attempted", "controls_completed"), 0)
        self._lock = threading.Lock()

    def describe(self):
        return {"scope": SCOPE, "maximum_decisions": 1, "decisions_used": int(self._used),
            "configuration": asdict(self._config), "sequence_calls": self._calls,
            "forecast_control_counts": dict(self._control_counts), "solver_sweeps": self._sweeps,
            "maximum_native_substeps": self._config.maximum_controls * self._substeps,
            "maximum_result_bytes": MAX_RESULT_BYTES, "active_seconds": self._seconds,
            "native_counts": dict(self._counts), "native_counts_complete": self._counts_complete}

    def project(self, first_base, second_fixed):
        if not self._lock.acquire(blocking=False):
            return None, {"status": "unavailable", "reason": "sequence_projection_already_running"}
        try:
            return self._project(first_base, second_fixed)
        finally:
            self._lock.release()

    def _action(self, action):
        if (not isinstance(action, (np.ndarray, list, tuple))
                or isinstance(action, np.ndarray) and action.shape != (self._env.action_count,)
                or isinstance(action, (list, tuple)) and (len(action) != self._env.action_count or any(
                    not isinstance(value, (int, float, np.integer, np.floating))
                    or isinstance(value, (bool, np.bool_)) for value in action))):
            raise ValueError("sequence_projection_requires_bounded_float32_or_float64_actions")
        value = np.asarray(action)
        if (value.shape != (self._env.action_count,) or value.dtype not in (np.dtype("float32"), np.dtype("float64"))
                or not np.isfinite(value).all() or np.any(value < -1) or np.any(value > 1)):
            raise ValueError("sequence_projection_requires_bounded_float32_or_float64_actions")
        return value.copy()

    def _project(self, first_base, second_fixed):
        if self._used:
            return None, {"status": "unavailable", "reason": "decision_lifetime_budget", **self.describe()}
        self._used = True
        started, before, pair, pending = time.monotonic(), None, None, None
        report = {"status": "unavailable", "reason": None, "scope": SCOPE, "evaluations": [],
            "verified_sequence": None, "first_root_commands_changed": False, "second_action_changed": False,
            "measured_boundary_proposal_only": True, "task_metrics_forecast": False,
            "linearized_feasibility_is_not_physical_feasibility": True}

        def check():
            if time.monotonic() - started >= self._config.maximum_seconds:
                raise TimeoutError("sequence_projection_active_budget")
            if (self._env._engine is not self._engine or self._engine.model is not self._model
                    or self._engine.data is not self._live or self._env._accumulator is not self._accumulator
                    or self._env._done or self._engine._done or self._engine.dt != self._dt
                    or self._engine.substeps != self._substeps
                    or not np.array_equal(self._engine._q, self._measurement._q)):
                raise RuntimeError("environment_contract_changed")
            if before is not None and (_identity(self._live) != before or self._engine.steps != initial_step):
                raise RuntimeError("live_state_changed_during_sequence_projection")

        def evaluate(first, kind, **details):
            nonlocal pending
            pending = {"kind": kind, "first_action": first.tolist(), **details, "prediction_attempted": False}
            check()
            if self._calls >= self._config.maximum_sequences or self._control_counts["controls_reserved"] + 2 > self._config.maximum_controls:
                raise RuntimeError("sequence_projection_prediction_budget")
            if (first.dtype != base.dtype or first[19:].tobytes() != base[19:].tobytes()
                    or np.any(abs(first[:19].astype(float) - base[:19]) > self._config.trust_radius)
                    or np.any(abs(first) > 1) or fixed.tobytes() != second_bytes):
                raise RuntimeError("fixed_action_or_original_trust_identity_changed")
            self._calls += 1
            pending["prediction_attempted"] = True
            try:
                prediction = self._predictor.predict((first.copy(), fixed.copy()))
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
                    or prediction.get("controls_completed") != 2 or prediction.get("requested_controls") != 2
                    or prediction.get("controls_reserved") != 2 or prediction.get("controls_attempted") != 2
                    or len(prediction.get("controls", [])) != 2 or prediction.get("omitted_detail")):
                raise RuntimeError("sequence_prediction_unavailable:" + str(prediction.get("reason")))
            values = []
            for index, row in enumerate(prediction["controls"]):
                expected_time = initial_time + index * self._dt
                if (row.get("complete") is not True or row.get("control") != index + 1
                        or not math.isclose(row["initial_time_s"], expected_time, rel_tol=0, abs_tol=1e-10)):
                    raise RuntimeError("incomplete_or_mistimed_sequence_boundary")
                values.extend(self._measurement._measure(row, expected_time, targets[index]))
            values = np.asarray(values)
            passes = bool(np.all(values <= self._limits + 1e-12))
            report["evaluations"].append({"kind": kind, **details, "first_action": first.tolist(),
                "values": values.tolist(), "passes_measured_constraints": passes,
                "endpoint_sha256": [_hash(row["endpoint"]) for row in prediction["controls"]],
                "substep_contacts_sha256": [_hash(row["substep_contacts"]) for row in prediction["controls"]]})
            report.pop("failed_sequence")
            pending = None
            check()
            return prediction, values, passes

        def solver_check():
            check()
            if self._sweeps >= self._config.maximum_sweeps:
                raise RuntimeError("sequence_projection_solver_budget")
            self._sweeps += 1

        try:
            check()
            base, fixed = self._action(first_base), self._action(second_fixed)
            second_bytes = fixed.tobytes()
            before, initial_step, initial_time = _identity(self._live), self._engine.steps, float(self._live.time)
            report.update(first_base_action=base.tolist(), first_dtype=str(base.dtype),
                fixed_second_action=fixed.tolist(), second_dtype=str(fixed.dtype), second_action_sha256=sha256(second_bytes).hexdigest())
            targets = [np.asarray(self._engine._target_at(initial_time + step * self._dt)[0], float).copy() for step in (1, 2)]
            if any(target.shape != (22,) or not np.isfinite(target).all() or np.max(abs(target)) > 1e6 for target in targets):
                raise RuntimeError("unsupported_sequence_endpoint_reference")
            report["constraints"] = [{**row, "control": index + 1, "time_s": initial_time + (index + 1) * self._dt}
                for index in range(2) for row in self._measurement._constraints]
            baseline, baseline_values, passes = evaluate(base, "base")
            best, best_prediction, best_distance = (base, baseline, 0.) if passes else (None, None, math.inf)
            if not passes:
                if self._config.maximum_sequences - self._calls < 23 or self._config.maximum_controls - self._control_counts["controls_reserved"] < 46:
                    raise RuntimeError("insufficient_budget_for_complete_sequence_search")
                matrix = np.zeros((len(self._limits), 19))
                for joint in range(19):
                    change = np.zeros(19); change[joint] = self._config.difference * (1 if base[joint] <= 0 else -1)
                    trial = _bounded_trial(base, change, self._config.trust_radius)
                    difference = float(trial[joint]) - float(base[joint])
                    if not math.isfinite(difference) or difference == 0:
                        raise RuntimeError("unavailable_sequence_finite_difference")
                    prediction, values, passes = evaluate(trial, "finite_difference", joint=joint, difference=difference)
                    matrix[:, joint] = (values - baseline_values) / (self._limits * difference)
                    if passes and difference ** 2 < best_distance:
                        best, best_prediction, best_distance = trial, prediction, difference ** 2
                lower = np.maximum(-1 - base[:19].astype(float), -self._config.trust_radius)
                upper = np.minimum(1 - base[:19].astype(float), self._config.trust_radius)
                delta, linear = _linear_projection(matrix, 1 - 1e-6 - baseline_values / self._limits,
                    lower, upper, self._config.maximum_sweeps, solver_check)
                report["linear_model"] = linear
                if delta is not None:
                    for scale in (1., .5, .25):
                        trial = _bounded_trial(base, scale * delta, self._config.trust_radius)
                        prediction, values, passes = evaluate(trial, "candidate", correction_scale=scale)
                        actual = trial[:19].astype(float) - base[:19]
                        report["evaluations"][-1]["linear_fit_max_ratio_error"] = float(np.max(abs(
                            values / self._limits - baseline_values / self._limits - matrix @ actual)))
                        distance = float(actual @ actual)
                        if passes and distance < best_distance:
                            best, best_prediction, best_distance = trial, prediction, distance
                if best is None:
                    report.update(status="no_verified_candidate", reason=("linearized_" + linear["status"] if delta is None
                        else "nonlinear_sequences_fail_measured_constraints"))
            if best is not None:
                repeated, _, passes = evaluate(best, "selected_repeat")
                if (not passes or any(repeated["controls"][i][key] != best_prediction["controls"][i][key]
                        for i in range(2) for key in ("endpoint", "substep_contacts", "human_action", "initial_time_s"))):
                    raise RuntimeError("selected_sequence_not_repeatable")
                pair = (best.copy(), fixed.copy())
                report.update(status="unchanged" if best_distance == 0 else "selected",
                    reason="measured_boundary_pair_repeated", correction_squared_norm=best_distance,
                    verified_sequence={"initial_time_s": initial_time, "controls": repeated["controls"]})
            check()
        except Exception as error:
            pair = None
            report.update(status="unavailable", reason=str(error)[:300], verified_sequence=None)
            if pending is not None:
                report["failed_evaluation"] = {**pending, "reason": report["reason"]}
        try:
            if len(json.dumps(report, allow_nan=False, separators=(",", ":")).encode()) > MAX_RESULT_BYTES - 8192:
                raise ValueError("sequence_projection_result_byte_budget")
        except (ValueError, TypeError) as error:
            pair = None
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
            pair = None
            report.update(status="unavailable", reason=str(error)[:300], verified_sequence=None)
        self._seconds = time.monotonic() - started
        if self._seconds >= self._config.maximum_seconds:
            pair = None
            report.update(status="unavailable", reason="sequence_projection_active_budget", verified_sequence=None)
        report.update(self.describe())
        return pair, report


def prepare_action_sequence_projector(env, *, config=SequenceProjectionConfig()):
    return ActionSequenceProjector(env, config)
