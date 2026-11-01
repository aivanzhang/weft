"""Bounded open-loop forecasts of fixed actions on one copied simulator state."""

from copy import deepcopy
from hashlib import sha256
import json
import math
import threading
import time
from types import SimpleNamespace

import mujoco
import numpy as np

from weft.shared_exercise_action_prediction import ActionPredictor, STATE_FIELDS, _callback_issue, _identity
from weft.shared_world import SharedWorldEnv


PROFILE = "shared-exercise-detached-sequence-prediction-v1"
MAX_CONTROLS = 64
MAX_SEQUENCE_CONTROLS = 4
MAX_RESULT_BYTES = 2 << 20
SCOPE = ("Counted hypothetical open-loop integration of explicit fixed actions on copied simulator data. "
         "No policy callback, optimization, live control, task reward or task termination is evaluated or changed. "
         "Partial forecasts are unavailable, not shortened successful sequences. Native parity requires separate "
         "runtime qualification; no collision guarantee, task-success or physical-validation claim. Caller must "
         "serialize live commands and model changes; this lock only excludes overlapping sequence predictions.")
_COUNTERS = ("native_substeps", "native_substeps_attempted", "boundary_forwards", "boundary_forwards_attempted",
             "copies", "copies_attempted", "data_allocations", "data_allocations_attempted")


class ActionSequencePredictor:
    def __init__(self, env, *, maximum_sequences, maximum_controls, maximum_sequence_controls, maximum_seconds):
        for name, value, maximum in (("maximum_sequences", maximum_sequences, MAX_CONTROLS),
                ("maximum_controls", maximum_controls, MAX_CONTROLS),
                ("maximum_sequence_controls", maximum_sequence_controls, MAX_SEQUENCE_CONTROLS)):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer from one to {maximum}")
        # Reuse the frozen model, action-contract, motor and timing validation without integrating.
        self._validated = ActionPredictor(env, maximum_predictions=maximum_controls, maximum_seconds=maximum_seconds)
        self._maximum_sequences, self._maximum_controls = maximum_sequences, maximum_controls
        self._maximum_sequence_controls, self._maximum_seconds = maximum_sequence_controls, float(maximum_seconds)
        self._sequences, self._controls, self._seconds, self._scratch = 0, 0, 0., None
        self._lock = threading.Lock()

    def describe(self):
        return {"profile": PROFILE, "scope": SCOPE, "maximum_sequences": self._maximum_sequences,
            "maximum_controls": self._maximum_controls, "maximum_sequence_controls": self._maximum_sequence_controls,
            "maximum_native_substeps": self._maximum_controls * self._validated._substeps,
            "maximum_active_compute_seconds": self._maximum_seconds, "maximum_result_bytes": MAX_RESULT_BYTES,
            "sequences_used": self._sequences, "controls_reserved": self._controls,
            "active_compute_seconds_used": self._seconds, "scratch_data_objects": int(self._scratch is not None),
            "copies_per_sequence": 1, "boundary_forwards_per_control": 1,
            "clock_scope": "substep contacts at solver interval start; boundary contacts after each final forward"}

    def predict(self, actions):
        if not self._lock.acquire(blocking=False):
            return {"profile": PROFILE, "status": "unavailable", "reason": "sequence_prediction_already_running",
                    "sequence_complete": False, "endpoint": None}
        try:
            return self._predict(actions)
        finally:
            self._lock.release()

    def _actions(self, actions):
        if not isinstance(actions, (list, tuple, np.ndarray)) or isinstance(actions, np.ndarray) and actions.ndim != 2:
            raise ValueError("Sequence requires a materialized list, tuple or two-dimensional array of fixed actions")
        if not 1 <= len(actions) <= self._maximum_sequence_controls:
            raise ValueError("Sequence action count exceeds its explicit bounded horizon")
        result, count = [], self._validated._env.action_count
        if isinstance(actions, np.ndarray) and (actions.shape[1] != count or actions.dtype.kind not in "fiu"):
            raise ValueError("Sequence actions must match the bounded numeric action contract")
        for action in actions:
            if (not isinstance(action, (list, tuple, np.ndarray))
                    or isinstance(action, np.ndarray) and (action.ndim != 1 or action.shape != (count,))
                    or isinstance(action, (list, tuple)) and (len(action) != count or any(
                        not isinstance(value, (int, float, np.integer, np.floating))
                        or isinstance(value, (bool, np.bool_)) for value in action))):
                raise ValueError("Sequence action must match the finite bounded human-only action contract")
            values = np.asarray(action)
            if (values.shape != (count,) or values.dtype.kind not in "fiu" or not np.isfinite(values).all()
                    or np.any(values < -1) or np.any(values > 1)
                    or isinstance(action, (list, tuple)) and any(isinstance(v, (bool, np.bool_)) for v in action)):
                raise ValueError("Sequence action must match the finite bounded human-only action contract")
            result.append(values.astype(float, copy=True))
        return result

    def _predict(self, actions):
        started = time.monotonic()
        values = self._actions(actions)
        v = self._validated
        report = {"profile": PROFILE, "status": "unavailable", "reason": None, "scope": SCOPE,
            "sequence_complete": False, "endpoint": None, "controls": [], "requested_controls": len(values),
            "actions": [action.tolist() for action in values], "controls_reserved": 0,
            "controls_attempted": 0, "controls_completed": 0, "native_parity_claim": False,
            "live_state_unchanged": None, **dict.fromkeys(_COUNTERS, 0)}
        before, start_steps, retained_bytes = None, None, 0

        def check():
            if time.monotonic() - started + self._seconds >= self._maximum_seconds:
                raise TimeoutError("sequence_active_compute_budget")
            issue = _callback_issue()
            if issue:
                raise RuntimeError(issue)
            if v._env._engine is not v._engine or v._engine.model is not v._model or v._engine.data is not v._live:
                raise RuntimeError("engine_model_or_data_replaced")
            if (not v._engine._initialized or v._engine._done or v._env._done
                    or v._engine.steps >= min(v._env.horizon_steps, v._engine.horizon_steps)
                    or start_steps is not None and v._engine.steps != start_steps):
                raise RuntimeError("environment_not_ready_for_sequence")
            if start_steps is not None and start_steps + len(values) > min(v._env.horizon_steps, v._engine.horizon_steps):
                raise RuntimeError("insufficient_live_control_horizon")
            if (v._model.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST
                    or v._engine.substeps != v._substeps or v._engine.dt != v._dt
                    or not np.array_equal(v._engine.human_limits, v._limits)
                    or not np.array_equal(v._engine._actuators, v._actuators)
                    or not math.isclose(v._model.opt.timestep * v._substeps, v._dt, rel_tol=0, abs_tol=1e-12)):
                raise RuntimeError("model_timing_changed")

        def unavailable(reason):
            report.update(status="unavailable", reason=reason, sequence_complete=False, endpoint=None)

        try:
            if self._sequences >= self._maximum_sequences:
                raise RuntimeError("sequence_call_budget")
            if self._controls + len(values) > self._maximum_controls:
                raise RuntimeError("sequence_control_budget")
            self._sequences += 1
            check()
            start_steps = v._engine.steps
            if start_steps + len(values) > min(v._env.horizon_steps, v._engine.horizon_steps):
                raise RuntimeError("insufficient_live_control_horizon")
            # Reserving the whole sequence prevents retrying failed partial work outside its budget.
            self._controls += len(values)
            report["controls_reserved"] = len(values)
            before = _identity(v._live)
            if self._scratch is None:
                report["data_allocations_attempted"] += 1
                self._scratch = mujoco.MjData(v._model)
                report["data_allocations"] += 1
            if self._scratch is v._live:
                raise RuntimeError("scratch_aliases_live_data")
            report["copies_attempted"] += 1
            mujoco.mj_copyData(self._scratch, v._model, v._live)
            report["copies"] += 1
            if _identity(self._scratch) != before:
                raise RuntimeError("scratch_copy_identity_mismatch")
            data = self._scratch
            view = SimpleNamespace(**v._view_fields, data=data)
            issue = SharedWorldEnv._instability(view)
            if issue or not math.isfinite(float(data.time)):
                raise RuntimeError("initial_state:" + (issue or "nonfinite_time"))
            report["initial_time_s"] = float(data.time)
            for control, action in enumerate(values, 1):
                check()
                human = np.zeros(22); human[:v._env.action_count] = action
                row = {"control": control, "initial_time_s": float(data.time), "human_action": human.tolist(),
                    "complete": False, "native_substeps": 0, "native_substeps_attempted": 0,
                    "boundary_forwards": 0, "boundary_forwards_attempted": 0,
                    "substep_contacts": [], "endpoint": None}
                report["controls"].append(row)
                report["controls_attempted"] += 1
                data.ctrl[:] = 0.
                data.ctrl[v._actuators] = human * v._limits
                data.qfrc_applied[:] = 0.
                data.xfrc_applied[:] = 0.
                for index in range(v._substeps):
                    check()
                    solver_time = float(data.time)
                    row["native_substeps_attempted"] += 1
                    report["native_substeps_attempted"] += 1
                    mujoco.mj_step(v._model, data)
                    row["native_substeps"] += 1
                    report["native_substeps"] += 1
                    issue = SharedWorldEnv._instability(view)
                    if issue or not math.isfinite(float(data.time)):
                        raise RuntimeError("predicted_instability:" + (issue or "nonfinite_time"))
                    row["substep_contacts"].append({"substep": index + 1, "solver_time_s": solver_time,
                        "integrated_time_s": float(data.time),
                        "contacts": deepcopy(v._engine._current_contacts(data=data))})
                check()
                row["boundary_forwards_attempted"] += 1
                report["boundary_forwards_attempted"] += 1
                mujoco.mj_forward(v._model, data)
                row["boundary_forwards"] += 1
                report["boundary_forwards"] += 1
                if not math.isclose(float(data.time), row["initial_time_s"] + v._dt, rel_tol=0, abs_tol=1e-10):
                    raise RuntimeError("predicted_clock_mismatch")
                endpoint = {name: np.asarray(getattr(data, name)).copy() for name in STATE_FIELDS}
                if any(not np.isfinite(value).all() for value in endpoint.values()):
                    raise RuntimeError("nonfinite_predicted_endpoint")
                row["endpoint"] = {"time_s": float(data.time), **{k: a.tolist() for k, a in endpoint.items()},
                    "contacts": deepcopy(v._engine._current_contacts(data=data))}
                check()
                row["complete"] = True
                report["controls_completed"] += 1
                retained_bytes += len(json.dumps(row, allow_nan=False, separators=(",", ":")).encode())
                if retained_bytes > MAX_RESULT_BYTES - 16384:
                    raise RuntimeError("sequence_result_byte_budget")
                check()
            report.update(status="predicted", sequence_complete=True, endpoint=report["controls"][-1]["endpoint"])
        except Exception as error:
            report["failure_type"] = type(error).__name__
            unavailable(str(error)[:500])
        finally:
            if before is not None:
                report["live_state_unchanged"] = _identity(v._live) == before
                if not report["live_state_unchanged"]:
                    unavailable("live_state_changed_during_sequence")
            report.update(seconds=time.monotonic() - started, sequences_used=self._sequences, controls_used=self._controls)
            if self._seconds + report["seconds"] >= self._maximum_seconds:
                unavailable("sequence_active_compute_budget")
        encoded = None
        try:
            encoded = json.dumps(report, allow_nan=False, separators=(",", ":")).encode()
            if len(encoded) > MAX_RESULT_BYTES - 2048:
                raise ValueError("sequence_result_byte_budget")
        except (ValueError, TypeError) as error:
            unavailable(str(error)[:500])
            report["omitted_detail"] = {"reason": report["reason"], "control_rows": len(report["controls"]),
                "endpoints": sum(row["endpoint"] is not None for row in report["controls"]),
                "substep_contacts": sum(len(row["substep_contacts"]) for row in report["controls"]),
                "pre_omission_sha256": sha256(encoded).hexdigest() if encoded is not None else None}
            for row in report["controls"]:
                row.update(endpoint=None, substep_contacts=[])
        try:
            check()
        except Exception as error:
            unavailable(str(error)[:500])
        if before is not None:
            report["live_state_unchanged"] = _identity(v._live) == before
            if not report["live_state_unchanged"]:
                unavailable("live_state_changed_during_sequence")
        report["seconds"] = time.monotonic() - started
        self._seconds += report["seconds"]
        if self._seconds >= self._maximum_seconds:
            unavailable("sequence_active_compute_budget")
        return report


def prepare_action_sequence_predictor(env, *, maximum_sequences=4, maximum_controls=8,
                                     maximum_sequence_controls=4, maximum_seconds=2.):
    return ActionSequencePredictor(env, maximum_sequences=maximum_sequences, maximum_controls=maximum_controls,
        maximum_sequence_controls=maximum_sequence_controls, maximum_seconds=maximum_seconds)
