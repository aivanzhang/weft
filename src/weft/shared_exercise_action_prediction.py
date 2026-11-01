"""Counted detached one-control predictions; no optimizer, scorer or live-state advance."""

from copy import deepcopy
import json
import math
import threading
import time
from types import SimpleNamespace

import mujoco
import numpy as np

from weft.shared_world import SharedWorldEnv
from weft.spatial_pose import SPATIAL_JOINT_NAMES


PROFILE = "shared-exercise-detached-action-prediction-v1"
MAX_PREDICTIONS = 64
MAX_SUBSTEPS = 16
MAX_SECONDS = 10.
MAX_RESULT_BYTES = 2 << 20
_CALLBACK_GETTERS = tuple(sorted(name for name in dir(mujoco) if name.startswith("get_mjcb_")))
STATE_FIELDS = ("qpos", "qvel", "act", "ctrl", "qacc", "qacc_warmstart", "qfrc_applied",
                "xfrc_applied", "qfrc_bias", "qfrc_passive", "qfrc_actuator", "qfrc_constraint",
                "qfrc_smooth", "qacc_smooth", "xpos", "xquat", "geom_xpos", "geom_xmat",
                "userdata", "mocap_pos", "mocap_quat", "plugin_state")
SCOPE = ("Explicitly counted hypothetical one-control integration on copied simulator data. "
         "No live state, controller, task reward or success is updated. Not an optimizer, learned policy, "
         "collision guarantee or physical validation. Native parity requires separate runtime qualification. "
         "Caller must serialize live environment commands and model changes; this lock only excludes concurrent predictions.")


def _callback_issue():
    for name in _CALLBACK_GETTERS:
        if getattr(mujoco, name)() is not None:
            return "unexpected_callback:" + name
    return None


def _identity(data):
    return (np.float64(data.time).tobytes(), *(np.asarray(getattr(data, name)).tobytes() for name in STATE_FIELDS))


class ActionPredictor:
    def __init__(self, env, *, maximum_predictions, maximum_seconds):
        if type(maximum_predictions) is not int or not 1 <= maximum_predictions <= MAX_PREDICTIONS:
            raise ValueError("Prediction count must be an integer from one to sixty-four")
        if (type(maximum_seconds) not in (int, float) or not math.isfinite(maximum_seconds)
                or not .001 <= maximum_seconds <= MAX_SECONDS):
            raise ValueError("Prediction active-compute budget must be .001–10 seconds")
        scene, task = env.scene, env.task
        if (scene.get("profile") not in {"shared-world-v1", "shared-world-v2"}
                or task.get("profile") != "shared-exercise-task-v1"):
            raise ValueError("Detached prediction requires the original coupled equipment task")
        if (env.action_count not in (19, 22) or len(env.action_names) != env.action_count
                or tuple(env.action_names[:19]) != SPATIAL_JOINT_NAMES
                or env.action_count == 22 and tuple(env.action_names[19:]) != ("root_x", "root_y", "root_z")):
            raise ValueError("Detached prediction requires nineteen joints and optional original root controls")
        self._env, self._engine = env, env._engine
        self._model, self._live = self._engine.model, self._engine.data
        model, engine = self._model, self._engine
        if (model.nu != 22 or model.na or model.nplugin or model.nsensor or
                not 22 <= model.nv <= 128 or not 22 <= model.nq <= 160
                or not 1 <= model.nbody <= 128 or not 1 <= model.ngeom <= 128
                or tuple(engine.human_action_names[:env.action_count]) != tuple(env.action_names)
                or engine.human_motor_count != 22 or engine.ankle_names):
            raise ValueError("Detached prediction model exceeds its original rigid-motor envelope")
        if (type(engine.substeps) is not int or not 1 <= engine.substeps <= MAX_SUBSTEPS
                or model.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST
                or not math.isclose(model.opt.timestep * engine.substeps, engine.dt, rel_tol=0, abs_tol=1e-12)):
            raise ValueError("Detached prediction requires matching bounded implicitfast timing")
        self._substeps, self._dt = engine.substeps, float(engine.dt)
        self._limits = np.asarray(engine.human_limits, dtype=float).copy()
        self._actuators = np.asarray(engine._actuators, dtype=int).copy()
        if (self._limits.shape != (22,) or not np.isfinite(self._limits).all() or np.any(self._limits <= 0)
                or self._actuators.shape != (22,) or set(self._actuators) != set(range(22))):
            raise ValueError("Detached prediction requires the exact finite physical motor map")
        self._limits.flags.writeable = self._actuators.flags.writeable = False
        self._maximum_predictions, self._maximum_seconds = maximum_predictions, float(maximum_seconds)
        self._calls, self._seconds, self._scratch = 0, 0., None
        self._lock = threading.Lock()
        self._view_fields = {name: getattr(engine, name) for name in (
            "model", "scene", "scale", "_q", "_v", "ankle_names", "_dynamic_objects", "_object_body_ids")}

    def describe(self):
        return {"profile": PROFILE, "scope": SCOPE, "maximum_predictions": self._maximum_predictions,
                "maximum_native_substeps": self._maximum_predictions * self._substeps,
                "maximum_active_compute_seconds": self._maximum_seconds, "substeps_per_prediction": self._substeps,
                "boundary_forwards_per_prediction": 1, "maximum_result_bytes": MAX_RESULT_BYTES,
                "calls_used": self._calls, "active_compute_seconds_used": self._seconds,
                "scratch_data_objects": int(self._scratch is not None),
                "clock_scope": "substep contacts at solver interval start; boundary contacts after final forward"}

    def predict(self, action):
        if not self._lock.acquire(blocking=False):
            return {"profile": PROFILE, "status": "unavailable", "reason": "prediction_already_running"}
        try:
            return self._predict(action)
        finally:
            self._lock.release()

    def _predict(self, action):
        started = time.monotonic()
        report = {"profile": PROFILE, "status": "unavailable", "reason": None, "scope": SCOPE,
                  "native_substeps": 0, "boundary_forwards": 0, "copies": 0,
                  "native_substeps_attempted": 0, "boundary_forwards_attempted": 0, "copies_attempted": 0,
                  "data_allocations": 0, "data_allocations_attempted": 0,
                  "substep_contacts": [], "endpoint": None, "native_parity_claim": False}
        values = np.asarray(action)
        if (values.shape != (self._env.action_count,) or values.dtype.kind not in "fiu"
                or not np.isfinite(values).all() or np.any(abs(values) > 1)
                or isinstance(action, (list, tuple)) and any(isinstance(v, (bool, np.bool_)) for v in action)):
            raise ValueError("Prediction action must match the finite bounded human-only action contract")
        values = values.astype(float, copy=True)
        before = None

        def check():
            if time.monotonic() - started + self._seconds >= self._maximum_seconds:
                raise TimeoutError("prediction_active_compute_budget")
            issue = _callback_issue()
            if issue:
                raise RuntimeError(issue)

        try:
            if self._calls >= self._maximum_predictions:
                raise RuntimeError("prediction_call_budget")
            self._calls += 1
            check()
            if self._engine.model is not self._model or self._engine.data is not self._live:
                raise RuntimeError("engine_model_or_data_replaced")
            if (not self._engine._initialized or self._engine._done or self._env._done
                    or self._engine.steps >= self._env.horizon_steps):
                raise RuntimeError("environment_not_ready_for_control")
            if (self._model.opt.integrator != mujoco.mjtIntegrator.mjINT_IMPLICITFAST
                    or self._engine.substeps != self._substeps or self._engine.dt != self._dt
                    or not np.array_equal(self._engine.human_limits, self._limits)
                    or not np.array_equal(self._engine._actuators, self._actuators)
                    or not math.isclose(self._model.opt.timestep * self._substeps, self._dt, rel_tol=0, abs_tol=1e-12)):
                raise RuntimeError("model_timing_changed")
            before = _identity(self._live)
            if self._scratch is None:
                report["data_allocations_attempted"] += 1
                self._scratch = mujoco.MjData(self._model)
                report["data_allocations"] += 1
            if self._scratch is self._live:
                raise RuntimeError("scratch_aliases_live_data")
            report["copies_attempted"] += 1
            mujoco.mj_copyData(self._scratch, self._model, self._live)
            report["copies"] += 1
            if _identity(self._scratch) != before:
                raise RuntimeError("scratch_copy_identity_mismatch")
            data = self._scratch
            view = SimpleNamespace(**self._view_fields, data=data)
            issue = SharedWorldEnv._instability(view)
            if issue or not math.isfinite(float(data.time)):
                raise RuntimeError("initial_state:" + (issue or "nonfinite_time"))
            report["initial_time_s"] = float(data.time)
            human = np.zeros(22); human[:self._env.action_count] = values
            data.ctrl[:] = 0.
            data.ctrl[self._actuators] = human * self._limits
            data.qfrc_applied[:] = 0.
            data.xfrc_applied[:] = 0.
            report["human_action"] = human.tolist()
            for index in range(self._substeps):
                check()
                solver_time = float(data.time)
                report["native_substeps_attempted"] += 1
                mujoco.mj_step(self._model, data)
                report["native_substeps"] += 1
                issue = SharedWorldEnv._instability(view)
                if issue or not math.isfinite(float(data.time)):
                    raise RuntimeError("predicted_instability:" + (issue or "nonfinite_time"))
                contacts = self._engine._current_contacts(data=data)
                report["substep_contacts"].append({"substep": index + 1, "solver_time_s": solver_time,
                    "integrated_time_s": float(data.time), "contacts": deepcopy(contacts)})
            check()
            report["boundary_forwards_attempted"] += 1
            mujoco.mj_forward(self._model, data)
            report["boundary_forwards"] += 1
            if not math.isclose(float(data.time), report["initial_time_s"] + self._dt, rel_tol=0, abs_tol=1e-10):
                raise RuntimeError("predicted_clock_mismatch")
            endpoint = {name: np.asarray(getattr(data, name)).copy() for name in STATE_FIELDS}
            if any(not np.isfinite(value).all() for value in endpoint.values()):
                raise RuntimeError("nonfinite_predicted_endpoint")
            report["endpoint"] = {"time_s": float(data.time), **{k: v.tolist() for k, v in endpoint.items()},
                                  "contacts": deepcopy(self._engine._current_contacts(data=data))}
            check()
            report["status"] = "predicted"
        except (RuntimeError, ValueError, TimeoutError) as error:
            report.update(status="unavailable", reason=str(error)[:500], endpoint=None)
        finally:
            if before is not None and _identity(self._live) != before:
                report.update(status="unavailable", reason="live_state_changed_during_prediction", endpoint=None)
            report["seconds"] = time.monotonic() - started
            report["calls_used"] = self._calls
            if self._seconds + report["seconds"] >= self._maximum_seconds:
                report.update(status="unavailable", reason="prediction_active_compute_budget", endpoint=None)
        try:
            size = len(json.dumps(report, allow_nan=False, separators=(",", ":")).encode())
            if size > MAX_RESULT_BYTES - 1024:
                raise ValueError("prediction_result_byte_budget")
        except (ValueError, TypeError) as error:
            report.update(status="unavailable", reason=str(error)[:500], endpoint=None, substep_contacts=[])
        report["seconds"] = time.monotonic() - started
        self._seconds += report["seconds"]
        if self._seconds >= self._maximum_seconds:
            report.update(status="unavailable", reason="prediction_active_compute_budget", endpoint=None)
        return report


def prepare_action_predictor(env, *, maximum_predictions=4, maximum_seconds=2.):
    return ActionPredictor(env, maximum_predictions=maximum_predictions, maximum_seconds=maximum_seconds)
