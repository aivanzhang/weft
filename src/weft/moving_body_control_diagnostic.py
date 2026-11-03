"""One frozen-policy trace; records existing engine telemetry without changing its controls or scorer."""

from hashlib import sha256
import json
import math
from pathlib import Path
import time

import numpy as np

from weft.io import canonical_hash
from weft import moving_equipment_learning as learning
from weft.moving_equipment_task import MovingEquipmentTaskEnv
from weft.shared_world_packaging import write_package_files
from weft.spatial_scene_inputs import _json

PROFILE = "moving-body-control-diagnostic-v1"
MAX_SUBSTEPS = 16_000
MAX_CONTROLS = 1_000
MAX_TRACE_BYTES = 64 << 20
SCOPE = ("Unchanged frozen-policy diagnostic in an authored simulated world. Reference jumps, saturation and contacts "
         "are measured together, not isolated causal effects or physical accuracy. No policy, task or source repair.")


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Body-control diagnostic reached its aggregate deadline")


def _pinned(raw, pin):
    if type(raw) is not bytes or not 0 < len(raw) <= 64 << 10 or sha256(raw).hexdigest() != pin:
        raise ValueError("Exact bounded policy and expected-metrics byte pins required")
    return _json(raw, 64 << 10)


class _Trace:
    def __init__(self, root, deadline):
        self.deadline = deadline
        self.size = 0
        self.files = {}
        self.handles = {}
        try:
            for name in ("controls.jsonl", "substeps.jsonl"):
                self.handles[name] = (root/name).open("xb")
        except Exception:
            for handle in self.handles.values():
                handle.close()
            raise
        self.hashes = {name: sha256() for name in self.handles}

    def append(self, name, row):
        _check(self.deadline)
        raw = learning._bytes(row, 64 << 10)
        if self.size + len(raw) > MAX_TRACE_BYTES:
            raise ValueError("Body-control trace exceeds its 64 MiB bound")
        self.handles[name].write(raw)
        self.hashes[name].update(raw)
        self.size += len(raw)
        _check(self.deadline)

    def close(self):
        for name, handle in self.handles.items():
            handle.close()
            self.files[name] = self.hashes[name].hexdigest()


def _vector(value):
    values = np.asarray(value, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Finite one-dimensional native telemetry required")
    return values.tolist()


class _Observer:
    def __init__(self, env, trace, deadline):
        self.engine, self.trace, self.deadline = env._engine, trace, deadline
        self.original = self.engine._substep_observer
        self.names = list(self.engine.human_action_names)
        self.count = len(self.names)
        self.angular_count = len(self.engine.joint_names)
        self.samples = 0
        self.integrals = np.zeros(self.count)
        self.maximum = np.zeros(self.count)
        self.maximum_times = np.zeros(self.count)
        self.saturation = np.zeros(self.count, dtype=int)
        self.peak = None
        self.invalid = None

    def __call__(self, sample):
        # The original scorer always receives its untouched sample exactly once.
        self.original(sample)
        _check(self.deadline)
        if self.samples >= MAX_SUBSTEPS:
            raise ValueError("Body-control diagnostic exceeds 16000 native samples")
        self.samples += 1
        if not sample.valid:
            self.invalid = {"step": sample.step_index, "substep": sample.substep_index, "failure": sample.failure}
            self.trace.append("substeps.jsonl", {"valid": False, **self.invalid})
            return
        engine, data = self.engine, self.engine.data
        q, target, action = map(np.asarray, (sample.human_qpos, sample.target_qpos, sample.human_action))
        error = q-target
        if any(x.shape != (self.count,) for x in (q, target, action)) or not np.isfinite(error).all():
            raise ValueError("Diagnostic human telemetry layout differs")
        duration = sample.interval_end_s-sample.interval_start_s
        if not 0 < duration <= engine.dt:
            raise ValueError("Diagnostic native sample duration differs")
        absolute = np.abs(error)
        changed = absolute > self.maximum
        self.maximum[changed], self.maximum_times[changed] = absolute[changed], sample.interval_end_s
        self.integrals += error**2*duration
        self.saturation += np.abs(action) >= 1.-1e-12
        angular_error = float(np.sqrt(np.mean(error[:self.angular_count]**2)))
        # implicitfast force/contact caches describe the solve start; qpos/qvel are integrated endpoints.
        contacts = []
        if data.ncon > 128:
            raise ValueError("Diagnostic contact inventory exceeds128; no silent truncation")
        for index in range(data.ncon):
            contact = data.contact[index]
            names = [engine.model.geom(int(gid)).name for gid in (contact.geom1, contact.geom2)]
            contacts.append({"geometries": names, "distance_world": float(contact.dist)})
        object_contacts = [{"human_geom": c.human_geom, "object_id": c.object_id, "distance_world": c.distance,
            "force_on_human_world": list(c.force_on_human_world), "normal_force": c.normal_force} for c in sample.contacts]
        row = {"valid": True, "step": sample.step_index, "substep": sample.substep_index,
            "start_s": sample.interval_start_s, "end_s": sample.interval_end_s, "solve_cache_time_s": sample.contact_time_s,
            "qpos": _vector(q), "qvel": list(sample.human_qvel), "target_qpos": _vector(target),
            "held_human_action": _vector(action), "joint_error": _vector(error), "angular_rmse_rad": angular_error,
            "actuator_force": _vector(data.actuator_force[engine._actuators]),
            "bias_force": _vector(data.qfrc_bias[engine._v]), "passive_force": _vector(data.qfrc_passive[engine._v]),
            "constraint_force": _vector(data.qfrc_constraint[engine._v]), "solve_acceleration": _vector(data.qacc[engine._v]),
            "contacts": contacts, "human_object_contacts": object_contacts,
            "equipment": [{"id": item.object_id, "position": list(item.position), "velocity": list(item.velocity),
                           "virtual_force": list(item.virtual_force)} for item in sample.equipment],
            "floor_penetration_model_units": sample.max_human_floor_penetration_model_units}
        if self.peak is None or angular_error > self.peak["angular_rmse_rad"]:
            self.peak = row
        self.trace.append("substeps.jsonl", row)

    def summary(self, elapsed):
        return {"samples": self.samples, "invalid_sample": self.invalid, "peak_angular_sample": self.peak,
            "axes": [{"name": name, "maximum_abs_error": float(self.maximum[i]),
                "maximum_error_time_s": float(self.maximum_times[i]),
                "time_weighted_rmse": math.sqrt(float(self.integrals[i])/elapsed) if elapsed else None,
                "saturated_samples": int(self.saturation[i]), "units": "rad" if i < self.angular_count else "world_length"}
                for i, name in enumerate(self.names)],
            "force_scope": "Cached pre-integration solve forces, not inverse-dynamics requirements. Constraint force includes contacts/limits; no floor-load equilibrium inference.",
            "saturation_scope": "Held normalized action magnitude>=1-1e-12; not proof the reference is infeasible."}


def diagnose_moving_body_control(admission, policy_raw, out, *, policy_sha256, expected_metrics_raw,
                                 expected_metrics_sha256, max_seconds=30):
    """Admission is supplied once by the caller; one exact-start episode, no optimization or source reload."""
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 0 < max_seconds <= 60:
        raise ValueError("Diagnostic deadline must be positive and at most60 seconds")
    started = time.monotonic(); deadline = started+max_seconds
    policy = _pinned(policy_raw, policy_sha256)
    expected = _pinned(expected_metrics_raw, expected_metrics_sha256)
    admission, scene, task, _ = learning._context(admission)
    horizon = task["interval"]["horizon_steps"]
    if horizon > MAX_CONTROLS or horizon*scene["substeps"] > MAX_SUBSTEPS:
        raise ValueError("Diagnostic requires the full interval within1000controls/16000substeps")
    if expected.get("task_sha256") != canonical_hash(task):
        raise ValueError("Expected metrics must bind the exact unchanged task")
    # The fixed existing validator authenticates policy/source/runtime; it never runs an episode.
    policy = learning.validate_moving_equipment_policy(policy, admission)
    _check(deadline)
    hashes, versions, _ = learning._runtime(admission, scene, deadline)
    own_raw = Path(__file__).read_bytes(); own_pin = sha256(own_raw).hexdigest()
    target = Path(out).absolute()
    if ".." in target.parts or target.exists() or any(p.is_symlink() for p in (target, *target.parents)):
        raise ValueError("Choose a new unlinked diagnostic output directory")
    target.mkdir(parents=True)
    write_package_files(target, {"policy.json": policy_raw, "expected-metrics.json": expected_metrics_raw,
        "implementation/moving_body_control_diagnostic.py": own_raw})
    trace = _Trace(target, deadline)
    env = observer = None
    controls = 0; metrics = None; failure = None; causes = []; matched = False; runtime_rechecked = False
    try:
        _check(deadline)
        env = MovingEquipmentTaskEnv(scene, task, admission.bundle)
        observer = _Observer(env, trace, deadline)
        env._engine._substep_observer = observer
        env.reset(seed=policy["training_seed"])
        feedback = learning._Feedback(env, scene, task, policy["weights"])
        while True:
            _check(deadline)
            if controls >= horizon:
                raise ValueError("Engine did not finish within the complete original interval")
            engine = env._engine
            q, velocity, target_q, target_v, features = engine._human_state()
            action = feedback()
            weights = np.asarray(policy["weights"][:12]).reshape(4, 3)
            unclipped = np.sum(features*weights[engine.group_indices], axis=1)
            trace.append("controls.jsonl", {"step": controls+1, "start_s": float(engine.data.time),
                "qpos": _vector(q), "qvel": _vector(velocity), "target_qpos": _vector(target_q),
                "target_qvel": _vector(target_v), "features": features.tolist(),
                "unclipped_human_action": _vector(unclipped), "action": _vector(action),
                "physical_motor_limits": _vector(engine.human_limits)})
            controls += 1
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        metrics = env.metrics()
        matched = canonical_hash(metrics) == canonical_hash(expected)
        final_hashes, final_versions, _ = learning._runtime(admission, scene, deadline)
        _check(deadline)
        if hashes != final_hashes or versions != final_versions or Path(__file__).read_bytes() != own_raw:
            raise ValueError("Diagnostic runtime changed during the episode")
        runtime_rechecked = True
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"[:1000]
        cause = error
        while cause is not None and len(causes) < 4:
            causes.append(f"{type(cause).__name__}: {cause}"[:1000])
            cause = cause.__cause__
        if env is not None and env._metrics is not None:
            metrics = env.metrics()
    finally:
        if env is not None:
            env._engine._substep_observer = observer.original if observer is not None else env._engine._substep_observer
            env.close()
        trace.close()
    report = {"schema_version": 1, "profile": PROFILE,
        "status": "complete" if failure is None and matched else "baseline_mismatch" if failure is None else "incomplete",
        "failure": failure, "failure_chain": causes, "scope": SCOPE, "source": learning._pins(admission, scene, task),
        "policy_sha256": policy_sha256, "expected_metrics_sha256": expected_metrics_sha256,
        "expected_metrics_scope": "Caller-pinned saved metric bytes; caller retains their original report identity and extraction rule. Not an independent physical oracle.",
        "metrics": metrics, "exact_saved_metrics_match": matched, "runtime_readback_complete": runtime_rechecked,
        "runtime_sha256": {**hashes, "moving_body_control_diagnostic.py": own_pin}, "runtime_versions": versions,
        "control_steps": controls, "native_substeps": observer.samples if observer else 0,
        "model_compilations": int(env is not None), "diagnostic_extra_forward_inverse_step_calls": 0,
        "trace_bytes": trace.size, "diagnosis": observer.summary(metrics["elapsed_s"] if metrics else 0) if observer else None,
        "seconds": time.monotonic()-started, "api_calls": 0, "inference_calls": 0,
        "policy_installed": False, "closed": True}
    if time.monotonic() >= deadline:
        report.update(status="incomplete", failure=report["failure"] or "Aggregate deadline exceeded during finalization")
    encoded = learning._bytes(report, 2 << 20)
    if time.monotonic() >= deadline and report["status"] != "incomplete":
        report.update(status="incomplete", failure="Aggregate deadline exceeded during report encoding")
        encoded = learning._bytes(report, 2 << 20)
    write_package_files(target, {"result.json": encoded})
    files = {**trace.files, "result.json": sha256(encoded).hexdigest(), "policy.json": policy_sha256,
        "expected-metrics.json": expected_metrics_sha256, "implementation/moving_body_control_diagnostic.py": own_pin}
    write_package_files(target, {"manifest.json": learning._bytes({"schema_version": 1, "profile": PROFILE, "status": report["status"], "files": files})})
    return report
