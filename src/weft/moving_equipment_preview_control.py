"""Explicit interval-average velocity feedback; original positions, force limits and scorer remain unchanged."""
from hashlib import sha256
import json
import math
from pathlib import Path
import time

import numpy as np

from weft.io import canonical_hash
from weft import moving_equipment_learning as learning
from weft.moving_body_control_diagnostic import _Observer, _Trace, _pinned, _vector
from weft.moving_equipment_task import MovingEquipmentTaskEnv
from weft.native_interval import action_duration
from weft.shared_world_packaging import write_package_files

PROFILE = "moving-equipment-preview-control-v1"
FEATURE_PROFILE = "velocity_only_interval_average_v1"
SCOPE = ("One authored controller hypothesis using the already available future reference over the exact next action. "
         "Only target velocity changes; no source smoothing, new gains, force-limit change or scorer modification. "
         "Privileged simulated reference preview, not recovered intent or physical validation.")
MAX_CONTROLS = 1000
MAX_SUBSTEPS = 16000


def _bounded_vector(value):
    if type(value) is np.ndarray:
        valid = value.ndim == 1 and 1 <= value.size <= 28 and value.dtype.kind in "fiu"
    elif type(value) in (list, tuple):
        valid = 1 <= len(value) <= 28 and all(
            type(item) in (int, float) and -float("inf") < item < float("inf")
            and abs(item) <= float(np.finfo(float).max) for item in value)
    else:
        valid = False
    if not valid:
        raise ValueError("Preview requires bounded finite matching human target vectors")
    return np.asarray(value)


def interval_average_velocity(target_now, target_next, duration):
    if type(duration) not in (int, float) or not 0 < duration <= 1 or not math.isfinite(duration):
        raise ValueError("Preview requires the finite positive exact action duration")
    first, last = _bounded_vector(target_now), _bounded_vector(target_next)
    if (first.ndim != 1 or first.shape != last.shape or not 1 <= len(first) <= 28
            or first.dtype.kind not in "fiu" or last.dtype.kind not in "fiu"
            or not np.isfinite(first).all() or not np.isfinite(last).all()):
        raise ValueError("Preview requires bounded finite matching human target vectors")
    result = (last.astype(float)-first.astype(float))/duration
    if not np.isfinite(result).all():
        raise ValueError("Preview velocity is not finite")
    return result


class _PreviewControl:
    def __init__(self, env, scene, policy, contract):
        self._engine = env._engine
        self._base = learning._Feedback(env, scene, env.task, policy["weights"])
        self._scene = {"dt": scene["dt"], "task": json.loads(json.dumps(scene["task"]))}
        self._weights = np.asarray(policy["weights"][:12], dtype=float).reshape(4, 3).copy()
        self._weights.flags.writeable = False
        self._contract = learning._bytes(contract, 128 << 10)
        self._last = None

    @property
    def contract(self):
        return json.loads(self._contract)

    @property
    def last_diagnostic(self):
        return None if self._last is None else json.loads(self._last)

    def __call__(self):
        engine = self._engine
        if not engine._initialized or engine._done or not 0 <= engine.steps < self._scene["task"]["horizon_steps"]:
            raise RuntimeError("Preview requires a live reset-bound original interval")
        duration = action_duration(self._scene, engine.steps)
        q, velocity, target, instantaneous, original = engine._human_state()
        future = engine._target_at(float(engine.data.time)+duration)[0]
        average = interval_average_velocity(target, future, duration)
        features = original.copy()
        features[:, 1] = engine._kd*(average-velocity)/engine.human_limits
        unclipped = np.sum(features*self._weights[engine.group_indices], axis=1)
        # Existing object targets, force normalization and clipping stay byte-for-byte on their original path.
        action = self._base()
        action[:engine.human_motor_count] = np.clip(unclipped, -1., 1.)
        self._last = learning._bytes({"step": engine.steps+1, "start_s": float(engine.data.time), "action_duration_s": duration,
            "qpos": _vector(q), "qvel": _vector(velocity), "target_qpos": _vector(target),
            "target_qvel": _vector(average), "original_instantaneous_target_qvel": _vector(instantaneous),
            "next_target_qpos": _vector(future), "features": features.tolist(), "original_features": original.tolist(),
            "unclipped_human_action": _vector(unclipped), "action": _vector(action),
            "physical_motor_limits": _vector(engine.human_limits)}, 64 << 10)
        return action


def prepare_preview_control(admission, base_policy_raw, env, *, base_policy_sha256):
    admission, scene, task, _ = learning._context(admission)
    policy = _pinned(base_policy_raw, base_policy_sha256)
    policy = learning.validate_moving_equipment_policy(policy, admission)
    if (not isinstance(env, MovingEquipmentTaskEnv) or canonical_hash(env.scene) != canonical_hash(scene)
            or canonical_hash(env.task) != canonical_hash(task)):
        raise ValueError("Preview requires the exact admitted original scene and task")
    contract = {"schema_version": 1, "profile": PROFILE, "feature_profile": FEATURE_PROFILE,
        "base_policy_sha256": base_policy_sha256, "base_policy_canonical_sha256": canonical_hash(policy),
        "source": learning._pins(admission, scene, task), "action_names": policy["action_names"],
        "original_weights_sha256": canonical_hash(policy["weights"]),
        "runtime_sha256": {**policy["runtime_sha256"], **{name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("moving_equipment_preview_control.py", "moving_body_control_diagnostic.py")}},
        "runtime_versions": policy["runtime_versions"], "position_target": "unchanged_current_time_reference",
        "velocity_target": "exact_next_action_endpoint_difference_divided_by_actual_duration",
        "terminal_interval": "original_native_partial_duration", "object_controller": "unchanged_original_feedback",
        "policy_installed": False, "scope": SCOPE}
    return _PreviewControl(env, scene, policy, contract)


def evaluate_preview_control(admission, base_policy_raw, out, *, base_policy_sha256, baseline_metrics_raw,
                             baseline_metrics_sha256, max_seconds=30):
    """One fixed controller trial; the previously recorded baseline is not replayed or optimized."""
    if type(max_seconds) not in (int, float) or not 0 < max_seconds <= 60 or not math.isfinite(max_seconds):
        raise ValueError("Preview evaluation requires a positive deadline at most60seconds")
    started = time.monotonic(); deadline = started+max_seconds
    admission, scene, task, _ = learning._context(admission)
    baseline = _pinned(baseline_metrics_raw, baseline_metrics_sha256)
    policy = _pinned(base_policy_raw, base_policy_sha256)
    if baseline.get("task_sha256") != canonical_hash(task):
        raise ValueError("Preview baseline must bind the unchanged task")
    # Invalid original policy cannot compile a candidate model.
    policy = learning.validate_moving_equipment_policy(policy, admission)
    learning._check(deadline)
    horizon = task["interval"]["horizon_steps"]
    if horizon > MAX_CONTROLS or horizon*scene["substeps"] > MAX_SUBSTEPS:
        raise ValueError("Preview must retain the full interval within1000controls/16000substeps")
    path = Path(out).absolute()
    if path.exists() or ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Choose a new nonlinked preview evaluation directory")
    hashes, versions, _ = learning._runtime(admission, scene, deadline)
    extra = {name: Path(__file__).with_name(name).read_bytes() for name in
             ("moving_equipment_preview_control.py", "moving_body_control_diagnostic.py")}
    path.mkdir(parents=True)
    write_package_files(path, {"base-policy.json": base_policy_raw, "baseline-metrics.json": baseline_metrics_raw,
                              **{"implementation/"+n: raw for n, raw in extra.items()}})
    trace = _Trace(path, deadline)
    env = observer = controller = None
    controls = 0; failure = None; causes = []; metrics = None; runtime_rechecked = False
    try:
        learning._check(deadline)
        env = MovingEquipmentTaskEnv(scene, task, admission.bundle)
        controller = prepare_preview_control(admission, base_policy_raw, env, base_policy_sha256=base_policy_sha256)
        observer = _Observer(env, trace, deadline); env._engine._substep_observer = observer
        env.reset(seed=policy["training_seed"])
        while True:
            learning._check(deadline)
            if controls >= horizon:
                raise ValueError("Preview exceeded the original full control interval")
            action = controller(); trace.append("controls.jsonl", controller.last_diagnostic)
            controls += 1
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        metrics = env.metrics()
        current, current_versions, _ = learning._runtime(admission, scene, deadline)
        learning._check(deadline)
        if (current != hashes or versions != current_versions
                or any(Path(__file__).with_name(n).read_bytes() != raw for n, raw in extra.items())):
            raise ValueError("Preview runtime changed during the trial")
        runtime_rechecked = True
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"[:1000]
        cause = error
        while cause is not None and len(causes) < 4:
            causes.append(f"{type(cause).__name__}: {cause}"[:1000]); cause = cause.__cause__
        if env is not None and env._metrics is not None:
            metrics = env.metrics()
    finally:
        if env is not None:
            if observer is not None: env._engine._substep_observer = observer.original
            env.close()
        trace.close()
    compared = (*task["body_gates"], "object_distance_world", "object_speed_world_s", "terminal_consecutive_hold_s", "mean_reward")
    differences = {name: metrics[name]-baseline[name] for name in compared
        if metrics is not None and type(metrics.get(name)) in (int, float) and type(baseline.get(name)) in (int, float)}
    report = {"schema_version": 1, "profile": PROFILE, "feature_profile": FEATURE_PROFILE,
        "status": "complete" if failure is None else "incomplete", "failure": failure, "failure_chain": causes,
        "scope": SCOPE, "source": learning._pins(admission, scene, task),
        "base_policy_sha256": base_policy_sha256, "baseline_metrics_sha256": baseline_metrics_sha256,
        "baseline_scope": "Caller-pinned prior saved metrics; not rerun or a new independent holdout.",
        "controller_contract": controller.contract if controller else None,
        "metrics": metrics, "differences_candidate_minus_baseline": differences,
        "runtime_readback_complete": runtime_rechecked,
        "control_steps": controls, "native_substeps": observer.samples if observer else 0,
        "diagnosis": observer.summary(metrics["elapsed_s"] if metrics else 0) if observer else None,
        "model_compilations": int(env is not None), "trace_bytes": trace.size, "seconds": time.monotonic()-started,
        "changed_start_evaluation": "not_run", "policy_installed": False, "api_calls": 0, "inference_calls": 0,
        "baseline_replays": 0, "training_runs": 0, "closed": True}
    if time.monotonic() >= deadline:
        report.update(status="incomplete", failure=report["failure"] or "Aggregate deadline exceeded during finalization")
    raw = learning._bytes(report, 2 << 20)
    if time.monotonic() >= deadline and report["status"] != "incomplete":
        report.update(status="incomplete", failure="Aggregate deadline exceeded during encoding"); raw = learning._bytes(report, 2 << 20)
    write_package_files(path, {"result.json": raw})
    files = {**trace.files, "result.json": sha256(raw).hexdigest(), "base-policy.json": base_policy_sha256,
        "baseline-metrics.json": baseline_metrics_sha256,
        **{"implementation/"+n: sha256(value).hexdigest() for n, value in extra.items()}}
    write_package_files(path, {"manifest.json": learning._bytes({"schema_version": 1, "profile": PROFILE, "status": report["status"], "files": files})})
    return report
