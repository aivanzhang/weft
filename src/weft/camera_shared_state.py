"""Strict actual-v6 snapshots with an internal validation-only metric-context view."""

from copy import deepcopy
import math

from weft import camera_shared_task as task_api
from weft.camera_shared_receipt import IDENTITIES
from weft.io import canonical_hash
from weft.shared_exercise_feet_state import _bounded_json, _prepare_legacy_metrics_validator, _same
from weft.shared_exercise_reset import describe_reset_protocol
from weft.shared_world import CONTACT_KINDS, _validate_shared_snapshot


MAX_SNAPSHOT_BYTES = 480 << 10
_METRIC_EXTRAS = frozenset({"kernel_branch", "observed_source_time_s", "metric_context_sha256",
    "kernel_evaluator_sha256", "adapter_implementation_sha256", "camera_coupling_sha256",
    "source_clock_sha256", "body_sha256"})


def _number(value, lower=-1e30, upper=1e30):
    return type(value) in (int, float) and lower <= value <= upper and math.isfinite(value)


def _validate_action_values(action, count):
    if (not isinstance(action, (list, tuple)) or len(action) != count
            or any(not _number(value, -1., 1.) for value in action)):
        raise ValueError(f"Camera actions require exactly {count} finite numeric values in [-1,1]")


def _prepare_engine_validator(actual, camera_identity):
    # The scene is already prepared; the shared validator performs no native work on v6 states.
    return lambda state: _validate_shared_snapshot(state, actual, camera_identity=camera_identity)


def validate_camera_shared_snapshot(snapshot, scene, task):
    return prepare_camera_shared_snapshot_validator(scene, task)(snapshot)


def prepare_camera_shared_snapshot_validator(scene, task):
    actual, outer = task_api.prepare_camera_shared_task(task, scene)
    return _prepare_validated_camera_shared_snapshot_validator(actual, outer)


def _prepare_validated_camera_shared_snapshot_validator(actual, outer):
    """Reuse already validated scene/task context without repeating native admission."""
    arguments = outer["metric_context"]["kernel_arguments"]
    validate_engine = _prepare_engine_validator(actual, outer["camera_coupling_sha256"])
    validate_kernel_metrics = _prepare_legacy_metrics_validator(actual, arguments)
    arguments_sha = outer["metric_context"]["kernel_arguments_sha256"]
    fixed = {"schema_version": 1, "profile": task_api.STATE_PROFILE,
             "scene_sha256": outer["scene_sha256"], "task_sha256": canonical_hash(outer),
             **{key: outer[key] for key in IDENTITIES}}
    keys = set(fixed) | {"engine_state", "task_metrics", "last_action", "engine_state_status",
                         "resolved_reset_options", "terminated", "truncated"}
    roots = outer["control"]["root_assistance"] == "explicit_3_axis"
    count = 22 if roots else 19
    substeps, dt = actual["substeps"], actual["dt"]
    maximum = outer["interval"]["horizon_steps"] * substeps
    source_start = actual["camera_coupling"]["clock"]["source_time_offset_s"]
    metric_fixed = {"profile": task_api.METRICS_PROFILE, "kernel_branch": arguments["profile"],
                    "scene_sha256": fixed["scene_sha256"], "task_sha256": fixed["task_sha256"],
                    **{key: outer[key] for key in IDENTITIES if key != "source_sha256"},
                    "source_scope": outer["scope"]}

    def validate(snapshot):
        _bounded_json(snapshot)
        if (not isinstance(snapshot, dict) or set(snapshot) != keys
                or any(not _same(snapshot[key], value) for key, value in fixed.items())):
            raise ValueError("Camera snapshot fields or actual scene/task/source/context identities differ")
        options = snapshot["resolved_reset_options"]
        if not _same(options, task_api._reset_options(options)):
            raise ValueError("Camera snapshot reset options must be resolved human-only options")
        state = snapshot["engine_state"]
        validate_engine(state)
        if (state["controlled_id"] != "human" or state["human"].get("reference", False) is not False
                or state["objects"] != [] or state["target"] is not None
                or state["camera_coupling_sha256"] != outer["camera_coupling_sha256"]):
            raise ValueError("Camera task state must retain actual human-only v6 dynamics, not reference playback")
        if (type(state.get("episode_steps")) is not int or state["episode_steps"] != state["steps"]
                or type(state.get("substeps")) is not int or state["substeps"] != substeps
                or any(row["kind"] != "human_floor" for row in state["contact_details"])
                or state["contact_summary"] != {kind: len(state["contact_details"]) if kind == "human_floor" else 0
                                               for kind in CONTACT_KINDS}):
            raise ValueError("Camera engine counters or human-floor-only contact accounting differ")
        status = snapshot["engine_state_status"]
        if status not in ("current", "last_valid_control_boundary"):
            raise ValueError("Camera geometry must distinguish its current or stale failure boundary")
        action = snapshot["last_action"]
        if not isinstance(action, list):
            raise ValueError("Camera snapshot action must be a JSON list")
        _validate_action_values(action, count)
        full = list(action) + ([0., 0., 0.] if not roots else [])
        engine_action = state["normalized_action"]
        if (len(engine_action) != 22 or not roots and any(engine_action[19:])
                or status == "current" and full != engine_action):
            raise ValueError("Camera task/engine actions differ or introduce unauthorized roots")
        metrics = snapshot["task_metrics"]
        if (not isinstance(metrics, dict) or not _METRIC_EXTRAS <= metrics.keys()
                or any(not _same(metrics.get(key), value) for key, value in metric_fixed.items())
                or not _same(metrics.get("reset_protocol"), task_api._reset_protocol(options))):
            raise ValueError("Camera metric identities, source scope or reset provenance differ")
        # Only metadata is projected for the existing validation algebra; raw state stays authoritative.
        projected = {key: value for key, value in metrics.items() if key not in _METRIC_EXTRAS}
        projected.update(profile=arguments["profile"], task_sha256=arguments_sha,
                         evaluator_sha256=outer["kernel_evaluator_sha256"],
                         reset_protocol=describe_reset_protocol(options))
        validate_kernel_metrics({**snapshot, "task_metrics": projected})
        observed, valid = metrics["observed_substeps"], metrics["valid_substeps"]
        if status == "current":
            partial = state.get("partial_action")
            physical = ((state["steps"] - 1) * substeps + partial["completed_substeps"]
                        if partial is not None else state["steps"] * substeps)
            if observed != physical:
                raise ValueError("Camera observed substeps differ from the actual engine boundary")
        elif not state["steps"] * substeps <= observed <= (state["steps"] + 1) * substeps:
            raise ValueError("Camera stale geometry may precede only its failed control action")
        source_time = metrics["observed_source_time_s"]
        if (not _number(source_time, 0., 3600.) or not math.isclose(source_time,
                source_start + metrics["simulated_duration_s"], rel_tol=0, abs_tol=1e-8)):
            raise ValueError("Camera observed source time differs from its original clock")
        for mean, peak in (("angular_rmse_rad", "peak_angular_rmse_rad"),
                           ("root_rmse_model_units", "peak_root_rmse_model_units")):
            if valid and metrics[mean] > metrics[peak] + 1e-12:
                raise ValueError("Camera mean tracking error cannot exceed its observed peak")
        failure = metrics["failure"]
        penalty = metrics["failure_reward_penalty"]
        if penalty != (-1. if failure is not None else 0.):
            raise ValueError("Camera task failures must preserve the single original failure penalty")
        effort = .01 * metrics["joint_effort_mean_squared"] + .02 * metrics["root_assistance_mean_squared"]
        tracking_integral = metrics["reward_integral"] - penalty * dt + effort * metrics["valid_observed_duration_s"]
        if not -1e-8 <= tracking_integral <= metrics["valid_observed_duration_s"] + 1e-8:
            raise ValueError("Camera reward must reflect observed tracking and explicit action effort")
        if failure == "cancelled" and (status != "current" or observed >= maximum
                or state["terminated"] or state["truncated"]):
            raise ValueError("Camera cancellation cannot relabel a completed or failed engine boundary")
        return deepcopy(snapshot)

    return validate
