"""Separate saved control-boundary evidence from opt-in, training-only substep diagnosis."""

from bisect import bisect_left
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time

from weft.io import canonical_hash, local_time
from weft.shared_exercise_lab import _json
from weft.shared_exercise_recording import prepare_snapshot_validator, validate_event, validate_recording_scene
from weft.shared_exercise_task import GATES, GATES_SHA256, HUMAN_ONLY_TASK_PROFILE
from weft.shared_world import _HUMAN_SEGMENT_BY_GEOM


PROFILE = "shared-exercise-failure-timeline-v1"
METRICS = ("displacement_model_units", "orientation_rad", "speed_model_units_s", "angular_speed_rad_s")
HUMAN_METRICS = {
    "angular_rmse_rad": "running_rms", "root_rmse_model_units": "running_rms",
    "peak_angular_rmse_rad": "cumulative_peak", "peak_root_rmse_model_units": "cumulative_peak",
    "max_floor_penetration_model_units": "cumulative_peak",
}
MAX_EVENTS = 1801
MAX_EPISODES = 64
MAX_SUMMARY_BYTES = 512 * 1024
MAX_JOURNAL_BYTES = 16 * 1024 * 1024
MAX_EVENT_BYTES = 65536
MAX_STUDY_BYTES = 32 * 1024 * 1024
RECORDING_FILES = {"scene.json", "recording.json", "events.jsonl", "video.mp4", "preview.gif", "replay.html"}


def _bytes(value):
    return (json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n").encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _deadline(seconds, maximum):
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 < seconds <= maximum:
        raise ValueError(f"Diagnosis requires a positive wall-time bound of at most {maximum} seconds")
    return time.monotonic() + seconds


def _check(deadline):
    if time.monotonic() > deadline:
        raise TimeoutError("Diagnosis exceeded its cooperative wall-time bound")


@contextmanager
def _safe_directory(directory):
    # Pin every ancestor, not just the final component; selected sessions cannot escape through links.
    path = Path(directory).absolute()
    if ".." in path.parts:
        raise ValueError("Diagnosis paths cannot traverse parent components")
    fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in path.parts[1:]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd
    except OSError as error:
        raise ValueError("Diagnosis requires a regular directory without linked ancestors") from error
    finally:
        os.close(fd)


def _read(fd, name, limit, expected=None):
    if Path(name).name != name:
        raise ValueError("Diagnosis snapshot reads require a single file name")
    try:
        child = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        with os.fdopen(child, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
                raise ValueError("Diagnosis input exceeds its regular-file byte bound")
            raw = stream.read(limit + 1)
            after = os.fstat(stream.fileno())
            if (len(raw) > limit or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) !=
                    (after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
                raise ValueError("Diagnosis input changed during bounded read")
    except OSError as error:
        raise ValueError("Diagnosis input is missing or unsafe") from error
    if expected is not None and _sha(raw) != expected:
        raise ValueError("Diagnosis input differs from its pinned SHA256")
    return raw


class _ReferenceClock:
    def __init__(self, scene):
        reference = scene["human_spec"]["reference"]
        self.times = [row["time_s"] for row in reference]
        self.origin = self.times[0] + scene["task"].get("start_offset_s", 0.)
        evidence = {row["time_s"]: row for row in scene["human_spec"].get("reference_evidence", [])}
        total_frames = scene["source_binding"].get("total_source_frames")
        self.rows = []
        previous_frame = None
        for index, stamp in enumerate(self.times):
            identity = evidence.get(stamp, {}).get("evidence_id")
            match = re.fullmatch(r"[0-9a-f]{64}:frame:(\d+)", identity or "")
            if match and (len(match[1]) > 8 or total_frames is None or not 0 <= int(match[1]) < total_frames):
                raise ValueError("Diagnosis reference frame lies outside declared source slots")
            frame = int(match[1]) if match else None
            if identity is not None and match is None or frame is not None and previous_frame is not None and frame <= previous_frame:
                raise ValueError("Diagnosis reference frame IDs must be valid and strictly increasing")
            if frame is not None:
                previous_frame = frame
            self.rows.append({"reference_index": index, "time_s": stamp, "evidence_id": identity,
                              "source_frame_index": frame})

    def at(self, elapsed):
        stamp = self.origin + elapsed
        upper = bisect_left(self.times, stamp)
        if upper < len(self.times) and math.isclose(stamp, self.times[upper], rel_tol=0, abs_tol=1e-8):
            lower = upper
        elif upper and math.isclose(stamp, self.times[upper - 1], rel_tol=0, abs_tol=1e-8):
            lower = upper = upper - 1
        else:
            lower = upper - 1
        lo = self.rows[lower] if 0 <= lower < len(self.rows) else None
        hi = self.rows[upper] if 0 <= upper < len(self.rows) else None
        a, b = (row["source_frame_index"] if row else None for row in (lo, hi))
        if a is not None and b is not None and not 0 <= b - a <= 1800:
            raise ValueError("Unsupported diagnosis reference-frame gap; maximum is 1800 slots")
        missing = list(range(a + 1, b)) if a is not None and b is not None else None
        return {"reference_clock_s": stamp, "source_frame_bracket": {
            "lower": deepcopy(lo), "upper": deepcopy(hi),
            "status": "outside_reference" if lo is None or hi is None else "exact_reference" if lo == hi else "between_references",
            "frames_without_reference_between": missing,
            "native_video_time_available": False}}


def _contacts(state, object_id):
    rows = []
    for contact in state["contact_details"]:
        if object_id not in contact["objects"]:
            continue
        row = deepcopy(contact)
        geom = next((name for name in row["objects"] if name in _HUMAN_SEGMENT_BY_GEOM), None)
        row.update(human_geom=geom, human_segment=_HUMAN_SEGMENT_BY_GEOM.get(geom))
        rows.append(row)
    if len(rows) > 64:
        raise ValueError("Diagnosis contact boundary exceeds its 64-row bound; no clipping")
    return rows


def _bounded_summary(result):
    if len(_bytes(result)) > MAX_SUMMARY_BYTES:
        raise ValueError("Diagnosis summary exceeds its 512 KiB bound; no clipping")
    return result


def _human_tracking(episode, metrics, previous, timing):
    summary = episode["human_tracking"]
    seen = {row["metric"] for row in summary["first_crossings"]}
    crossings, terminal = [], []
    old = previous["task_metrics"] if previous else {}
    for name, aggregation in HUMAN_METRICS.items():
        value, before, threshold = metrics[name], old.get(name), GATES[name]
        if aggregation == "cumulative_peak" and before is not None and value < before:
            raise ValueError("Diagnosis human maxima decreased without a reset")
        status = "unavailable" if value is None else "above_limit" if value > threshold + 1e-12 else "within_limit"
        terminal.append({"metric": name, "value": value, "threshold": threshold,
                         "status": status, "aggregation": aggregation})
        if status != "above_limit" or name in seen:
            continue
        # Running RMS can recover; its first observed value has no preceding safe measurement.
        crossing = {**timing, "metric": name, "value": value, "previous_value": before,
                    "threshold": threshold, "aggregation": aggregation,
                    "previous_safe_elapsed_s": timing["previous_safe_elapsed_s"] if before is not None else None,
                    "status": "failed_at_reset" if previous is None else
                              "first_observed_boundary" if before is None else "first_reported_boundary"}
        crossings.append(crossing)
        summary["first_crossings"].append(crossing)
    if crossings and summary["first_failure"] is None:
        summary["first_failure"] = {**timing, "episode_index": episode["episode_index"],
            "reset_event_index": episode["reset_event_index"], "crossings": crossings,
            "previous_safe_elapsed_s": timing["previous_safe_elapsed_s"]
                if all(row["previous_value"] is not None for row in crossings) else None}
    summary["terminal"] = terminal


def recording_failure_timeline(recording_scene, events, *, max_seconds=5):
    """Validate and reduce saved boundaries without decoding media or stepping physics.

    Peaks are cumulative; running RMS can recover. Saved boundaries cannot locate
    a crossing's exact substep or establish its physical cause.
    """
    deadline = _deadline(max_seconds, 10)
    if not isinstance(events, (list, tuple)) or not 1 <= len(events) <= MAX_EVENTS:
        raise ValueError("Diagnosis requires 1–1801 complete recording events")
    if len(_bytes(recording_scene)) > 4 * 1024 * 1024:
        raise ValueError("Diagnosis scene exceeds its 4 MiB bound")
    envelope = validate_recording_scene(recording_scene)
    scene, task = envelope["engine_scene"], envelope["exercise_task"]
    if len(task["equipment_ids"]) > 12:
        raise ValueError("Diagnosis supports at most 12 equipment objects")
    validate, clock = prepare_snapshot_validator(envelope), _ReferenceClock(scene)
    episodes, episode, previous, seen, evaluator = [], None, None, set(), None
    total_event_bytes = 0
    for index, event in enumerate(events):
        _check(deadline)
        size = len(_bytes(event))
        total_event_bytes += size
        if size > MAX_EVENT_BYTES or total_event_bytes > MAX_JOURNAL_BYTES:
            raise ValueError("Diagnosis event/journal byte bound exceeded")
        if (not isinstance(event, dict) or type(event.get("index")) is not int or event["index"] != index
                or event.get("type") not in {"start", "reset", "step"}
                or (index == 0) != (event["type"] == "start")):
            raise ValueError("Diagnosis requires one initial start and contiguous event indices")
        state = event.get("snapshot")
        validate(state)
        validate_event(event, envelope)
        metrics, engine = state["task_metrics"], state["engine_state"]
        if evaluator is not None and evaluator != metrics["evaluator_sha256"]:
            raise ValueError("Diagnosis evaluator changed inside the saved journal")
        evaluator = metrics["evaluator_sha256"]
        if event["type"] != "step":
            if len(episodes) >= MAX_EPISODES:
                raise ValueError("Diagnosis exceeds its 64-episode bound")
            episode = {"episode_index": len(episodes), "reset_event_index": index, "seed": event["seed"],
                       "controller_counts": {}, "first_equipment_failure": None, "first_crossings": [],
                       "human_tracking": {"profile": "shared-exercise-human-tracking-timeline-v1",
                                          "first_failure": None, "first_crossings": [], "terminal": []},
                       "resolved_reset_options": deepcopy(state["resolved_reset_options"])}
            episodes.append(episode)
            previous, seen, control_step = None, set(), 0
        else:
            if previous["terminated"] or previous["truncated"]:
                raise ValueError("Diagnosis cannot advance a terminal episode without a reset")
            old = previous["task_metrics"]
            if (metrics["observed_substeps"] < old["observed_substeps"]
                    or metrics["observed_substeps"] - old["observed_substeps"] > scene["substeps"]
                    or metrics["simulated_duration_s"] < old["simulated_duration_s"]
                    or state["resolved_reset_options"] != previous["resolved_reset_options"]):
                raise ValueError("Diagnosis episode clocks or reset identity are inconsistent")
            if state["engine_state_status"] == "current" and engine["episode_steps"] != control_step + 1:
                raise ValueError("Diagnosis episode is missing a control boundary")
            control_step += 1
            name = event.get("controller", "unspecified")
            episode["controller_counts"][name] = episode["controller_counts"].get(name, 0) + 1
        elapsed = metrics["simulated_duration_s"]
        timing = {"event_index": index, "control_step": control_step,
                  "previous_safe_elapsed_s": previous["task_metrics"]["simulated_duration_s"] if previous else None,
                  "first_failed_elapsed_s": elapsed, "last_valid_geometry_elapsed_s": engine["time_s"],
                  "engine_state_status": state["engine_state_status"], "clock_status": metrics["clock_status"],
                  **clock.at(elapsed)}
        _human_tracking(episode, metrics, previous, timing)
        old_equipment = {row["object_id"]: row for row in previous["task_metrics"]["equipment"]} if previous else {}
        crossings = []
        for row in metrics["equipment"]:
            for metric in METRICS:
                key, gate = (row["object_id"], metric), "equipment_" + metric
                old_value = old_equipment.get(row["object_id"], {}).get(metric)
                if old_value is not None and row[metric] < old_value:
                    raise ValueError("Diagnosis equipment maxima decreased without a reset")
                if row[metric] > GATES[gate] + 1e-12 and key not in seen:
                    seen.add(key)
                    crossing = {**timing, "object_id": row["object_id"], "metric": metric,
                                "gate_key": gate, "threshold": GATES[gate], "value": row[metric],
                                "previous_maximum": old_value,
                                "status": "failed_at_reset" if previous is None else "first_reported_boundary",
                                "value_scope": "cumulative_peak_not_boundary_instantaneous_value",
                                "contact_scope": "post_step_boundary_not_substep_history",
                                "boundary_contacts": _contacts(engine, row["object_id"])}
                    episode["first_crossings"].append(crossing)
                    crossings.append(crossing)
        if crossings and episode["first_equipment_failure"] is None:
            episode["first_equipment_failure"] = {**timing, "episode_index": episode["episode_index"],
                                                  "reset_event_index": episode["reset_event_index"],
                                                  "crossings": crossings}
        episode["terminal"] = {"event_index": index, "elapsed_s": elapsed, "control_steps": control_step,
                               "completed_interval": metrics["completed_interval"], "success": metrics["success"],
                               "failed_gates": deepcopy(metrics["failed_gates"]), "failure": metrics["failure"],
                               "clock_status": metrics["clock_status"], "engine_state_status": state["engine_state_status"],
                               "last_valid_geometry_elapsed_s": engine["time_s"],
                               "equipment": deepcopy(metrics["equipment"])}
        previous = state
    _check(deadline)
    return _bounded_summary({"schema_version": 1, "profile": PROFILE,
        "scene_sha256": canonical_hash(scene), "task_sha256": canonical_hash(task),
        "recording_scene_sha256": canonical_hash(envelope), "gates_sha256": GATES_SHA256,
        "evaluator_sha256": evaluator, "policy_sha256": envelope.get("learned_policy", {}).get("sha256"),
        "source_identity": deepcopy(scene["source_binding"]), "source_scope": deepcopy(task["source_scope"]),
        "source_clock": {"kind": "declared_human_reference_clock", "native_video_time_available": False,
                         "alignment_method": scene["coordinate_frame"]["clock_binding"]["alignment_method"],
                         "reference_start_s": clock.origin},
        "events": len(events), "episodes": episodes,
        "equipment_stability_applicability": "not_applicable" if task["profile"] == HUMAN_ONLY_TASK_PROFILE else "applied",
        "first_equipment_failure": next((row["first_equipment_failure"] for row in episodes
                                          if row["first_equipment_failure"] is not None), None),
        "scope": "Saved simulated peaks bracket first reported limit crossings; running RMS can recover. Unavailable values are not crossings. Body errors aggregate joints, not identify a failing joint. Boundary contacts are co-occurrence, not causal substep evidence. Reference frame times are declared priors, not native-video synchronization or physical measurements."})


def _recording_inputs(directory, deadline):
    limits = {"scene.json": 4 * 1024 * 1024, "recording.json": 65536, "events.jsonl": MAX_JOURNAL_BYTES}
    with _safe_directory(directory) as fd:
        manifest_raw = _read(fd, "manifest.json", 65536)
        manifest = _json(manifest_raw)
        if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
                or manifest.get("status") not in (None, "closed")
                or not isinstance(manifest.get("files"), dict)
                or set(manifest.get("files", {})) != RECORDING_FILES
                or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                       for value in manifest["files"].values())):
            raise ValueError("Diagnosis requires a complete sealed recording export manifest")
        files = {name: _read(fd, name, limit, manifest["files"][name]) for name, limit in limits.items()}
        _check(deadline)
    return manifest_raw, manifest, files


def inspect_recording_failure(recording_dir, *, max_seconds=10):
    """Inspect one closed, sealed recording; media hashes and physics replay are outside this scope."""
    deadline = _deadline(max_seconds, 30)
    manifest_raw, manifest, files = _recording_inputs(recording_dir, deadline)
    metadata, scene = _json(files["recording.json"]), _json(files["scene.json"])
    lines = files["events.jsonl"].splitlines(keepends=True)
    if not 1 <= len(lines) <= MAX_EVENTS or any(len(line) > MAX_EVENT_BYTES or not line.endswith(b"\n") for line in lines):
        raise ValueError("Diagnosis journal has excessive or incomplete events")
    events = [_json(line) for line in lines]
    if any(not isinstance(row, dict) for row in events):
        raise ValueError("Diagnosis journal events must be JSON objects")
    if (not isinstance(metadata, dict) or metadata.get("status") != "closed" or metadata.get("profile") != "shared-exercise"
            or metadata.get("schema_version") != 1 or metadata.get("scene_sha256") != canonical_hash(scene)
            or metadata.get("events") != len(events) or metadata.get("journal_bytes") != len(files["events.jsonl"])
            or manifest.get("video_frames") != len(events) or metadata.get("seed") != events[0].get("seed")):
        raise ValueError("Diagnosis requires matching closed recording metadata, scene, seed and complete journal")
    _check(deadline)
    result = recording_failure_timeline(scene, events, max_seconds=min(10, deadline - time.monotonic()))
    # Reopen the named directory to detect replacement as well as mutation of pinned handles.
    again, _, checked = _recording_inputs(recording_dir, deadline)
    if again != manifest_raw or checked != files:
        raise ValueError("Diagnosis recording changed during inspection")
    result["integrity"] = {"manifest_sha256": _sha(manifest_raw),
        "required_files_sha256": {name: _sha(raw) for name, raw in files.items()},
        "verified_scope": "scene_task_policy_journal_metadata_only", "media_verified": False,
        "action_replay_performed": False}
    _check(deadline)
    return _bounded_summary(result)


class _StudyOutput:
    def __init__(self, directory):
        from weft.shared_world_packaging import write_package_files

        self.root, self.used, self.hashes = Path(directory), 0, {}
        self.root.mkdir(mode=0o700, parents=False, exist_ok=False)
        self._write = write_package_files
        self.trace = None
        self.trace_hash = hashlib.sha256()

    def write(self, name, raw, *, terminal=False):
        limit = MAX_STUDY_BYTES if terminal else MAX_STUDY_BYTES - 128 * 1024
        if self.used + len(raw) > limit:
            raise ValueError("Diagnosis output would exceed 32 MiB; refused before write")
        self._write(self.root, {name: raw})
        self.used += len(raw)
        self.hashes[name] = _sha(raw)

    def append(self, raw):
        if self.used + len(raw) > MAX_STUDY_BYTES - 128 * 1024:
            raise ValueError("Diagnosis substep evidence would exceed 32 MiB; refused before write")
        if self.trace is None:
            fd = os.open(self.root / "substeps.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            self.trace = os.fdopen(fd, "wb")
        self.trace.write(raw)
        self.trace_hash.update(raw)
        self.used += len(raw)

    def seal(self, status):
        if self.trace is not None:
            self.trace.close()
            self.trace = None
            self.hashes["substeps.jsonl"] = self.trace_hash.hexdigest()
        self.write("manifest.json", _bytes({"schema_version": 1, "created_at": local_time(),
                   "status": status, "files": dict(self.hashes)}), terminal=True)


class _SubstepDiagnosis:
    """Observe detached telemetry after the unchanged scorer has consumed the same sample."""

    def __init__(self, scene, task, emit, deadline):
        self.scene, self.task, self.emit, self.deadline = scene, task, emit, deadline
        self.scale = scene["coordinate_frame"]["scale_bridge"]["body_height_to_world_scale"]
        self.clock = _ReferenceClock(scene)
        self.initial, self.maximum, self.previous = {}, {}, {}
        self.crossings, self.seen, self.contacts, self.first = [], set(), {}, None
        self.samples = self.contact_rows = 0
        self.trace_bytes = 0

    def reset(self, initial):
        if self.initial:
            raise ValueError("Diagnosis permits exactly one original training reset")
        self.initial = dict(initial)
        for name, item in self.initial.items():
            values = self.values(item)
            self.maximum[name] = dict(values)
            self.previous[name] = dict(values)
        if any(value > GATES["equipment_" + key] + 1e-12 for row in self.maximum.values() for key, value in row.items()):
            raise ValueError("Substep diagnosis requires equipment stable at reset; use the saved reset timeline for initial failures")

    def values(self, current):
        import numpy as np
        from weft.shared_exercise_task import _quat_angle

        initial = self.initial[current.object_id]
        return {"displacement_model_units": math.dist(current.position, initial.position) / self.scale,
                "orientation_rad": _quat_angle(current.quaternion, initial.quaternion),
                "speed_model_units_s": float(np.linalg.norm(current.velocity)) / self.scale,
                "angular_speed_rad_s": float(np.linalg.norm(current.angular_velocity))}

    def consume(self, sample):
        _check(self.deadline)
        if self.samples >= self.task["interval"]["horizon_steps"] * self.scene["substeps"]:
            raise ValueError("Diagnosis exceeded its frozen substep bound")
        self.samples += 1
        self.contact_rows += len(sample.contacts)
        if self.contact_rows > 10000:
            raise ValueError("Diagnosis exceeds its 10000-contact-row bound")
        contacts = []
        duration = (sample.interval_end_s - sample.interval_start_s
                    if sample.valid and sample.interval_end_s is not None else None)
        for contact in sample.contacts:
            row = asdict(contact)
            row["force_on_object_world"] = [-value for value in contact.force_on_human_world]
            row["impulse_on_object_world"] = ([value * duration for value in row["force_on_object_world"]]
                                               if duration is not None else None)
            contacts.append(row)
            if duration is None or contact.normal_force <= 0:
                continue
            key = (contact.object_id, contact.human_geom)
            if key not in self.contacts:
                self.contacts[key] = {"object_id": contact.object_id, "human_geom": contact.human_geom,
                    "human_segment": contact.human_segment, "first_solver_contact_time_s": sample.contact_time_s,
                    **self.clock.at(sample.contact_time_s), "last_solver_contact_time_s": sample.contact_time_s,
                    "contact_rows": 0, "peak_normal_force_prior_units": 0.,
                    "impulse_on_object_world": [0., 0., 0.], "impulse_before_or_at_first_overall_gate_world": [0., 0., 0.]}
            summary = self.contacts[key]
            summary["last_solver_contact_time_s"] = sample.contact_time_s
            summary["contact_rows"] += 1
            summary["peak_normal_force_prior_units"] = max(summary["peak_normal_force_prior_units"], contact.normal_force)
            for index, value in enumerate(row["impulse_on_object_world"]):
                summary["impulse_on_object_world"][index] += value
                if self.first is None:
                    summary["impulse_before_or_at_first_overall_gate_world"][index] += value
        rows, crossings = [], []
        if sample.valid:
            if duration is None or duration <= 0 or set(self.initial) - {row.object_id for row in sample.equipment}:
                raise ValueError("Diagnosis valid substep lacks elapsed time or complete equipment")
            for current in sample.equipment:
                if current.object_id not in self.initial:
                    continue
                values = self.values(current)
                maximum = self.maximum[current.object_id]
                for key, value in values.items():
                    previous_maximum = maximum[key]
                    maximum[key] = max(maximum[key], value)
                    pair = (current.object_id, key)
                    if value > GATES["equipment_" + key] + 1e-12 and pair not in self.seen:
                        self.seen.add(pair)
                        crossing = {"object_id": current.object_id, "metric": key, "gate_key": "equipment_" + key,
                            "threshold": GATES["equipment_" + key], "value": value,
                            "previous_maximum": previous_maximum, "previous_instantaneous": self.previous[current.object_id][key],
                            "control_step": sample.step_index, "substep_index": sample.substep_index,
                            "previous_safe_elapsed_s": sample.interval_start_s,
                            "first_failed_elapsed_s": sample.interval_end_s,
                            "solver_contact_time_s": sample.contact_time_s, **self.clock.at(sample.interval_end_s),
                            "dynamic_contacts": [row for row in contacts if row["object_id"] == current.object_id]}
                        crossings.append(crossing)
                        self.crossings.append(crossing)
                rows.append({"object_id": current.object_id, "instantaneous": values, "maximum": dict(maximum),
                             "position": current.position, "velocity": current.velocity,
                             "angular_velocity": current.angular_velocity})
                self.previous[current.object_id] = values
        if crossings and self.first is None:
            self.first = {"control_step": sample.step_index, "substep_index": sample.substep_index,
                          "previous_safe_elapsed_s": sample.interval_start_s,
                          "first_failed_elapsed_s": sample.interval_end_s, "crossings": crossings}
        raw = _bytes({"sample_index": self.samples - 1, "control_step": sample.step_index,
            "substep_index": sample.substep_index, "interval_start_s": sample.interval_start_s,
            "interval_end_s": sample.interval_end_s, "solver_contact_time_s": sample.contact_time_s,
            "valid": sample.valid, "failure": sample.failure, "equipment": rows, "dynamic_contacts": contacts})
        if self.trace_bytes + len(raw) > 16 * 1024 * 1024:
            raise ValueError("Diagnosis substep journal exceeds its 16 MiB bound")
        self.emit(raw)
        self.trace_bytes += len(raw)

    def summary(self):
        return {"observed_substeps": self.samples, "contact_rows": self.contact_rows,
            "first_equipment_failure": self.first, "first_crossings": self.crossings,
            "human_object_contact_pairs": list(self.contacts.values()),
            "equipment_maxima": deepcopy(self.maximum),
            "scope": "Dynamic simulator solver forces at interval start, integrated equipment state at interval end. Forces and impulses use authored mass/scale priors. Gravity, floor and object-object reactions are not included in this human-contact telemetry; no isolated causal or physical measurement claim."}


def _selected_training_trial(policy, training, settings):
    if (training.get("status") != "trained" or not isinstance(training.get("history"), list)
            or not settings.get("training_seeds") or not settings.get("training_reset_options")):
        raise ValueError("Diagnosis requires sealed original training history and starts")
    matches = []
    for generation, batch in enumerate(training["history"]):
        for candidate, row in enumerate(batch["candidates"]):
            if row.get("residual_coefficients") == policy["residual_coefficients"] and row.get("cached") is False:
                matches.append((generation, candidate, row))
    if not matches:
        raise ValueError("Diagnosis policy has no original evaluated training candidate")
    generation, candidate, row = matches[0]
    trial = row["trials"][0]
    if (not row.get("complete") or row.get("score") != policy["training_score"]
            or trial.get("seed") != settings["training_seeds"][0]
            or trial.get("reset_options") != settings["training_reset_options"][0]
            or trial.get("status") != "completed" or trial.get("control_steps") != settings["horizon_steps"]
            or not 1 <= trial["control_steps"] <= 254):
        raise ValueError("Diagnosis must replay the first original completed training trial, at most 254 controls")
    return {"generation": generation, "candidate": candidate, "trial_index": 0}, trial


def _reference_inputs(directory, scene, task, deadline):
    from weft.shared_world import prepare_shared_snapshot_validator

    required = {"scene.json": 4 * 1024 * 1024, "exercise-task.json": 65536,
                "exercise-lab.json": 512 * 1024, "human-reference.json": 8 * 1024 * 1024}
    with _safe_directory(directory) as fd:
        manifest_raw = _read(fd, "manifest.json", 1024 * 1024)
        manifest = _json(manifest_raw)
        hashes = manifest.get("files") if isinstance(manifest, dict) else None
        if (not isinstance(manifest, dict) or manifest.get("status") != "ready" or manifest.get("control_profile") != "shared_exercise"
                or not isinstance(hashes, dict) or not set(required) <= set(hashes)):
            raise ValueError("Diagnosis reference requires a sealed ready exercise lab")
        files = {name: _read(fd, name, limit, hashes[name]) for name, limit in required.items()}
    _check(deadline)
    contract = _json(files["exercise-lab.json"])
    if (not isinstance(contract, dict) or _json(files["scene.json"]) != scene or _json(files["exercise-task.json"]) != task
            or contract.get("scene_sha256") != canonical_hash(scene) or contract.get("task_sha256") != canonical_hash(task)):
        raise ValueError("Diagnosis kinematic reference must share the exact scene and task")
    data = _json(files["human-reference.json"])
    references = scene["human_spec"]["reference"]
    frames = data.get("frames") if isinstance(data, dict) else None
    if (not isinstance(frames, list) or not 1 <= len(frames) <= 900 or len(frames) != len(references)
            or data.get("schema_version") != 1 or data.get("profile") != "shared-world-kinematic-reference-v1"
            or data.get("reference_poses_total") != len(frames) or data.get("reference_poses_included") != len(frames)):
        raise ValueError("Diagnosis kinematic reference coverage differs from all declared poses")
    clock, rows = _ReferenceClock(scene), []
    validate = prepare_shared_snapshot_validator(scene)
    nominal = {row["id"]: row for row in scene["objects"]}
    scale = scene["coordinate_frame"]["scale_bridge"]["body_height_to_world_scale"]
    for index, (frame, reference) in enumerate(zip(frames, references)):
        _check(deadline)
        validate(frame)
        if (not isinstance(frame, dict) or frame.get("substeps") != 0 or frame.get("episode_steps") != 0
                or frame.get("human", {}).get("reference") is not True or frame.get("source_time_s") != reference["time_s"]
                or frame.get("coordinate_frame") != scene["coordinate_frame"]
                or not isinstance(frame.get("contact_details"), list) or len(frame["contact_details"]) > 128):
            raise ValueError("Diagnosis reference frame is not an intact zero-physics reference snapshot")
        if any(math.dist(row["position"], nominal[row["id"]]["position"]) > 1e-9
               or abs(sum(a * b for a, b in zip(row["quaternion"], nominal[row["id"]]["quaternion"]))) < 1 - 1e-9
               for row in frame["objects"]):
            raise ValueError("Diagnosis reference equipment must retain its nominal initial pose")
        contacts = []
        for contact in frame["contact_details"]:
            if contact.get("kind") != "human_object":
                continue
            pair, distance = contact.get("objects"), contact.get("distance")
            if (not isinstance(pair, list) or len(pair) != 2 or any(not isinstance(value, str) for value in pair)
                    or type(distance) not in (int, float) or not math.isfinite(distance) or abs(distance) > 1e6):
                raise ValueError("Diagnosis reference overlap requires finite distance and known collision shapes")
            geom = next((name for name in pair if name in _HUMAN_SEGMENT_BY_GEOM), None)
            equipment = next((name for name in pair if name in task["equipment_ids"]), None)
            if geom is None or equipment is None:
                raise ValueError("Diagnosis reference overlap is not bound to this body and equipment")
            if distance < 0:
                # mj_forward may populate solver force fields, but these are not integrated dynamic evidence.
                contacts.append({"object_id": equipment, "human_geom": geom,
                    "human_segment": _HUMAN_SEGMENT_BY_GEOM[geom], "distance_world": distance,
                    "penetration_model_units": -distance / scale})
        rows.append({"reference_index": index, "reference_clock_s": reference["time_s"],
                     "source_reference": clock.rows[index], "original_frame_sha256": canonical_hash(frame),
                     "overlaps": contacts})
    summary = {"profile": "shared-exercise-saved-kinematic-overlaps-v1", "rows": rows,
        "scope": "Saved zero-physics pose overlaps against initial equipment. No dynamic contact, impulse, support or measured force is inferred; source solver-force fields are deliberately excluded."}
    if len(_bytes(summary)) > 512 * 1024:
        raise ValueError("Diagnosis reference-overlap summary exceeds 512 KiB; no clipping")
    identity = {"directory": str(Path(directory).absolute()), "manifest_sha256": _sha(manifest_raw),
                "required_files_sha256": {name: _sha(raw) for name, raw in files.items()},
                "verified_scope": "sealed_scene_task_lab_contract_and_zero_physics_reference_only"}
    return manifest_raw, files, identity, summary


def _diagnosis_html(result):
    from html import escape

    first = result["training_substeps"]["first_equipment_failure"]
    timing = (f"First failure: {first['previous_safe_elapsed_s']:.6f}–{first['first_failed_elapsed_s']:.6f} simulated seconds."
              if first else "No equipment gate crossed in this original training trial.")
    rows = "".join("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in
        (row["object_id"], row["metric"], f"{row['first_failed_elapsed_s']:.6f}",
         f"{row['value']:.6f}", row["threshold"],
         ", ".join(sorted({contact["human_segment"] for contact in row["dynamic_contacts"]})))) + "</tr>"
        for row in result["training_substeps"]["first_crossings"])
    return ("<!doctype html><html lang='en'><meta charset='utf-8'><title>Weft equipment failure timeline</title>"
        "<style>body{font:17px system-ui;max-width:1100px;margin:40px auto;padding:20px}td,th{padding:10px;text-align:left;border-bottom:1px solid #ccc}code{overflow-wrap:anywhere}</style>"
        "<h1>Equipment failure timeline</h1><p>One unchanged, original training trial—not a new evaluation or a successful repair.</p>"
        f"<p>{escape(timing)}</p><p>Exact saved training terminal state and metrics matched. "
        "Contact forces precede the integrated crossing state. Contact here means simulated collision, not verified grasp or muscle measurement.</p>"
        "<table><thead><tr><th>Object</th><th>First failed metric</th><th>Task seconds</th><th>Value</th><th>Limit</th><th>Contacting body segment</th></tr></thead><tbody>"
        + rows + "</tbody></table><p>Source/reference clocks are frame/FPS priors, not native-video synchronization. "
        "Missing reference frames remain missing. The JSON report also separates every saved recording episode.</p>"
        f"<p>Policy: <code>{escape(result['policy_sha256'])}</code></p>"
        "<p><a href='diagnosis.json'>Complete diagnosis and provenance</a> · <a href='recording-timeline.json'>Saved recording boundary timeline</a> · <a href='substeps.jsonl'>Every diagnosed substep</a></p></html>"
        ).encode()


def diagnose_shared_exercise_failure(recording_dir, learning_run, out, *, reference_run=None, max_seconds=30):
    """Run at most one selected-policy ORIGINAL TRAINING trial; never consume evaluation trials."""
    from weft.shared_exercise_lab import _validate_policy_package, _verified_inputs
    from weft.shared_exercise_residual_learning import _Budget, _runtime_hashes, _trial
    from weft.shared_exercise_task import SharedExerciseTaskEnv
    from weft.shared_world_packaging import read_package_file, read_package_manifest

    started = time.monotonic()
    deadline = _deadline(max_seconds, 30)
    out, recording_dir, learning_run = Path(out).absolute(), Path(recording_dir).absolute(), Path(learning_run).absolute()
    if any(out == source or out.is_relative_to(source) or source.is_relative_to(out)
           for source in (recording_dir, learning_run)):
        raise ValueError("Diagnosis output must be separate from immutable inputs")
    with _safe_directory(learning_run):
        manifest_raw, files = _verified_inputs(learning_run, deadline=deadline)
    scene, task = _json(files["scene.json"]), _json(files["exercise-task.json"])
    policy, identity = _validate_policy_package(manifest_raw, files, scene, task)
    if policy.get("algorithm") != "shared_exercise_phase_residual_cem_v1":
        raise ValueError("Substep diagnosis supports the frozen phase-residual training policy only")
    settings, training = _json(files["settings.json"]), _json(files["training.json"])
    candidate, expected = _selected_training_trial(policy, training, settings)
    timeline = inspect_recording_failure(recording_dir, max_seconds=min(10, deadline - time.monotonic()))
    if any(timeline[key] != identity["policy_sha256"] if key == "policy_sha256" else timeline[key] != policy[key]
           for key in ("scene_sha256", "task_sha256", "policy_sha256")):
        raise ValueError("Diagnosis recording and selected training policy must share exact scene, task and policy")
    recording_manifest, _, recording_files = _recording_inputs(recording_dir, deadline)
    if (_sha(recording_manifest) != timeline["integrity"]["manifest_sha256"]
            or {name: _sha(raw) for name, raw in recording_files.items()} != timeline["integrity"]["required_files_sha256"]):
        raise ValueError("Diagnosis recording changed after its boundary timeline was inspected")
    reference_manifest, reference_files, reference_identity, reference_summary = None, {}, None, None
    if reference_run is not None:
        reference_run = Path(reference_run).absolute()
        if out == reference_run or out.is_relative_to(reference_run) or reference_run.is_relative_to(out):
            raise ValueError("Diagnosis output must be separate from the saved reference lab")
        reference_manifest, reference_files, reference_identity, reference_summary = _reference_inputs(
            reference_run, scene, task, deadline)
    implementation = Path(__file__).read_bytes()
    hashes = _runtime_hashes()
    project = Path(__file__).parents[2]
    project_pins = {name: read_package_file(project, name, max_bytes=1024 * 1024)
                    for name in ("pyproject.toml", "requirements.lock", "requirements-lab.lock")}
    if any(raw != files["implementation/project/" + name] for name, raw in project_pins.items()):
        raise ValueError("Diagnosis project configuration differs from the original training runtime")
    plan = {"schema_version": 1, "profile": "shared-exercise-contact-diagnosis-v1", "created_at": local_time(),
        **identity, "recording_manifest_sha256": _sha(recording_manifest),
        "scene_sha256": policy["scene_sha256"], "task_sha256": policy["task_sha256"], "gates_sha256": GATES_SHA256,
        "runtime_sha256": hashes, "diagnostics_sha256": _sha(implementation),
        "runtime_versions": policy["runtime_versions"], "project_sha256": {name: _sha(raw) for name, raw in project_pins.items()},
        "original_learning_path": str(learning_run), "original_recording_path": str(recording_dir),
        "source_manifest_sha256": _sha(files["input-manifest.json"]), "source_binding": scene["source_binding"],
        "source_proof_reference": {"learning_manifest_sha256": _sha(manifest_raw), "path": "source-verification.json",
                                   "sha256": _sha(files["source-verification.json"])},
        "saved_kinematic_reference": reference_identity,
        "selected_training_candidate": candidate, "seed": expected["seed"], "reset_options": expected["reset_options"],
        "expected_initial_state_sha256": expected["initial_state_sha256"],
        "expected_final_snapshot_sha256": expected["final_snapshot_sha256"],
        "maximum_training_rollouts": 1, "maximum_control_steps": expected["control_steps"],
        "maximum_physics_substeps": expected["control_steps"] * scene["substeps"], "maximum_contact_rows": 10000,
        "maximum_output_bytes": MAX_STUDY_BYTES, "max_seconds": max_seconds, "external_api_usd": 0,
        "evaluation_used": False, "source_video_decodes": 0, "policy_changes": 0, "scorer_changes": 0,
        "unanswered_question": "Which original training substep first crossed each stability gate, and which human collision shapes exerted force before/at that crossing? Saved training stores terminal metrics only; recording contacts are control-boundary samples.",
        "scope": "Single diagnostic repetition of a previously completed training trial; not optimization, evaluation, repair admission or physical validation."}
    copies = {"plan.json": _bytes(plan), "implementation/shared_exercise_diagnostics.py": implementation,
        "recording/manifest.json": recording_manifest, "learning/manifest.json": manifest_raw,
        "recording-timeline.json": _bytes(timeline),
        **{"recording/" + name: raw for name, raw in recording_files.items()},
        **{"learning/" + name: files[name] for name in ("scene.json", "exercise-task.json", "policy.json", "training.json",
            "settings.json", "input-manifest.json", "source-verification.json")},
        **{"implementation/" + name: files["implementation/" + name] for name in hashes}}
    copies.update({"implementation/project/" + name: raw for name, raw in project_pins.items()})
    if reference_manifest is not None:
        copies.update({"reference/manifest.json": reference_manifest,
                       "reference/exercise-lab.json": reference_files["exercise-lab.json"],
                       "reference-overlaps.json": _bytes(reference_summary)})
    if sum(map(len, copies.values())) > 12 * 1024 * 1024:
        raise ValueError("Diagnosis pinned input copies exceed 12 MiB; no physics started")
    with _safe_directory(out.parent):
        output = _StudyOutput(out)
    diagnostic, env, actual = None, None, None
    try:
        for name, raw in copies.items():
            _check(deadline)
            output.write(name, raw)
        diagnostic = _SubstepDiagnosis(scene, task, output.append, deadline)

        class ObservedTrainingEnv(SharedExerciseTaskEnv):
            def reset(self, **kwargs):
                result = super().reset(**kwargs)
                diagnostic.reset(self._accumulator.initial)
                return result

            def _observe_substep(self, sample):
                super()._observe_substep(sample)
                diagnostic.consume(sample)

        _check(deadline)
        env = ObservedTrainingEnv(scene, task)
        budget = _Budget(expected["control_steps"])
        actual = _trial(env, "learned", expected["seed"], expected["reset_options"],
                        policy["residual_coefficients"], deadline, budget)
        if actual != expected or budget.used != expected["control_steps"]:
            raise ValueError("Diagnosed trial differs from exact saved training initial/final state, return, metrics or termination")
        if _runtime_hashes() != hashes or Path(__file__).read_bytes() != implementation:
            raise ValueError("Diagnosis runtime changed during the original training replay")
        for name, raw in project_pins.items():
            read_package_file(project, name, max_bytes=1024 * 1024, expected_sha256=_sha(raw))
        if read_package_manifest(learning_run)[0] != manifest_raw:
            raise ValueError("Diagnosis learning manifest changed during the study")
        for name in ("scene.json", "exercise-task.json", "policy.json", "training.json", "settings.json",
                     "input-manifest.json", "source-verification.json"):
            _check(deadline)
            read_package_file(learning_run, name, max_bytes=max(1, len(files[name])), expected_sha256=_sha(files[name]))
        checked, _, checked_files = _recording_inputs(recording_dir, deadline)
        if checked != recording_manifest or checked_files != recording_files:
            raise ValueError("Diagnosis recording changed during the study")
        if reference_manifest is not None:
            with _safe_directory(reference_run) as fd:
                _read(fd, "manifest.json", 1024 * 1024, _sha(reference_manifest))
                for name, raw in reference_files.items():
                    _check(deadline)
                    _read(fd, name, max(1, len(raw)), _sha(raw))
        substeps = diagnostic.summary()
        if substeps["equipment_maxima"] != {row["object_id"]: {key: row[key] for key in METRICS}
                                              for row in actual["metrics"]["equipment"]}:
            raise ValueError("Independent diagnosis maxima differ from the unchanged task scorer")
        result = {**plan, "status": "diagnosed", "exact_saved_training_match": True,
                  "actual_training_trial": actual, "training_substeps": substeps,
                  "recording_boundary_timeline": "recording-timeline.json", "seconds": time.monotonic() - started,
                  "actual_control_steps": budget.used, "actual_physics_substeps": diagnostic.samples,
                  "required_inputs_unchanged": True,
                  "input_recheck_scope": "Recording manifest/scene/metadata/journal; learning manifest/scene/task/policy/history/settings/source-proof; runtime/project pins; optional lab manifest/scene/task/contract/reference. Original source archive verified at entry, not decoded or reread after replay.",
                  "reference_overlap_artifact": "reference-overlaps.json" if reference_summary is not None else None}
        output.write("diagnosis.json", _bytes(result))
        output.write("report.html", _diagnosis_html(result))
        _check(deadline)
        output.seal("diagnosed")
        return {"status": "diagnosed", "directory": str(out), "bytes": output.used,
                "manifest_sha256": _sha((out / "manifest.json").read_bytes()),
                "exact_saved_training_match": True, "actual_control_steps": budget.used,
                "actual_physics_substeps": diagnostic.samples, "seconds": result["seconds"]}
    except Exception as error:
        failure = {"status": "failed", "error": f"{type(error).__name__}: {error}"[:2000],
            "actual_training_trial": actual or getattr(error, "partial_trial", None),
            "observed_substeps": diagnostic.samples if diagnostic else 0,
            "contact_rows": diagnostic.contact_rows if diagnostic else 0, "no_retry_performed": True}
        if len(_bytes(failure)) > 64 * 1024:
            failure["actual_training_trial"] = {"status": "see_partial_substeps", "scope": "Failure summary byte bound reached"}
        output.write("failure.json", _bytes(failure), terminal=True)
        output.seal("failed")
        raise
    finally:
        if env is not None:
            env.close()
