"""Append-only interaction journals and portable, state-based replay exports."""

import json
import math
import os
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from weft.io import canonical_hash, digest, local_time, write_json

MAX_EVENTS = 18001
MAX_SECONDS = 1200
MAX_EVENT_BYTES = 65536
MAX_JOURNAL_BYTES = 128 * 1024 * 1024
MAX_HUMAN_PARTS = 64
MAX_HUMAN_GEOMETRY = 128
MAX_MOBILE_CONTACTS = 32
MAX_MOBILE_CONTACT_COUNT = 128
WHOLE_BODY_PROFILE = "exercise-imitation-v1"
FOOT_BODY_PROFILE = "exercise-feet-posture-v1"
DIRECT_PROFILES = {WHOLE_BODY_PROFILE: (19, "whole_body_reference_tracking"),
                   FOOT_BODY_PROFILE: (25, "starting_pose_hold")}


def _encoded(value):
    return json.dumps(value, allow_nan=False, separators=(",", ":"))


def _check_deadline(deadline):
    if deadline is not None:
        if type(deadline) not in (int, float) or not math.isfinite(deadline):
            raise ValueError("Recording deadline must be finite monotonic time")
        if time.monotonic() > deadline:
            raise TimeoutError("Recording operation exceeded its cooperative deadline; partial files are preserved")


def _seed(seed):
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Recording seed must be an integer between 0 and 2^32-1")
    return seed


def _finite_vector(value, size, name, limit=100):
    if (not isinstance(value, (list, tuple)) or len(value) != size
            or any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > limit for v in value)):
        raise ValueError(f"{name} requires {size} finite bounded coordinates")



def _mobility_snapshot(mobility):
    required = {"support_enabled", "translation", "linear_velocity", "support_force", "floor_height",
                "contacts", "contact_count", "total_normal_force", "max_penetration", "scope"}
    if not isinstance(mobility, dict) or set(mobility) not in (required, required | {"floor_normal", "floor_scope"}):
        raise ValueError("Mobile human snapshot requires the complete mobility state")
    if "floor_normal" in mobility:
        normal = mobility["floor_normal"]
        _finite_vector(normal, 3, "Mobile floor normal", limit=1)
        if normal[2] < .25 or not math.isclose(math.sqrt(sum(v*v for v in normal)), 1., abs_tol=1e-6, rel_tol=0):
            raise ValueError("Mobile floor normal must be unit length with positive camera-up component >= .25")
        if not isinstance(mobility["floor_scope"], str) or not 1 <= len(mobility["floor_scope"]) <= 1000:
            raise ValueError("Mobile floor scope must be bounded text")
    if type(mobility["support_enabled"]) is not bool:
        raise ValueError("Mobile support_enabled must be boolean")
    for field, limit in (("translation", 100), ("linear_velocity", 100), ("support_force", 2000)):
        _finite_vector(mobility[field], 3, f"Mobile {field}", limit=limit)
    floor = mobility["floor_height"]
    if type(floor) not in (float, int) or not math.isfinite(floor) or floor != 0:
        raise ValueError("Mobile floor height must preserve the authored zero plane")
    contacts, count = mobility["contacts"], mobility["contact_count"]
    if (not isinstance(contacts, list) or len(contacts) > MAX_MOBILE_CONTACTS
            or type(count) is not int or not len(contacts) <= count <= MAX_MOBILE_CONTACT_COUNT):
        raise ValueError("Mobile contacts exceed the bounded contact-count contract")
    for contact in contacts:
        if not isinstance(contact, dict) or set(contact) != {"geom", "position", "distance", "normal_force"}:
            raise ValueError("Mobile contact requires geometry, position, distance and normal force")
        if not isinstance(contact["geom"], str) or not 1 <= len(contact["geom"]) <= 100:
            raise ValueError("Mobile contact geometry must be bounded text")
        _finite_vector(contact["position"], 3, "Mobile contact position")
        _finite_vector([contact["distance"]], 1, "Mobile contact distance")
        force = contact["normal_force"]
        if type(force) not in (int, float) or not math.isfinite(force) or not 0 <= force <= 1_000_000_000:
            raise ValueError("Mobile contact normal force must be finite, nonnegative and bounded")
    for field, limit in (("total_normal_force", 1_000_000_000), ("max_penetration", 100)):
        value = mobility[field]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= limit:
            raise ValueError(f"Mobile {field} must be finite, nonnegative and bounded")
    if not isinstance(mobility["scope"], str) or not 1 <= len(mobility["scope"]) <= 1000:
        raise ValueError("Mobile scope must be bounded text")


def _whole_body_contract(scene):
    rotational, task_type = DIRECT_PROFILES[scene["profile"]]
    count = rotational + 3
    contract = scene.get("action_contract")
    names = contract.get("names") if isinstance(contract, dict) else None
    if (not isinstance(names, list) or len(names) != count
            or any(not isinstance(name, str) or not 1 <= len(name) <= 100 for name in names)
            or len(set(names)) != count):
        raise ValueError(f"Whole-body recording requires {count} unique action names")
    bounds = contract.get("normalized_bounds")
    _finite_vector(bounds, 2, "Whole-body normalized bounds", limit=1)
    if list(bounds) != [-1, 1]:
        raise ValueError("Whole-body normalized bounds must be [-1, 1]")
    limits = contract.get("physical_limits")
    _finite_vector(limits, count, "Whole-body physical limits", limit=2000)
    if any(limit <= 0 or limit > (120 if i < rotational else 2000) for i, limit in enumerate(limits)):
        raise ValueError("Whole-body physical limits require positive torques <=120 Nm and root forces <=2000 N")
    if contract.get("units") != ["Nm"] * rotational + ["N"] * 3:
        raise ValueError(f"Whole-body action units require {rotational} torques and 3 root forces")
    if type(contract.get("substeps_per_action")) is not int or contract["substeps_per_action"] != 8:
        raise ValueError("Whole-body actions require exactly 8 substeps")
    if contract.get("hidden_assistance") is not False:
        raise ValueError("Whole-body actions must explicitly disable hidden assistance")
    task = scene.get("task")
    if not isinstance(task, dict) or task.get("type") != task_type:
        raise ValueError(f"Whole-body recording requires the {task_type} task")
    if scene["profile"] == FOOT_BODY_PROFILE:
        from weft.foot_body import ANKLE_NAMES, FOOT_BODY_CONTRACT
        from weft.spatial_pose import SPATIAL_JOINT_NAMES
        if names != list(SPATIAL_JOINT_NAMES) + list(ANKLE_NAMES) + ["root_x", "root_y", "root_z"]:
            raise ValueError("Foot recordings require the exact 25 joint and three root motor identities")
        candidate = scene.get("foot_candidate")
        if not isinstance(candidate, dict) or any(candidate.get(k) != v for k, v in FOOT_BODY_CONTRACT.items()):
            raise ValueError("Foot recordings must retain the unverified candidate assumptions")
        if limits[19:25] != [FOOT_BODY_CONTRACT["motor_limit_nm"]] * 6:
            raise ValueError("Foot motor limits must match the authored candidate")
        for field in ("body_sha256", "foot_reference_sha256"):
            value = candidate.get(field)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Foot recordings require body and reference hashes")
    return names


def _action(action, scene):
    if scene.get("profile") == "shared-exercise-recording-v1":
        from weft.shared_exercise_recording import validate_action
        validate_action(action, scene)
        return
    if scene.get("profile") == "shared-world-v1":
        from weft.shared_world import validate_shared_world
        validated = validate_shared_world(scene)
        _finite_vector(action, len(validated["action_contract"]["names"]), "Shared-world recording action", limit=1)
        return
    if scene.get("profile") in DIRECT_PROFILES:
        _finite_vector(action, DIRECT_PROFILES[scene["profile"]][0] + 3, "Whole-body recording action", limit=1)
        return
    maximum = 9 if scene.get("profile") == "exercise-v3" else 7 if scene.get("profile") == "spatial-rigid-v1" else 5
    if type(action) is not int or not 0 <= action <= maximum:
        raise ValueError(f"Recording action must be an integer from 0 to {maximum}")


def _whole_body_event(event, scene):
    if scene.get("profile") == "shared-exercise-recording-v1":
        from weft.shared_exercise_recording import validate_event
        validate_event(event, scene)
        return
    if scene.get("profile") == "shared-world-v1":
        normalized = event["snapshot"]["normalized_action"]
        if event["type"] == "step":
            if list(event["action"]) != list(normalized):
                raise ValueError("Shared-world event action must match snapshot normalized_action")
        elif any(value != 0 for value in normalized):
            raise ValueError("Shared-world start/reset snapshots require zero normalized_action")
        return
    if scene.get("profile") not in DIRECT_PROFILES:
        return
    normalized = event["snapshot"]["human"]["normalized_action"]
    if event["type"] == "step":
        if list(event["action"]) != list(normalized):
            raise ValueError("Whole-body event action must exactly match the snapshot normalized action")
    elif any(value != 0 for value in normalized):
        raise ValueError("Whole-body start/reset snapshots require zero normalized actions")


def _human_snapshot(human):
    if not isinstance(human, dict):
        raise ValueError("Human snapshot must be an object")
    for field in ("segments", "muscles"):
        geometry = human.get(field)
        if not isinstance(geometry, list) or len(geometry) > MAX_HUMAN_GEOMETRY:
            raise ValueError(f"Human {field} must contain at most {MAX_HUMAN_GEOMETRY} items")
        identifiers = set()
        for item in geometry:
            if not isinstance(item, dict):
                raise ValueError(f"Human {field} geometry must be an object")
            name = item.get("id" if field == "segments" else "name")
            if not isinstance(name, str) or not 1 <= len(name) <= 100 or name in identifiers:
                raise ValueError(f"Human {field} require unique bounded names")
            identifiers.add(name)
            _finite_vector(item.get("a"), 3, f"Human {field} start")
            _finite_vector(item.get("b"), 3, f"Human {field} end")
            if field == "segments":
                radius = item.get("radius")
                if type(radius) not in (int, float) or not math.isfinite(radius) or not 0 < radius <= 2:
                    raise ValueError("Human segment radius must be finite and between 0 and 2")
            else:
                activation = item.get("activation")
                if (type(activation) not in (int, float) or not math.isfinite(activation)
                        or not 0 <= activation <= 1 or item.get("status") != "illustrative"):
                    raise ValueError("Human muscle activation must be illustrative and between 0 and 1")
    if not human["segments"]:
        raise ValueError("Human snapshot requires body segments")
    angles = human.get("angles", {})
    values = angles.values() if isinstance(angles, dict) else angles
    if (not isinstance(angles, (dict, list)) or len(angles) > MAX_HUMAN_PARTS
            or any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 360 for v in values)):
        raise ValueError("Human joint angles must be a bounded finite collection")
    if isinstance(angles, dict) and any(not isinstance(k, str) or not 1 <= len(k) <= 100 for k in angles):
        raise ValueError("Human angle names must be bounded text")
    for field in ("joints", "angular_velocities"):
        if field not in human:
            continue
        values = human[field]
        if not isinstance(values, dict) or len(values) > MAX_HUMAN_PARTS:
            raise ValueError(f"Human {field} must be a bounded object")
        for name, value in values.items():
            if not isinstance(name, str) or not 1 <= len(name) <= 100:
                raise ValueError(f"Human {field} names must be bounded text")
            if field == "joints":
                _finite_vector(value, 3, "Human joint position")
            elif type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1000:
                raise ValueError("Human angular velocities must be finite and bounded")
    selected = human.get("selected_joint")
    if selected is not None and (not isinstance(selected, str) or not 1 <= len(selected) <= 100):
        raise ValueError("Human selected joint must be a bounded name")
    target = human.get("target_angle")
    if target is not None and (type(target) not in (int, float) or not math.isfinite(target) or abs(target) > 360):
        raise ValueError("Human target angle must be finite and bounded")
    if "height" in human:
        height = human["height"]
        if type(height) not in (int, float) or not math.isfinite(height) or not 0.2 <= height <= 10:
            raise ValueError("Human height must be finite and between 0.2 and 10")
    if "origin" in human:
        _finite_vector(human["origin"], 3, "Human origin")
    if "view_height" in human or "view_origin" in human:
        height = human.get("view_height")
        maximum = 30 if "mobility" in human else 10
        if type(height) not in (int, float) or not math.isfinite(height) or not .2 <= height <= maximum:
            raise ValueError(f"Human view height must be finite and between 0.2 and {maximum}")
        _finite_vector(human.get("view_origin"), 3, "Human view origin")


def _snapshot(snapshot, scene):
    if scene.get("profile") == "shared-exercise-recording-v1":
        from weft.shared_exercise_recording import validate_snapshot
        validate_snapshot(snapshot, scene)
        _encoded(snapshot)
        return
    if scene.get("profile") == "shared-world-v1":
        from weft.shared_world import validate_shared_snapshot
        validate_shared_snapshot(snapshot, scene)
        _encoded(snapshot)
        return
    if scene.get("profile") == "spatial-rigid-v1":
        from weft.spatial_rigid import validate_snapshot
        validate_snapshot(snapshot, scene)
        _encoded(snapshot)
        return
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("objects"), list):
        raise ValueError("Recording snapshot requires an objects list")
    oriented = scene.get("profile") == "planar-rigid-v2"
    if oriented:
        from weft.scene import MAX_ANGULAR_SPEED
        if (snapshot.get("units") != scene.get("units")
                or any(type(snapshot.get(key)) is not bool for key in ("angular_speed_limited", "substep_limited"))
                or type(snapshot.get("substeps")) is not int or not 0 <= snapshot["substeps"] <= 256):
            raise ValueError("Oriented snapshot must preserve units and integration limits")
    expected = {obj["id"] for obj in scene["objects"]}
    seen = set()
    for obj in snapshot["objects"]:
        if not isinstance(obj, dict) or obj.get("id") not in expected or obj["id"] in seen:
            raise ValueError("Snapshot object identities do not match the recorded scene")
        seen.add(obj["id"])
        for field in ("position", "velocity"):
            vector = obj.get(field)
            if not isinstance(vector, (list, tuple)) or len(vector) != 2 or any(
                    type(v) not in (int, float) or not math.isfinite(v) for v in vector):
                raise ValueError(f"Snapshot {field} must contain two finite numbers")
        if oriented:
            for field, limit in (("angle", 10000), ("angular_velocity", MAX_ANGULAR_SPEED)):
                value = obj.get(field)
                if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > limit:
                    raise ValueError(f"Oriented snapshot requires finite bounded {field}")
    if seen != expected:
        raise ValueError("Snapshot must preserve every scene object")
    if "human" in snapshot:
        _human_snapshot(snapshot["human"])
        if scene.get("profile") in {"exercise-v3", *DIRECT_PROFILES}:
            _mobility_snapshot(snapshot["human"].get("mobility"))
        if scene.get("profile") in DIRECT_PROFILES:
            human, names = snapshot["human"], _whole_body_contract(scene)
            rotational = DIRECT_PROFILES[scene["profile"]][0]
            count = rotational + 3
            if scene["profile"] == FOOT_BODY_PROFILE:
                candidate = scene["foot_candidate"]
                if human.get("foot_candidate") != {k: v for k, v in candidate.items()
                                                    if k not in {"body_sha256", "foot_reference_sha256"}}:
                    raise ValueError("Foot snapshot must retain its candidate assumptions")
                for side in ("left", "right"):
                    for point in ("ankle", "heel", "foot_index"):
                        _finite_vector(human.get("joints", {}).get(side + "_" + point), 3, "Foot joint position")
                if not {"left_foot", "right_foot"} <= {segment["id"] for segment in human["segments"]}:
                    raise ValueError("Foot snapshots must retain both collision-capsule segments")
            if human.get("control_mode") != "direct_motor_actions":
                raise ValueError("Whole-body snapshots require direct_motor_actions control mode")
            if (not isinstance(human.get("angles"), dict) or set(human["angles"]) != set(names[:rotational])
                    or len(human["muscles"]) != 2 * rotational):
                raise ValueError(f"Whole-body snapshots must retain {rotational} named angles and {2 * rotational} illustrative bands")
            if human.get("action_names") != names:
                raise ValueError("Whole-body snapshot action names differ from the scene contract")
            _finite_vector(human.get("normalized_action"), count, "Whole-body normalized action", limit=1)
            _finite_vector(human.get("motor_forces"), count, "Whole-body motor forces", limit=2000)
            if human.get("motor_units") != ["Nm"] * rotational + ["N"] * 3:
                raise ValueError(f"Whole-body motor units must distinguish {rotational} torques and 3 root forces")
            forces, normalized = human["motor_forces"], human["normalized_action"]
            limits = scene["action_contract"]["physical_limits"]
            if any(not math.isclose(force, action * limit, rel_tol=0, abs_tol=1e-9)
                   for force, action, limit in zip(forces, normalized, limits)):
                raise ValueError("Whole-body motor forces must match normalized actions times physical limits")
            mobility = human["mobility"]
            if mobility["support_enabled"] is not False:
                raise ValueError("Whole-body direct control must disable automatic support")
            if any(not math.isclose(force, root, rel_tol=0, abs_tol=1e-9)
                   for force, root in zip(forces[-3:], mobility["support_force"])):
                raise ValueError("Whole-body mobility root forces must match the last three motor forces")
    elif scene.get("profile") in {"exercise-v1", "exercise-v2", "exercise-v3", *DIRECT_PROFILES}:
        raise ValueError("Exercise snapshots require the complete human state")
    _encoded(snapshot)


def _snapshot_validator(scene):
    if scene.get("profile") == "shared-exercise-recording-v1":
        from weft.shared_exercise_recording import prepare_snapshot_validator
        return prepare_snapshot_validator(scene)
    return lambda snapshot: _snapshot(snapshot, scene)


def _atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class Recorder:
    def __init__(self, root, scene, profile="planar"):
        self.root = Path(root).resolve()
        self.scene = json.loads(_encoded(scene))
        if self.scene.get("profile") == "planar-rigid-v2":
            from weft.scene import validate_scene
            validate_scene(self.scene)
        if self.scene.get("profile") == "spatial-rigid-v1":
            from weft.spatial_rigid import validate_scene
            validate_scene(self.scene)
        if self.scene.get("profile") == "shared-world-v1":
            from weft.shared_world import validate_shared_world
            self.scene = validate_shared_world(self.scene)
        if self.scene.get("profile") == "shared-exercise-recording-v1":
            from weft.shared_exercise_recording import validate_recording_scene
            self.scene = validate_recording_scene(self.scene)
        human = profile in {"human", "exercise"} or self.scene.get("profile") in {"exercise-v1", "exercise-v2", "exercise-v3", *DIRECT_PROFILES}
        if self.scene.get("profile") in DIRECT_PROFILES:
            _whole_body_contract(self.scene)
        object_limit = MAX_HUMAN_PARTS if human else 12
        if not isinstance(self.scene.get("objects"), list) or not 1 <= len(self.scene["objects"]) <= object_limit:
            raise ValueError(f"A recording requires 1–{object_limit} scene objects")
        if len({obj["id"] for obj in self.scene["objects"]}) != len(self.scene["objects"]):
            raise ValueError("Scene object identities must be unique")
        self._validate_snapshot = _snapshot_validator(self.scene)
        self.profile = profile
        self._stream = None
        self._started = None
        self._closed = False
        self._count = 0
        self._steps = 0
        self._resets = 0
        self._bytes = 0
        self._metadata = None
        self._journal_limit = 16 * 1024 * 1024 if self.scene.get("profile") == "spatial-rigid-v1" else MAX_JOURNAL_BYTES

    def start(self, initial_snapshot, seed=0):
        if self._started is not None:
            raise RuntimeError("Recording has already started")
        _seed(seed)
        self._validate_snapshot(initial_snapshot)
        _whole_body_event({"type": "start", "seed": seed, "snapshot": initial_snapshot}, self.scene)
        if len(_encoded({"type": "start", "seed": seed, "snapshot": initial_snapshot,
                         "index": 0}).encode("utf-8")) + 1 > MAX_EVENT_BYTES:
            raise ValueError("Recording initial state exceeds the event size limit")
        self.root.mkdir(parents=True, exist_ok=False)
        self._metadata = {
            "schema_version": 1, "profile": self.profile, "created_at": local_time(),
            "scene_sha256": canonical_hash(self.scene), "seed": seed, "status": "recording",
            "scope": "Recorded simulated actions and states; not physical-world observations",
            "limits": {"max_events": MAX_EVENTS, "max_wall_seconds": MAX_SECONDS,
                       "max_journal_bytes": self._journal_limit},
        }
        write_json(self.root / "scene.json", self.scene)
        _atomic_json(self.root / "recording.json", self._metadata)
        self._stream = (self.root / "events.jsonl").open("x", encoding="utf-8")
        self._started = time.monotonic()
        self._write({"type": "start", "seed": seed, "snapshot": initial_snapshot})
        return self.summary()

    def _write(self, event):
        record = dict(event, index=self._count)
        line = _encoded(record) + "\n"
        length = len(line.encode("utf-8"))
        if length > MAX_EVENT_BYTES or self._bytes + length > self._journal_limit:
            raise ValueError("Recording size limit reached; stop and save this session")
        # Flush each action batch to keep completed lines available after interruption.
        self._stream.write(line)
        self._stream.flush()
        self._count += 1
        self._bytes += length
        self._steps += event["type"] == "step"
        self._resets += event["type"] == "reset"

    def check_capacity(self):
        if self._stream is None or self._closed:
            raise RuntimeError("Recording is not active")
        if (self._count >= MAX_EVENTS or time.monotonic() - self._started > MAX_SECONDS
                or self._bytes + MAX_EVENT_BYTES > self._journal_limit):
            raise ValueError("Recording limit reached; stop and save this session")

    def append(self, event):
        self.check_capacity()
        if not isinstance(event, dict) or event.get("type") not in {"step", "reset"}:
            raise ValueError("Recording event must be a step or reset")
        self._validate_snapshot(event.get("snapshot"))
        if event["type"] == "reset":
            _seed(event.get("seed"))
        else:
            _action(event.get("action"), self.scene)
            if type(event.get("reward")) not in (int, float) or not math.isfinite(event["reward"]):
                raise ValueError("Recording reward must be finite")
            if any(type(event.get(key)) is not bool for key in ("terminated", "truncated")):
                raise ValueError("Recording step requires termination flags")
        _whole_body_event(event, self.scene)
        self._write(event)
        return self.summary()

    def summary(self):
        return {"directory": str(self.root), "events": self._count,
                "bytes": self._bytes, "closed": self._closed,
                "steps": self._steps, "resets": self._resets}

    def close(self):
        if self._started is None:
            raise RuntimeError("Recording has not started")
        if self._closed:
            return self.summary()
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._stream.close()
        self._closed = True
        self._metadata.update(status="closed", closed_at=local_time(), events=self._count,
                              journal_bytes=self._bytes,
                              wall_seconds=time.monotonic() - self._started)
        _atomic_json(self.root / "recording.json", self._metadata)
        return self.summary()


def load_recording(root, *, allow_incomplete=False, deadline=None):
    """Recover complete journal lines without inventing a truncated final event."""
    _check_deadline(deadline)
    root = Path(root)
    scene = json.loads((root / "scene.json").read_text())
    if scene.get("profile") == "spatial-rigid-v1":
        from weft.spatial_rigid import validate_scene
        validate_scene(scene)
    if scene.get("profile") == "shared-world-v1":
        from weft.shared_world import validate_shared_world
        scene = validate_shared_world(scene)
    if scene.get("profile") == "shared-exercise-recording-v1":
        from weft.shared_exercise_recording import validate_recording_scene
        scene = validate_recording_scene(scene)
    metadata = json.loads((root / "recording.json").read_text())
    validate_snapshot = _snapshot_validator(scene)
    journal = root / "events.jsonl"
    if journal.stat().st_size > MAX_JOURNAL_BYTES:
        raise ValueError("Recording exceeds journal read bound")
    events, incomplete = [], False
    with journal.open("rb") as stream:
        while line := stream.readline(MAX_EVENT_BYTES + 1):
            _check_deadline(deadline)
            if len(line) > MAX_EVENT_BYTES:
                raise ValueError("Recording event exceeds read bound")
            if not line.endswith(b"\n"):
                if allow_incomplete and not stream.read(1):
                    incomplete = True
                    break
                raise ValueError("Recording contains an incomplete journal event")
            event = json.loads(line)
            validate_snapshot(event.get("snapshot"))
            if event.get("index") != len(events) or event.get("type") not in {"start", "step", "reset"}:
                raise ValueError("Recording event sequence is invalid")
            if (not events and event["type"] != "start") or (events and event["type"] == "start"):
                raise ValueError("Recording requires exactly one initial event")
            if event["type"] == "step":
                _action(event.get("action"), scene)
            _whole_body_event(event, scene)
            events.append(event)
            if len(events) > MAX_EVENTS:
                raise ValueError("Recording exceeds event read bound")
    if not events:
        raise ValueError("Recording has no complete events")
    if scene.get("profile") == "shared-exercise-recording-v1":
        _seed(metadata.get("seed"))
        if metadata["seed"] != events[0]["seed"]:
            raise ValueError("Exercise recording metadata seed differs from its initial event")
    if metadata["scene_sha256"] != canonical_hash(scene):
        raise ValueError("Recorded scene identity does not match metadata")
    if not allow_incomplete and metadata.get("status") != "closed":
        raise ValueError("Recording has not been closed; use incomplete recovery explicitly")
    if metadata.get("status") == "closed" and metadata.get("events") != len(events):
        raise ValueError("Recording event count does not match metadata")
    _check_deadline(deadline)
    return {"scene": scene, "metadata": metadata, "events": events,
            "incomplete_journal": incomplete, "interrupted": metadata.get("status") != "closed"}


def _render_human(snapshot, width):
    human = snapshot["human"]
    _human_snapshot(human)
    height = round(width * 0.9)
    height += height % 2
    image = Image.new("RGB", (width, height + 66), "#F5EADC")
    draw = ImageDraw.Draw(image)
    body_height = human.get("view_height", human.get("height"))
    if body_height is None:
        body_height = max(.2, max(point[2] for part in human["segments"] for point in (part["a"], part["b"])) + .15)
    scale = min(width / (body_height * 1.3), height / (body_height * 1.2))
    yaw, pitch = 0.5, 0.2
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)

    def project(point):
        x, y, z = np.subtract(point, human.get("view_origin", [0, 0, 0]))
        return (width * 0.5 + (x * cy - y * sy) * scale,
                height * (0.5 if "view_origin" in human else 0.88) - (z * cp - (x * sy + y * cy) * sp) * scale)

    def capsule(a, b, radius, color):
        a, b = project(a), project(b)
        size = max(1, round(2 * radius * scale))
        draw.line((*a, *b), fill=color, width=size)
        for x, y in (a, b):
            r = size / 2
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color)

    mobility = human.get("mobility")
    if mobility is not None:
        _mobility_snapshot(mobility)
    floor = mobility["floor_height"] if mobility else 0
    normal = mobility.get("floor_normal", [0, 0, 1]) if mobility else [0, 0, 1]
    grid = ([round(value*2)/2 for value in human['view_origin'][:2]]
            if human.get('view_mode') == 'body_follow' else [0, 0])

    def floor_point(x, y):
        x, y = x+grid[0], y+grid[1]
        return [x, y, floor - (normal[0] * x + normal[1] * y) / normal[2]]

    for offset in (-0.5, 0, 0.5):
        draw.line((*project(floor_point(offset, -0.5)), *project(floor_point(offset, 0.5))), fill="#D8C9BA", width=1)
        draw.line((*project(floor_point(-0.5, offset)), *project(floor_point(0.5, offset))), fill="#D8C9BA", width=1)
    for segment in sorted(human["segments"], key=lambda s: sum(p[0] * sy + p[1] * cy for p in (s["a"], s["b"]))):
        capsule(segment["a"], segment["b"], segment["radius"] + 0.006, "#214F50")
        capsule(segment["a"], segment["b"], segment["radius"], "#C3B19F")
    for muscle in human["muscles"]:
        activation = muscle["activation"]
        color = (round(150 + 50 * activation), round(86 - 25 * activation), round(68 - 15 * activation))
        capsule(muscle["a"], muscle["b"], 0.007 + 0.005 * activation, color)
    if mobility:
        for contact in mobility["contacts"]:
            x, y = project(contact["position"])
            draw.ellipse((x-3, y-3, x+3, y+3), fill="#965644")
    label = "INFERRED POSE · KINEMATIC REPLAY" if human.get("reference") else "RECORDED SIMULATION"
    if human.get("foot_candidate"):
        label += " · UNVERIFIED FOOT CANDIDATE"
    draw.text((10, height + 8),
              f"{label} | {snapshot.get('source_time_s', snapshot.get('time_s', 0)):.2f}s",
              fill="#214F50")
    draw.text((10, height + 28), "ILLUSTRATIVE MUSCLES · NOT MEASURED ACTIVATION", fill="#965644")
    if human.get("control_mode") == "direct_motor_actions":
        control = "DIRECT MOTOR ACTIONS · VIRTUAL ROOT FORCES · NO AUTOMATIC ASSISTANCE"
    else:
        support = ("SUPPORT ON" if mobility["support_enabled"] else "SUPPORT OFF") if mobility else "ANCHORED"
        control = f"{support} · Simulated body; not recovered anatomy or biomechanics"
        if human.get('view_mode') == 'body_follow' and mobility:
            position = ', '.join(f'{value:.2f}' for value in mobility['translation'])
            control = f'VIEW FOLLOWS BODY · position [{position}] · {support}'
    draw.text((10, height + 46), control, fill="#596562")
    return np.asarray(image)


def render_snapshot(snapshot, scene, width=640):
    if scene.get("profile") == "shared-exercise-recording-v1":
        from weft.shared_exercise_recording import render_snapshot as render_exercise
        return render_exercise(snapshot, scene, width=width)
    if scene.get("profile") == "shared-world-v1":
        from weft.shared_world import render_snapshot as render_shared
        return render_shared(snapshot, scene, width=width)
    if scene.get("profile") == "spatial-rigid-v1":
        from weft.spatial_rigid import render_snapshot as render_spatial
        return render_spatial(snapshot, scene, width=width)
    if scene.get("profile") in {"exercise-v3", *DIRECT_PROFILES}:
        _snapshot(snapshot, scene)
    if "human" in snapshot:
        return _render_human(snapshot, width)
    height = max(160, min(960, round(width * scene.get("height", 0.5625) / scene.get("width", 1))))
    height += height % 2
    image = Image.new("RGB", (width, height + 42), "#F5EADC")
    draw = ImageDraw.Draw(image)
    sx = width / scene.get("width", 1)
    sy = height / scene.get("height", 0.5625)
    target = snapshot.get("target", scene.get("target"))
    if target is not None:
        tolerance = scene.get("task", {}).get("position_tolerance", 0.06)
        tx, ty = target[0] * sx, target[1] * sy
        draw.ellipse((tx - tolerance * sx, ty - tolerance * sy,
                      tx + tolerance * sx, ty + tolerance * sy), outline="#965644", width=2)
        draw.text((tx + 4, ty + 4), "goal", fill="#965644")
    definitions = {obj["id"]: obj for obj in scene["objects"]}
    for state in snapshot["objects"]:
        obj = definitions[state["id"]] | state
        x, y = obj["position"][0] * sx, obj["position"][1] * sy
        color = obj.get("color", "#C5A059")
        if obj.get("shape", "circle") == "box":
            w, h = obj["size"]
            angle = obj.get("angle", 0)
            c, s = math.cos(angle), math.sin(angle)
            points = [(x + (dx * c - dy * s) * sx, y + (dx * s + dy * c) * sy)
                      for dx, dy in ((-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2))]
            draw.polygon(points, fill=color, outline="#214F50", width=2)
        else:
            radius = obj.get("radius", 0.03)
            draw.ellipse((x - radius * sx, y - radius * sy,
                          x + radius * sx, y + radius * sy), fill=color, outline="#214F50", width=2)
        draw.text((x + 7, y + 7), obj.get("label", obj["id"]), fill="#214F50")
    draw.rectangle((0, 0, width - 1, height - 1), outline="#214F50", width=2)
    draw.text((10, height + 12),
              f"RECORDED SIMULATION | step {snapshot.get('steps', 0)} | {snapshot.get('time_s', 0):.2f}s",
              fill="#214F50")
    return np.asarray(image)


def _replay_html(scene, events):
    if scene.get("profile") == "shared-exercise-recording-v1":
        from weft.shared_exercise_recording import replay_view
        data = _encoded(replay_view(scene, events)).replace("<", "\\u003c")
        template = Path(__file__).parent / "assets/shared-world-replay.html"
        return template.read_text().replace("__WEFT_DATA__", data)
    data = _encoded({"scene": scene, "events": events}).replace("<", "\\u003c")
    if scene.get("profile") == "shared-world-v1":
        template = Path(__file__).parent / "assets/shared-world-replay.html"
        return template.read_text().replace("__WEFT_DATA__", data)
    if scene.get("profile") == "spatial-rigid-v1":
        template = Path(__file__).parent / "assets/spatial-replay.html"
        return template.read_text().replace("__WEFT_DATA__", data)
    return """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Weft · recorded simulation</title><style>
body{margin:30px auto;padding:0 20px;max-width:1050px;background:#F5EADC;color:#214F50;font:16px system-ui}
canvas{width:100%;background:#FBF7F1;border:1px solid #D8C9BA}button,input{font:inherit}button{padding:10px;border:1px solid #965644;background:#965644;color:#F5EADC;cursor:pointer}input{width:70%}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#FBF7F1;padding:16px;font-size:13px}a{color:#965644}
</style><h1>Recorded simulation</h1><p>Recorded actions and states. This is a replay, not a live environment or physical-world evidence.</p>
<section id="human-controls" hidden><p><strong>ILLUSTRATIVE MUSCLES · NOT MEASURED ACTIVATION</strong><br>Inferred articulated model; not recovered anatomy or biomechanics. Turning the camera does not change the recorded state.</p>
<p id="mobility-state" aria-live="polite"></p>
<label>Camera yaw <input id="yaw" aria-label="Camera yaw" type="range" min="-3.14" max="3.14" step="0.02" value="0.5" style="width:40%"></label>
<label><input id="muscles" type="checkbox" checked style="width:auto"> Show illustrative muscles</label></section>
<label id="foot-focus-control" hidden><input id="foot-focus" type="checkbox" style="width:auto"> Close-up of feet</label>
<canvas id="view" width="960" height="540" aria-label="Recorded simulation frame"></canvas>
<p><button id="play">Play</button> <input id="scrub" aria-label="Recording frame" type="range" min="0" value="0"> <output id="frame"></output></p>
<p><a href="video.mp4" download>Download video</a> · <a href="events.jsonl" download>Download actions and states</a> · <a href="scene.json" download>Download scene</a></p>
<details><summary>Frame details</summary><pre id="detail"></pre></details>
<script id="data" type="application/json">""" + data + """</script><script>
'use strict';
const {scene,events}=JSON.parse(document.getElementById('data').textContent);
const canvas=document.getElementById('view'), ctx=canvas.getContext('2d'), slider=document.getElementById('scrub'), play=document.getElementById('play');
const human=Boolean(events[0].snapshot.human),yaw=document.getElementById('yaw'),muscles=document.getElementById('muscles');
const footFocus=document.getElementById('foot-focus');
document.getElementById('foot-focus-control').hidden=!events[0].snapshot.human?.foot_candidate;
document.getElementById('human-controls').hidden=!human;
canvas.height=human?Math.round(canvas.width*.9):Math.max(160,Math.min(1440,Math.round(canvas.width*scene.height/scene.width)));
slider.max=events.length-1; const shapes=new Map(scene.objects.map(o=>[o.id,o])); let timer=null;
function stop(){if(timer!==null)clearTimeout(timer);timer=null;play.textContent='Play';}
function drawHuman(h){const footPoints=footFocus.checked&&h.foot_candidate?['left_ankle','right_ankle','left_heel','right_heel','left_foot_index','right_foot_index'].map(k=>h.joints[k]):null;
const footCenter=footPoints?[0,1,2].map(i=>footPoints.reduce((sum,p)=>sum+p[i],0)/footPoints.length):null;
const footHeight=footPoints?Math.max(.35,...[0,1,2].map(i=>1.8*(Math.max(...footPoints.map(p=>p[i]))-Math.min(...footPoints.map(p=>p[i]))))):null;
const bodyHeight=footHeight||h.view_height||h.height||Math.max(.2,Math.max(...h.segments.flatMap(s=>[s.a[2],s.b[2]]))+.15);
const scale=Math.min(canvas.width/(bodyHeight*1.3),canvas.height/(bodyHeight*1.2)),cy=Math.cos(Number(yaw.value)),sy=Math.sin(Number(yaw.value)),cp=Math.cos(.2),sp=Math.sin(.2);
const origin=footCenter||h.view_origin||[0,0,0];
const project=([x,y,z])=>{x-=origin[0];y-=origin[1];z-=origin[2];return [canvas.width*.5+(x*cy-y*sy)*scale,canvas.height*(h.view_origin ? .5 : .88)-(z*cp-(x*sy+y*cy)*sp)*scale];};
function line(a,b,r,color){a=project(a);b=project(b);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.lineWidth=Math.max(1,2*r*scale);ctx.lineCap='round';ctx.strokeStyle=color;ctx.stroke();}
const floor=h.mobility?.floor_height??0;
const normal=h.mobility?.floor_normal||[0,0,1],grid=h.view_mode==='body_follow'?origin.slice(0,2).map(v=>Math.round(v*2)/2):[0,0];
const floorPoint=(x,y)=>{x+=grid[0];y+=grid[1];return [x,y,floor-(normal[0]*x+normal[1]*y)/normal[2]];};
for(const p of [-.5,0,.5]){line(floorPoint(p,-.5),floorPoint(p,.5),.001,'#D8C9BA');line(floorPoint(-.5,p),floorPoint(.5,p),.001,'#D8C9BA');}
const depth=s=>s.a[0]*sy+s.a[1]*cy+s.b[0]*sy+s.b[1]*cy;
for(const s of [...h.segments].sort((a,b)=>depth(a)-depth(b))){line(s.a,s.b,s.radius+.006,'#214F50');line(s.a,s.b,s.radius,'#C3B19F');}
if(muscles.checked)for(const m of h.muscles){const a=m.activation;line(m.a,m.b,.007+.005*a,`rgb(${Math.round(150+50*a)},${Math.round(86-25*a)},${Math.round(68-15*a)})`);}
for(const contact of h.mobility?.contacts||[]){const [x,y]=project(contact.position);ctx.beginPath();ctx.arc(x,y,4,0,Math.PI*2);ctx.fillStyle='#965644';ctx.fill();}
ctx.lineCap='butt';}
function draw(){const i=Number(slider.value),event=events[i],state=event.snapshot,sx=canvas.width/scene.width,sy=canvas.height/scene.height;
ctx.clearRect(0,0,canvas.width,canvas.height);ctx.strokeStyle='#965644';ctx.lineWidth=2;
if(state.human){drawHuman(state.human);const mobility=state.human.mobility;
const control=(state.human.control_mode==='direct_motor_actions'?'DIRECT MOTOR ACTIONS · VIRTUAL ROOT FORCES · NO AUTOMATIC ASSISTANCE':(mobility?.support_enabled?'SUPPORT ON':'SUPPORT OFF'))+(state.human.foot_candidate?' · UNVERIFIED FEET · STARTING-POSE TASK':'');
const rootForce=state.human.control_mode==='direct_motor_actions'&&mobility?` · virtual root force [${mobility.support_force.map(v=>v.toFixed(1)).join(', ')}] N`:'';
const travel=state.human.view_mode==='body_follow'&&mobility?` · view follows body · position [${mobility.translation.map(v=>v.toFixed(2)).join(', ')}] model units`:'';
document.getElementById('mobility-state').textContent=mobility?`${control}${rootForce}${travel} · ${mobility.contact_count} simulated contacts · ${mobility.total_normal_force.toFixed(1)} N normal force (not muscle activation)`:'ANCHORED · No mobile root or floor-contact model';}else{
const target=state.target||scene.target;if(target){ctx.beginPath();ctx.ellipse(target[0]*sx,target[1]*sy,(scene.task?.position_tolerance||.06)*sx,(scene.task?.position_tolerance||.06)*sy,0,0,2*Math.PI);ctx.stroke();}
for(const stateObj of state.objects){const o={...shapes.get(stateObj.id),...stateObj};ctx.save();ctx.translate(o.position[0]*sx,o.position[1]*sy);ctx.rotate(o.angle||0);ctx.fillStyle=o.color||'#C5A059';ctx.strokeStyle='#214F50';ctx.beginPath();if(o.shape==='box')ctx.rect(-o.size[0]*sx/2,-o.size[1]*sy/2,o.size[0]*sx,o.size[1]*sy);else ctx.ellipse(0,0,(o.radius||.03)*sx,(o.radius||.03)*sy,0,0,2*Math.PI);ctx.fill();ctx.stroke();ctx.restore();}
}
document.getElementById('frame').textContent=`${i+1} / ${events.length} · ${event.type} · ${(state.time_s||0).toFixed(2)}s`;
document.getElementById('detail').textContent=JSON.stringify(event,null,2);}
function tick(){if(document.hidden){stop();return;}if(Number(slider.value)>=events.length-1){stop();return;}slider.value=Number(slider.value)+1;draw();timer=setTimeout(tick,1000*(scene.dt||1/30));}
play.addEventListener('click',()=>{if(timer!==null){stop();return;}if(Number(slider.value)>=events.length-1)slider.value=0;play.textContent='Pause';draw();timer=setTimeout(tick,1000*(scene.dt||1/30));});
slider.addEventListener('input',()=>{stop();draw();});document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});window.addEventListener('pagehide',stop);draw();
yaw.addEventListener('input',draw);muscles.addEventListener('change',draw);footFocus.addEventListener('change',draw);
</script></html>"""


def export_recording(root, *, deadline=None, max_output_bytes=None):
    root = Path(root).resolve()
    _check_deadline(deadline)
    if max_output_bytes is not None and (type(max_output_bytes) is not int or not 1 <= max_output_bytes <= 256 * 1024 * 1024):
        raise ValueError("Recording export bound must be a positive integer of at most 256 MiB")

    def check_budget():
        _check_deadline(deadline)
        if max_output_bytes is not None:
            names = ("scene.json", "events.jsonl", "recording.json", "video.mp4", "video.partial.mp4", "replay.html", "preview.gif")
            size = sum(path.stat().st_size for name in names if (path := root / name).is_file())
            if size > max_output_bytes:
                raise ValueError("Recording export exceeded its byte bound; partial files are preserved")

    check_budget()
    if (root / "manifest.json").exists():
        return verify_recording(root, deadline=deadline)
    data = load_recording(root, deadline=deadline)
    scene, events = data["scene"], data["events"]
    for name in ("video.mp4", "video.partial.mp4", "replay.html", "preview.gif"):
        if (root / name).exists():
            raise FileExistsError(f"Partial export exists; preserve it and inspect before retrying: {root / name}")
    dt = scene.get("dt", 1 / 30)
    if type(dt) not in (int, float) or not math.isfinite(dt) or not 1/240 <= dt <= 1:
        raise ValueError("Recording scene timestep is unsupported")
    if scene.get("profile") == "shared-exercise-recording-v1":
        from weft.shared_exercise_recording import prepare_renderer
        render = prepare_renderer(scene)
    else:
        render = lambda snapshot: render_snapshot(snapshot, scene)
    first = render(events[0]["snapshot"])
    temporary = root / "video.partial.mp4"
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), 1 / dt,
                             (first.shape[1], first.shape[0]))
    if not writer.isOpened():
        raise RuntimeError("MP4 encoder is unavailable; the action journal is preserved")
    previews, preview_indices = [], set(np.linspace(0, len(events)-1, min(120, len(events)), dtype=int))
    try:
        for index, event in enumerate(events):
            check_budget()
            frame = first if index == 0 else render(event["snapshot"])
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if index in preview_indices:
                image = Image.fromarray(frame)
                image.thumbnail((480, 300))
                previews.append(image)
    finally:
        writer.release()
    check_budget()
    cap = cv2.VideoCapture(str(temporary))
    decoded = 0
    try:
        while cap.read()[0]:
            check_budget()
            decoded += 1
    finally:
        cap.release()
    if decoded != len(events):
        raise RuntimeError(f"Recording video decode mismatch: {decoded}/{len(events)}; journal preserved")
    check_budget()
    temporary.rename(root / "video.mp4")
    (root / "replay.html").write_text(_replay_html(scene, events), encoding="utf-8")
    # The GIF is a bounded preview; the MP4 and journal retain every recorded state.
    previews[0].save(root / "preview.gif", save_all=True, append_images=previews[1:],
                     duration=max(20, round(1000 * dt * len(events) / len(previews))), loop=0)
    check_budget()
    manifest = {"schema_version": 1, "created_at": local_time(), "video_frames": decoded,
                "files": {name: digest(root / name) for name in
                          ("scene.json", "events.jsonl", "recording.json", "video.mp4", "replay.html", "preview.gif")}}
    _atomic_json(root / "manifest.json", manifest)
    return verify_recording(root, deadline=deadline)


def verify_recording(root, *, deadline=None):
    _check_deadline(deadline)
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    required = {"scene.json", "events.jsonl", "recording.json", "video.mp4", "replay.html", "preview.gif"}
    if set(manifest.get("files", {})) != required:
        raise ValueError("Recording manifest does not contain the complete export")
    for name, expected in manifest["files"].items():
        _check_deadline(deadline)
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or digest(path) != expected:
            raise ValueError(f"Recording integrity failed: {name}")
    data = load_recording(root, deadline=deadline)
    if manifest.get("video_frames") != len(data["events"]):
        raise ValueError("Recording frame count does not match the journal")
    _check_deadline(deadline)
    return {"directory": str(root), "verified_files": len(required),
            "events": len(data["events"]),
            "steps": sum(e["type"] == "step" for e in data["events"]),
            "resets": sum(e["type"] == "reset" for e in data["events"]),
            "video_frames": manifest["video_frames"], "manifest_sha256": digest(root / "manifest.json"),
            "replay": str(root / "replay.html"), "video": str(root / "video.mp4"),
            "scope": "File integrity and complete recorded-state replay; not physical validation"}


def recover_incomplete_recording(source_dir, out_dir, *, export=True):
    """Non-destructively recover complete validated journal events into a new recording directory."""
    source_dir, out_dir = Path(source_dir).resolve(), Path(out_dir).resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Recording directory does not exist: {source_dir}")
    if out_dir.exists():
        raise FileExistsError(f"Recovery output directory already exists: {out_dir}")
    if out_dir == source_dir or out_dir.is_relative_to(source_dir) or source_dir.is_relative_to(out_dir):
        raise ValueError("Recovery output directory must be outside the source recording directory")

    partial_exports = sorted(
        name
        for name in ("video.mp4", "video.partial.mp4", "replay.html", "preview.gif", "manifest.json")
        if (source_dir / name).exists()
    )
    loaded = load_recording(source_dir, allow_incomplete=True)
    if (
        (source_dir / "manifest.json").is_file()
        and not loaded["interrupted"]
        and not loaded["incomplete_journal"]
    ):
        raise ValueError("Recording is already complete and sealed; use verify_recording instead")

    scene = loaded["scene"]
    events = loaded["events"]
    source_meta = dict(loaded["metadata"])
    journal_limit = 16 * 1024 * 1024 if scene.get("profile") == "spatial-rigid-v1" else MAX_JOURNAL_BYTES

    encoded_lines = []
    journal_bytes = 0
    for event in events:
        line = (_encoded(event) + "\n").encode("utf-8")
        if len(line) > MAX_EVENT_BYTES or journal_bytes + len(line) > journal_limit:
            raise ValueError("Recovered journal exceeds recording size bound")
        encoded_lines.append(line)
        journal_bytes += len(line)

    recovered_at = local_time()
    recovered_steps = sum(e["type"] == "step" for e in events)
    recovered_resets = sum(e["type"] == "reset" for e in events)
    recovery = {
        "schema_version": 1,
        "recovered_from": str(source_dir),
        "source_scene_sha256": digest(source_dir / "scene.json"),
        "source_recording_sha256": digest(source_dir / "recording.json"),
        "source_events_sha256": digest(source_dir / "events.jsonl"),
        "source_status": source_meta.get("status"),
        "source_interrupted": bool(loaded["interrupted"]),
        "dropped_trailing_partial_event": bool(loaded["incomplete_journal"]),
        "source_partial_export_files": partial_exports,
        "recovered_events": len(events),
        "recovered_steps": recovered_steps,
        "recovered_resets": recovered_resets,
        "recovered_at": recovered_at,
        "scope": (
            "Non-destructive recovery of complete validated journal events into a new directory; "
            "the source crash directory is preserved unchanged and truncated tail data is never fabricated."
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=False)
    write_json(out_dir / "scene.json", scene)
    with (out_dir / "events.jsonl").open("xb") as stream:
        for line in encoded_lines:
            stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())

    closed_meta = {
        **source_meta,
        "status": "closed",
        "closed_at": source_meta.get("closed_at") or recovered_at,
        "events": len(events),
        "journal_bytes": journal_bytes,
        "recovery": recovery,
    }
    _atomic_json(out_dir / "recording.json", closed_meta)
    if export:
        verified = export_recording(out_dir)
        return {**verified, "recovery": recovery}
    return {
        "directory": str(out_dir),
        "status": "closed",
        "events": len(events),
        "steps": recovered_steps,
        "resets": recovered_resets,
        "recovery": recovery,
    }
