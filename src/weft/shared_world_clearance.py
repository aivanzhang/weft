"""Versioned kinematic clearance comparison; never a rollout or a task-success gate.

``compare_shared_world_clearance(base, alternative, source_run=...)`` requires an
exact sealed base and complete source proof for source-qualified scenes. Analytic
fixtures need no directory. Optional ``exercise_tasks=(base_task, new_task)`` must
differ only in scene identity. Inputs are not modified; returned JSON is detached.

Every source slot, reference pose, full-span 30-Hz sample and declared task control
boundary is retained. Source slots without a reference stay unavailable; separate
grid samples may interpolate bounded references, never extrapolate. Pair columns
identify every actual rigid part. Semantic union minima compare different part
topologies without pretending part IDs correspond. Positive deltas are reported
even below numerical tolerance. This inspection output is not model-facing data.
"""

from bisect import bisect_left
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import re
import time

import mujoco
import numpy as np

from weft.human import joint_contract
from weft.io import canonical_hash
from weft.shared_exercise_task import validate_shared_exercise_task
from weft.shared_world import _geometry_names, shared_world_xml, validate_shared_world
from weft.shared_world_packaging import collect_shared_source_files, read_package_file
from weft.world_evidence import _check_tree, _nonfinite, _unique_object, load_world_evidence


PROFILE = "shared-world-clearance-comparison-v1"
MAX_SCENE_BYTES = 16 * 1024 * 1024
MAX_REFERENCE_POSES = 900
MAX_SOURCE_SLOTS = 1800
MAX_SAMPLES = 4800
MAX_DISTANCE_QUERIES = 2_000_000
MAX_REPORT_BYTES = 64 * 1024 * 1024
MAX_GAP_SECONDS = .25
NUMERICAL_TOLERANCE = 1e-8
SCOPE = ("Frozen-reference kinematics with rigid objects fixed at their declared poses. "
         "Not observed contact, source-image fit, dynamic stability, continuous-time clearance, "
         "grasp, task success or physical validation.")
_PROTOCOL = {
    "profile": PROFILE, "schema_version": 1, "sample_hz": 30,
    "samples": ["every_source_slot", "every_reference_pose", "full_span_grid_including_final_endpoint",
                "every_declared_task_control_boundary"],
    "interpolation": "piecewise_linear_joint_and_root_references; separately labeled, never observed",
    "maximum_interpolation_gap_s": MAX_GAP_SECONDS, "extrapolation": "forbidden",
    "rigid_pose": "unchanged_declared_initial_pose_at_every_sample",
    "pairs": ["every_human_collision_geom_x_every_rigid_part", "every_rigid_part_x_floor",
              "every_human_collision_geom_x_floor"],
    "excluded_pairs": ["human_self", "rigid_rigid"],
    "comparison": "semantic human/object or floor pair minimum over every rigid part",
    "penetration": "max(0,-signed_gap_world); all positive increases retained",
    "numerical_tolerance_world": NUMERICAL_TOLERANCE,
    "tolerance_scope": "additional count only; never removes raw increases or changes a scorer",
    "distance_cutoff_world": 100., "physics_steps": 0,
    "maximum_reference_poses": MAX_REFERENCE_POSES, "maximum_source_slots": MAX_SOURCE_SLOTS,
    "maximum_samples": MAX_SAMPLES, "maximum_distance_queries": MAX_DISTANCE_QUERIES,
    "maximum_report_bytes": MAX_REPORT_BYTES,
    "scope": SCOPE,
}


def clearance_protocol():
    """Return the frozen protocol before constructing or fitting an alternative."""
    return deepcopy(_PROTOCOL)


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _json(raw):
    value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_nonfinite)
    _check_tree(value)
    return value


def _deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Clearance comparison exceeded its cooperative wall-time bound")


def _scene(value):
    _check_tree(value)
    if len(_encoded(value)) > MAX_SCENE_BYTES:
        raise ValueError("Clearance scene exceeds its input byte bound")
    return validate_shared_world(value)


def _same_world(base, alternative, tasks):
    mutable = {"objects", "assumptions", "learned_policy_available"}
    if {k: v for k, v in base.items() if k not in mutable} != {
            k: v for k, v in alternative.items() if k not in mutable}:
        raise ValueError("Comparison requires identical body, references, frame, clocks, source and task")
    if [o["id"] for o in base["objects"]] != [o["id"] for o in alternative["objects"]]:
        raise ValueError("Comparison cannot change object identities or order")
    geometry = {"shape", "size", "radius", "geometry", "provenance"}
    source_keys = ("source_prefix", "source_reference", "source_bounds_px", "camera_hypothesis", "reference_object_force")
    for before, after in zip(base["objects"], alternative["objects"]):
        if {k: v for k, v in before.items() if k not in geometry} != {
                k: v for k, v in after.items() if k not in geometry}:
            raise ValueError("Comparison cannot change object pose, velocity, mass, material or role")
        if any(before["provenance"].get(k) != after["provenance"].get(k) for k in source_keys):
            raise ValueError("Comparison cannot change object source or camera provenance")
    if tasks is None:
        return {"status": "not_supplied", "scene_task_unchanged": True}
    if not isinstance(tasks, (tuple, list)) or len(tasks) != 2:
        raise ValueError("Exercise tasks must be an explicit baseline/alternative pair")
    a, b = (validate_shared_exercise_task(task, scene) for task, scene in zip(tasks, (base, alternative)))
    if {k: v for k, v in a.items() if k != "scene_sha256"} != {
            k: v for k, v in b.items() if k != "scene_sha256"}:
        raise ValueError("Exercise tasks may differ only in their scene hash")
    return {"status": "scene_identity_only", "baseline_sha256": canonical_hash(a),
            "alternative_sha256": canonical_hash(b)}


def _source(base, source_run, deadline):
    binding = base["source_binding"]
    if binding["mode"] == "analytic_fixture":
        if source_run is not None:
            raise ValueError("Analytic fixtures have no source directory or observed slots")
        return {"status": "analytic_fixture", "slots": [], "clock": None,
                "source_proof": None, "reference_frames": [None] * len(base["human_spec"]["reference"])}
    if source_run is None:
        raise ValueError("Source-qualified comparison requires the complete sealed source_run")
    _deadline(deadline)
    proof = collect_shared_source_files(base, source_run)
    _deadline(deadline)
    raw_scene = read_package_file(Path(source_run), "scene.json", max_bytes=MAX_SCENE_BYTES,
                                  expected_sha256=proof.manifest_files.get("scene.json"))
    if "scene.json" not in proof.manifest_files or canonical_hash(_scene(_json(raw_scene))) != canonical_hash(base):
        raise ValueError("source_run must pin the exact canonical baseline scene")
    human = next(row for row in binding["entity_bindings"] if row["world_entity_id"] == "human")
    digest, entity = human["observation_document_sha256"], human["observation_entity_id"]
    if digest is None or entity is None:
        raise ValueError("Source-qualified human needs an explicit observation document/entity binding")
    raw = proof.files[binding["evidence_documents"][digest]]
    registry = load_world_evidence({digest: raw})
    document = _json(registry.source_bytes(digest))
    observations = sorted((row for row in document["observations"] if row["object_id"] == entity), key=lambda row: row["frame"])
    if len(observations) != binding["total_source_frames"] or len(observations) > MAX_SOURCE_SLOTS:
        raise ValueError("Human observation document must retain every bounded canonical source slot")
    metadata = base["human_spec"].get("reference_evidence")
    references = base["human_spec"]["reference"]
    if not isinstance(metadata, list) or len(metadata) != len(references):
        raise ValueError("Source references require one original evidence identity per pose")
    by_frame = {row["frame"]: row for row in observations}
    frames, by_reference = [], {}
    for index, (reference, meta) in enumerate(zip(references, metadata)):
        if not isinstance(meta, dict) or not isinstance(meta.get("evidence_id"), str):
            raise ValueError("Source reference metadata must retain an exact evidence ID")
        match = re.fullmatch(re.escape(document["source"]["sha256"]) + r":frame:(\d+)(?::.*)?", meta["evidence_id"])
        frame = int(match[1]) if match else None
        row = by_frame.get(frame)
        if (row is None or frame in by_reference or meta.get("time_s") != reference["time_s"]
                or not math.isclose(row["time_s"], reference["time_s"], rel_tol=0, abs_tol=1e-6)
                or meta["evidence_id"] not in row["evidence_ids"] or row["partition"] != "construction"):
            raise ValueError("Reference identity/clock must resolve to its exact construction source slot; no implicit alias retiming")
        frames.append(frame)
        by_reference[frame] = index
    slots = [{"source_frame": row["frame"], "source_time_s": row["time_s"], "observation_id": row["id"],
              "visibility": row["visibility"], "partition": row["partition"],
              "reference_index": by_reference.get(row["frame"])} for row in observations]
    return {"status": "verified_source_binding", "slots": slots, "clock": deepcopy(document["time"]),
            "document_sha256": digest, "entity_id": entity, "document_source_sha256": document["source"]["sha256"],
            "canonical_source_sha256": binding["source_sha256"], "manifest_sha256": proof.manifest_sha256,
            "source_proof": proof.report, "reference_frames": frames}


def _weights(times, timestamp):
    right = bisect_left(times, timestamp)
    for index in (right, right - 1):
        if 0 <= index < len(times) and abs(times[index] - timestamp) <= 1e-10:
            return [[index, 1.]], "reference_pose"
    if right == 0 or right == len(times):
        return [], "outside_reference_span"
    span = times[right] - times[right - 1]
    if span > MAX_GAP_SECONDS + 1e-9:
        return [], "unsupported_reference_gap"
    fraction = (timestamp - times[right - 1]) / span
    return [[right - 1, 1 - fraction], [right, fraction]], "interpolated_reference"


def _schedule(base, source):
    references = base["human_spec"]["reference"]
    times = [row["time_s"] for row in references]
    if not 2 <= len(times) <= MAX_REFERENCE_POSES:
        raise ValueError("Reference poses exceed the frozen comparison bound")
    slots = source["slots"]
    start = min(times[0], slots[0]["source_time_s"]) if slots else times[0]
    end = max(times[-1], slots[-1]["source_time_s"]) if slots else times[-1]
    if not 0 < end - start <= 60:
        raise ValueError("Comparison span must be positive and within 60 seconds")
    grid = [start + i / 30 for i in range(math.floor((end - start) * 30 + 1e-9) + 1)]
    if not math.isclose(grid[-1], end, rel_tol=0, abs_tol=1e-10):
        grid.append(end)
    else:
        grid[-1] = end
    task_start = times[0] + base["task"]["start_offset_s"]
    task_grid = [task_start + i * base["dt"] for i in range(base["task"]["horizon_steps"] + 1)]
    count = len(slots) + len(times) + len(grid) + len(task_grid)
    if count > MAX_SAMPLES:
        raise ValueError("Scheduled rows exceed the frozen comparison bound")
    rows = []
    for slot in slots:
        index = slot["reference_index"]
        rows.append({"kind": "source_slot", **slot, "reference_time_s": None if index is None else times[index],
                     "weights": [] if index is None else [[index, 1.]],
                     "pose_status": "no_reference_pose" if index is None else "reference_pose"})
    rows += [{"kind": "reference_pose", "reference_index": i, "source_frame": source["reference_frames"][i],
              "source_time_s": None, "reference_time_s": stamp, "weights": [[i, 1.]], "pose_status": "reference_pose"}
             for i, stamp in enumerate(times)]
    for kind, stamps in (("full_span_grid", grid), ("task_control", task_grid)):
        for index, stamp in enumerate(stamps):
            weights, status = _weights(times, stamp)
            rows.append({"kind": kind, "grid_index": index, "reference_time_s": stamp, "source_time_s": None,
                         "task_elapsed_s": stamp - task_start if kind == "task_control" else None,
                         "weights": weights, "pose_status": status})
    for row in rows:
        row.update(status="pending" if row["weights"] else "unavailable", baseline_gaps_world=None,
                   alternative_gaps_world=None, penetration_increases=None)
    return rows, {"reference_span_s": [times[0], times[-1]], "full_span_s": [start, end],
                  "task_reference_clock_span_s": [task_grid[0], task_grid[-1]],
                  "scene_clock_binding": deepcopy(base["coordinate_frame"]["clock_binding"]),
                  "scope": "Original observation clocks remain separate; grid time uses the unchanged human reference clock."}


class _Kinematics:
    def __init__(self, scene):
        self.model = mujoco.MjModel.from_xml_string(shared_world_xml(scene))
        self.data = mujoco.MjData(self.model)
        names, _ = joint_contract(scene["human_spec"])
        self.q = [int(self.model.jnt_qposadr[self.model.joint(name).id]) for name in (*names, "root_x", "root_y", "root_z")]
        spec = scene["human_spec"]
        scale = scene["coordinate_frame"]["scale_bridge"]["body_height_to_world_scale"]
        self.reference = np.asarray([[*[row["joints"][name] for name in names],
                                      *[value * scale for value in root["translation"]]]
                                     for row, root in zip(spec["reference"], spec["root_reference"])])
        self.start = spec["reference"][0]["time_s"]
        self.floor = self.model.geom("mobile_floor").id
        humans = [(gid, mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid))
                  for gid in range(self.model.ngeom) if int(self.model.geom_contype[gid]) & 2]
        if not 1 <= len(humans) <= 32:
            raise ValueError("Unsupported human collision geometry count")
        parts = [(self.model.geom(name).id, obj["id"], part, obj["role"])
                 for obj, part, name in _geometry_names(scene)]
        self.columns, self.pairs = [], []
        self.distance_queries = 0
        for gid, object_id, part_id, role in parts:
            for human_gid, human_name in humans:
                self._add((human_gid, gid), "human_rigid", human_name, object_id, part_id, role)
            self._add((gid, self.floor), "rigid_floor", None, object_id, part_id, role)
        for human_gid, human_name in humans:
            self._add((human_gid, self.floor), "human_floor", human_name, None, None, None)

    def _add(self, pair, kind, human, obj, part, role):
        self.pairs.append(pair)
        self.columns.append({"kind": kind, "human_geom": human, "object_id": obj, "part_id": part, "object_role": role})

    def measure(self, weights, timestamp, deadline, result):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.q] = sum(self.reference[index] * weight for index, weight in weights)
        self.data.time = timestamp - self.start
        mujoco.mj_forward(self.model, self.data)
        for index, (first, second) in enumerate(self.pairs):
            if index % 32 == 0:
                _deadline(deadline)
            self.distance_queries += 1
            gap = float(mujoco.mj_geomDistance(self.model, self.data, first, second, 100., None))
            if not math.isfinite(gap) or abs(gap) >= 100:
                raise ValueError("Signed-distance query is nonfinite or saturated; no finite gap is inferred")
            result[index] = gap
        return result


def _comparison_columns(before, after):
    groups = {}
    for side, columns in (("baseline", before), ("alternative", after)):
        for index, row in enumerate(columns):
            key = (row["kind"], row["human_geom"], row["object_id"])
            group = groups.setdefault(key, {k: row[k] for k in ("kind", "human_geom", "object_id")})
            group.setdefault(side + "_pair_indices", []).append(index)
    if any("baseline_pair_indices" not in row or "alternative_pair_indices" not in row for row in groups.values()):
        raise ValueError("Semantic clearance pairs differ between inputs")
    return list(groups.values())


def compare_shared_world_clearance(baseline, alternative, *, source_run=None, exercise_tasks=None, max_seconds=30):
    """Compare every declared sample; unsupported/partial outcomes never receive a pass label."""
    started = time.monotonic()
    report = {"schema_version": 1, "profile": PROFILE, "protocol": clearance_protocol(),
              "protocol_sha256": canonical_hash(_PROTOCOL), "status": "unsupported_input", "scope": SCOPE,
              "physical_validation": "unverified", "physics_steps": 0, "reasons": [], "samples": [],
              "comparison_columns": [], "pair_columns": {}, "summary": {}}
    stage = "input_validation"
    before = after = None
    try:
        if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 1 <= max_seconds <= 60:
            raise ValueError("Clearance comparison requires a finite 1–60 second cooperative bound")
        deadline = started + max_seconds
        base, changed = _scene(baseline), _scene(alternative)
        if len(base["human_spec"]["reference"]) > MAX_REFERENCE_POSES:
            raise ValueError("Reference poses exceed the frozen comparison bound")
        if (base["source_binding"]["total_source_frames"] or 0) > MAX_SOURCE_SLOTS:
            raise ValueError("Source slots exceed the frozen comparison bound")
        report.update(baseline_scene_sha256=canonical_hash(base), alternative_scene_sha256=canonical_hash(changed),
                      exercise_tasks=_same_world(base, changed, exercise_tasks),
                      coordinate_frame=deepcopy(base["coordinate_frame"]))
        _deadline(deadline)
        stage = "source_verification"
        source = _source(base, source_run, deadline)
        report["source"] = {k: v for k, v in source.items() if k not in {"slots", "reference_frames"}}
        stage = "schedule"
        rows, clocks = _schedule(base, source)
        report.update(samples=rows, clocks=clocks)
        valid = sum(bool(row["weights"]) for row in rows)
        # The supported body has 12 geoms; reserve for the bounded 32-geom contract before compilation.
        parts = [sum(1 for _ in _geometry_names(scene)) for scene in (base, changed)]
        query_reservation = valid * sum(33 * count + 32 for count in parts)
        report_reservation = (len(_encoded(report)) + query_reservation * 64 + len(rows) * 1024
                              + (sum(parts) * 33 + 64) * 1024 + 65536)
        report["bounds"] = {"max_seconds": max_seconds, "distance_query_reservation": query_reservation,
                            "report_byte_reservation": report_reservation, "maximum_report_bytes": MAX_REPORT_BYTES,
                            "deadline_scope": "Cooperative; complete source verification has its existing independent 30s bound; native calls are not preemptible."}
        if query_reservation > MAX_DISTANCE_QUERIES or report_reservation > MAX_REPORT_BYTES:
            raise ValueError("Clearance query/report reservation exceeds the frozen bounds")
        _deadline(deadline)
        stage = "model_compilation"
        before, after = _Kinematics(base), _Kinematics(changed)
        groups = _comparison_columns(before.columns, after.columns)
        report.update(pair_columns={"baseline": before.columns, "alternative": after.columns}, comparison_columns=groups)
        stats = [{"checked_samples": 0, "baseline_max_penetration_world": None, "alternative_max_penetration_world": None,
                  "increase_count": 0, "above_numerical_tolerance_count": 0, "new_penetration_count": 0,
                  "maximum_increase_world": 0., "worst_increase_sample": None,
                  "worst_alternative_sample": None, "worst_alternative_pair": None} for _ in groups]
        report["comparison_summary"] = stats
        stage = "sampling"
        for sample_index, row in enumerate(rows):
            if not row["weights"]:
                continue
            _deadline(deadline)
            # Preserve completed individual queries even if the other side later fails.
            row["baseline_gaps_world"] = a = [None] * len(before.pairs)
            before.measure(row["weights"], row["reference_time_s"], deadline, a)
            row["alternative_gaps_world"] = b = [None] * len(after.pairs)
            after.measure(row["weights"], row["reference_time_s"], deadline, b)
            increases = []
            for group_index, (group, stat) in enumerate(zip(groups, stats)):
                ai = min(group["baseline_pair_indices"], key=a.__getitem__)
                bi = min(group["alternative_pair_indices"], key=b.__getitem__)
                pa, pb = max(0., -a[ai]), max(0., -b[bi])
                delta = pb - pa
                stat["checked_samples"] += 1
                stat["baseline_max_penetration_world"] = max(stat["baseline_max_penetration_world"] or 0., pa)
                if stat["alternative_max_penetration_world"] is None or pb > stat["alternative_max_penetration_world"]:
                    stat.update(alternative_max_penetration_world=pb, worst_alternative_sample=sample_index,
                                worst_alternative_pair=bi)
                if delta > 0:
                    increases.append([group_index, ai, bi, delta])
                    stat["increase_count"] += 1
                    stat["above_numerical_tolerance_count"] += int(delta > NUMERICAL_TOLERANCE)
                    stat["new_penetration_count"] += int(pa == 0)
                    if delta > stat["maximum_increase_world"]:
                        stat.update(maximum_increase_world=delta, worst_increase_sample=sample_index)
            row.update(status="checked", baseline_gaps_world=a, alternative_gaps_world=b, penetration_increases=increases)
        report["status"] = "complete"
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, TimeoutError, OverflowError, RecursionError) as error:
        report["status"] = "incomplete" if isinstance(error, TimeoutError) or stage == "sampling" else "unsupported_input"
        report["reasons"].append({"stage": stage, "type": type(error).__name__, "message": str(error)[:1000]})
    for row in report["samples"]:
        if row["status"] == "pending":
            present = any(gap is not None for side in ("baseline_gaps_world", "alternative_gaps_world") for gap in row[side] or [])
            row["status"] = "partially_checked" if present else "not_checked"
            if not present:
                row["baseline_gaps_world"] = row["alternative_gaps_world"] = None
    report["summary"] = {"scheduled": len(report["samples"]),
                         "status_counts": dict(Counter(row["status"] for row in report["samples"])),
                         "kind_counts": dict(Counter(row["kind"] for row in report["samples"])),
                         "unavailable_pose_counts": dict(Counter(row["pose_status"] for row in report["samples"] if row["status"] == "unavailable")),
                         "distance_queries_attempted": sum(model.distance_queries for model in (before, after) if model is not None),
                         "finite_gaps_retained": sum(gap is not None for row in report["samples"]
                                                     for side in ("baseline_gaps_world", "alternative_gaps_world") for gap in row[side] or []),
                         "pointwise_increases": sum(s["increase_count"] for s in report.get("comparison_summary", [])),
                         "new_penetrations": sum(s["new_penetration_count"] for s in report.get("comparison_summary", [])),
                         "increase_scope": "Complete paired samples, semantic union minima; individual part gaps remain inspectable."}
    report["seconds"] = time.monotonic() - started
    if len(_encoded(report)) > MAX_REPORT_BYTES:
        raise RuntimeError("Internal clearance report exceeded its conservative reservation")
    return report
