"""Kinematic comparisons retain missing evidence and every compound collision part."""

from copy import deepcopy
import hashlib
import json
import math

import mujoco
import pytest

from weft import shared_world_clearance as module
from weft.io import canonical_hash, write_json
from weft.observations import adapt_exercise
from weft.shared_exercise_task import build_shared_exercise_task
from weft.shared_world import validate_shared_world
from weft.shared_world_evidence import build_shared_world_binding
from weft.shared_world_lab import authored_shared_world_scene
from test_shared_world_packaging import candidate as mapped_candidate, seal


@pytest.fixture
def scene():
    result = authored_shared_world_scene()
    result["task"]["horizon_steps"] = 60
    return validate_shared_world(result)


def compound(obj, parts=None):
    result = deepcopy(obj)
    if parts is None:
        part = {"id": "original", "shape": obj["shape"], "position": [0., 0., 0.], "quaternion": [1., 0., 0., 0.]}
        part.update({key: obj[key] for key in ("size", "radius") if key in obj})
        parts = [part]
    result.pop("size", None)
    result.pop("radius", None)
    result.update(shape="compound", geometry={"profile": "rigid-compound-v1", "origin": "authored_hypothesis",
        "assumptions": ["Analytical test, not measured geometry."],
        "inertia": {"mode": "authored_centered_diagonal", "diagonal": [.004, .004, .004]}, "parts": parts})
    return result


def source_fixture(tmp_path, scene):
    scene = deepcopy(scene)
    human, source = scene["human_spec"], "a" * 64
    indices = [i for i in range(0, 61, 2) if i != 30]
    human["reference"] = [human["reference"][i] for i in indices]
    human["root_reference"] = [human["root_reference"][i] for i in indices]
    frames = []
    human["reference_evidence"] = []
    for index, ref, root in zip(indices, human["reference"], human["root_reference"]):
        frame = index + 2
        ref["time_s"] = root["time_s"] = frame / 30
        identifier = f"{source}:frame:{frame}"
        human["reference_evidence"].append({"time_s": ref["time_s"], "evidence_id": identifier})
        frames.append({"id": identifier, "frame": frame, "time_s": frame / 30, "status": "tracked",
                       "landmarks": [{"x": .2, "y": .4, "z": 0.}]})
    frames.append({"id": f"{source}:frame:32", "frame": 32, "time_s": 32 / 30,
                   "status": "not_detected", "landmarks": []})
    human["source_sha256"] = source
    human["mobility"]["floor_fit"]["reference_time_s"] = human["reference"][0]["time_s"]
    evidence = {"profile": "exercise-pose-v1", "source_sha256": source, "width": 640, "height": 480,
                "fps": 30, "frame_count": 65, "decoded_frames": 65, "frames": frames}
    document = adapt_exercise(evidence)
    raw = json.dumps(document, sort_keys=True).encode()
    digest = hashlib.sha256(raw).hexdigest()
    ids = ["human", *(obj["id"] for obj in scene["objects"])]
    scene["source_binding"], _ = build_shared_world_binding({digest: raw}, source_sha256=source, world_ids=ids,
        entity_refs={identifier: (digest, "person-0") if identifier == "human" else None for identifier in ids})
    scene = validate_shared_world(scene)
    directory = tmp_path / "source"
    directory.mkdir()
    target = directory / scene["source_binding"]["evidence_documents"][digest]
    target.parent.mkdir()
    target.write_bytes(raw)
    write_json(directory / "scene.json", scene)
    seal(directory)
    return scene, directory, target


def test_identical_compound_representation_retains_all_parts_and_never_steps(scene, monkeypatch):
    alternative = deepcopy(scene)
    alternative["objects"] = [compound(obj) for obj in scene["objects"]]
    original = deepcopy((scene, alternative))
    monkeypatch.setattr(mujoco, "mj_step", lambda *a, **k: pytest.fail("clearance cannot integrate physics"))
    result = module.compare_shared_world_clearance(scene, alternative)
    assert result["status"] == "complete", result["reasons"]
    assert result["summary"]["kind_counts"] == {"reference_pose": 61, "full_span_grid": 61, "task_control": 61}
    assert result["source"]["status"] == "analytic_fixture" and result["physics_steps"] == 0
    assert all(row["status"] == "checked" for row in result["samples"])
    assert all(row["part_id"] == "original" for row in result["pair_columns"]["alternative"] if row["object_id"])
    assert all(s["above_numerical_tolerance_count"] == 0 for s in result["comparison_summary"])
    assert max(abs(a - b) for row in result["samples"]
               for a, b in zip(row["baseline_gaps_world"], row["alternative_gaps_world"])) < 1e-12
    assert (scene, alternative) == original
    result["protocol"]["samples"].clear()
    assert module.clearance_protocol()["samples"]


def test_new_second_part_overlap_is_not_hidden_by_first_part_or_initial_admission(scene):
    alternative = deepcopy(scene)
    alternative["objects"][0] = compound(scene["objects"][0], [
        {"id": "far", "shape": "sphere", "position": [0., .3, 0.], "quaternion": [1., 0., 0., 0.], "radius": .02},
        {"id": "near", "shape": "sphere", "position": [0., 0., 0.], "quaternion": [1., 0., 0., 0.], "radius": .18},
    ])
    result = module.compare_shared_world_clearance(scene, alternative)
    assert result["status"] == "complete", result["reasons"]
    assert result["summary"]["new_penetrations"] > 0
    first = result["samples"][0]
    witnesses = [result["pair_columns"]["alternative"][row[2]] for row in first["penetration_increases"]]
    assert any(row["part_id"] == "near" and row["kind"] == "human_rigid" for row in witnesses)
    assert not any("pass" in key for key in result)


def test_positive_penetration_increases_below_tolerance_remain_visible(scene):
    scene["objects"][0]["radius"] = .18
    alternative = deepcopy(scene)
    alternative["objects"][0]["radius"] += 1e-10
    result = module.compare_shared_world_clearance(scene, alternative)
    assert result["status"] == "complete", result["reasons"]
    assert result["summary"]["pointwise_increases"] > 0
    assert sum(row["above_numerical_tolerance_count"] for row in result["comparison_summary"]) == 0
    assert any(row["penetration_increases"] for row in result["samples"])


def test_maximum_rigid_parts_are_all_measured_within_reserved_output(scene):
    extra = deepcopy(scene["objects"][2])
    extra["id"] = "extra_static"
    extra["position"] = [.4, 1.5, .4]
    scene["objects"].append(extra)
    scene["task"]["horizon_steps"] = 1
    for index, obj in enumerate(scene["objects"]):
        scene["objects"][index] = compound(obj, [
            {"id": f"part-{i}", "shape": "sphere", "position": [(i - 3.5) * .02, 0., 0.],
             "quaternion": [1., 0., 0., 0.], "radius": .01} for i in range(8)])
    result = module.compare_shared_world_clearance(scene, scene)
    assert result["status"] == "complete", result["reasons"]
    for side in ("baseline", "alternative"):
        assert sum(row["kind"] == "rigid_floor" for row in result["pair_columns"][side]) == 32
    queries = len(result["samples"]) * sum(len(cols) for cols in result["pair_columns"].values())
    assert result["summary"]["finite_gaps_retained"] == result["summary"]["distance_queries_attempted"] == queries
    assert queries <= result["bounds"]["distance_query_reservation"]
    assert len(json.dumps(result).encode()) < result["bounds"]["report_byte_reservation"]


def test_parent_and_local_rotation_compose_for_box_sphere_and_capsule_floor_gaps(scene):
    q = [math.sqrt(.5), 0., math.sqrt(.5), 0.]
    scene["objects"][1]["quaternion"] = q
    alternative = deepcopy(scene)
    alternative["objects"][1] = compound(scene["objects"][1], [
        {"id": "box", "shape": "box", "position": [-.1, 0., 0.], "quaternion": q, "size": [.1, .2, .3]},
        {"id": "round", "shape": "sphere", "position": [.05, 0., 0.], "quaternion": [1., 0., 0., 0.], "radius": .03},
        {"id": "bar", "shape": "capsule", "position": [.1, 0., 0.], "quaternion": q, "radius": .02, "length": .2},
    ])
    result = module.compare_shared_world_clearance(scene, alternative)
    assert result["status"] == "complete", result["reasons"]
    columns, gaps = result["pair_columns"]["alternative"], result["samples"][0]["alternative_gaps_world"]
    by_part = {row["part_id"]: gaps[i] for i, row in enumerate(columns)
               if row["kind"] == "rigid_floor" and row["object_id"] == "marker_box"}
    z = scene["objects"][1]["position"][2]
    assert by_part == pytest.approx({"box": z - .05, "round": z - .08, "bar": z - .22}, abs=1e-12)
    assert any(row["object_role"] == "static" and row["kind"] == "rigid_floor" for row in columns)


def test_complete_source_slots_and_full_interval_survive_missing_poses(tmp_path, scene):
    scene, directory, _ = source_fixture(tmp_path, scene)
    scene["task"]["horizon_steps"] = 3
    write_json(directory / "scene.json", scene)
    seal(directory)
    before = {p.relative_to(directory): p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    result = module.compare_shared_world_clearance(scene, scene, source_run=directory)
    assert result["status"] == "complete", result["reasons"]
    slots = [r for r in result["samples"] if r["kind"] == "source_slot"]
    assert [r["source_frame"] for r in slots] == list(range(65))
    assert sum(r["status"] == "checked" for r in slots) == 30
    assert slots[32]["visibility"] == "not_detected" and slots[32]["status"] == "unavailable"
    assert slots[1]["partition"] == "unsampled" and slots[1]["baseline_gaps_world"] is None
    grid = [r for r in result["samples"] if r["kind"] == "full_span_grid"]
    assert grid[0]["reference_time_s"] == 0 and grid[-1]["reference_time_s"] == 64 / 30
    assert grid[0]["status"] == grid[-1]["status"] == "unavailable"
    assert grid[32]["pose_status"] == "interpolated_reference" and grid[32]["source_time_s"] is None
    assert result["summary"]["kind_counts"]["task_control"] == 4
    assert result["clocks"]["full_span_s"][1] > result["clocks"]["task_reference_clock_span_s"][1]
    assert result["source"]["clock"]["status"] == "assumed"
    assert {p.relative_to(directory): p.read_bytes() for p in directory.rglob("*") if p.is_file()} == before


def alias_fixture(tmp_path):
    directory = tmp_path / "alias-source"
    scene, name = mapped_candidate(directory, equivalent=True)
    document = json.loads((directory / name).read_bytes())
    rows = [r for r in document["observations"] if r["object_id"] == "person"]
    human = scene["human_spec"]
    human["reference"] = human["reference"][:3]
    human["root_reference"] = human["root_reference"][:3]
    human["reference_evidence"] = []
    for ref, root, row in zip(human["reference"], human["root_reference"], rows):
        ref["time_s"] = root["time_s"] = row["time_s"]
        human["reference_evidence"].append({"time_s": row["time_s"], "evidence_id": row["evidence_ids"][0]})
    scene["task"]["horizon_steps"] = 1
    scene = validate_shared_world(scene)
    write_json(directory / "scene.json", scene)
    seal(directory)
    return scene, directory


def test_verified_alias_keeps_native_decoder_and_reference_clocks_distinct(tmp_path):
    scene, directory = alias_fixture(tmp_path)
    result = module.compare_shared_world_clearance(scene, scene, source_run=directory)
    assert result["status"] == "complete", result["reasons"]
    slots = [r for r in result["samples"] if r["kind"] == "source_slot"]
    assert [r["source_time_s"] for r in slots] == [0., .033, .067]
    assert result["source"]["clock"]["basis"] == "decoder_presentation_timestamps"
    assert result["source"]["document_source_sha256"] != result["source"]["canonical_source_sha256"]
    assert result["source"]["source_proof"]["source_evidence_verified"]
    grid = [r for r in result["samples"] if r["kind"] == "full_span_grid"]
    assert grid[-1]["reference_time_s"] == .067 and len(grid) == 4
    assert grid[1]["pose_status"] == "interpolated_reference"


@pytest.mark.parametrize("damage", ["mapping_file", "outer_manifest", "scene_identity", "reference_clock"])
def test_source_closure_and_clock_failures_stop_before_native_work(tmp_path, monkeypatch, damage):
    scene, directory = alias_fixture(tmp_path)
    mapping = scene["source_binding"]["source_equivalences"][0]["bundle_path"] + "/lossless.mp4"
    if damage == "mapping_file":
        (directory / mapping).write_bytes(b"different")
    elif damage == "outer_manifest":
        manifest = json.loads((directory / "manifest.json").read_bytes())
        del manifest["files"][mapping]
        write_json(directory / "manifest.json", manifest)
    elif damage == "scene_identity":
        scene["objects"][0]["mass"] += .1
    else:
        scene["human_spec"]["reference_evidence"][1]["time_s"] += .001
        write_json(directory / "scene.json", scene)
        seal(directory)
    monkeypatch.setattr(module, "_Kinematics", lambda *a: pytest.fail("unsupported source reached native code"))
    result = module.compare_shared_world_clearance(scene, scene, source_run=directory)
    assert result["status"] == "unsupported_input" and result["reasons"]


@pytest.mark.parametrize("field", ["body", "root", "clock", "task", "pose", "mass", "material", "source_provenance"])
def test_geometry_only_boundary_rejects_non_geometry_changes(scene, field):
    alternative = deepcopy(scene)
    if field == "body": alternative["human_spec"]["lengths"]["forearm"] += .01
    elif field == "root": alternative["human_spec"]["root_reference"][1]["translation"][0] += .01
    elif field == "clock": alternative["coordinate_frame"]["clock_binding"]["source_time_offset_s"] += .01
    elif field == "task": alternative["task"]["horizon_steps"] -= 1
    elif field == "pose": alternative["objects"][0]["position"][0] += .01
    elif field == "mass": alternative["objects"][0]["mass"] += .01
    elif field == "material": alternative["objects"][0]["friction"][0] += .01
    else: alternative["objects"][0]["provenance"]["camera_hypothesis"] = {"authored": True}
    result = module.compare_shared_world_clearance(scene, alternative)
    assert result["status"] == "unsupported_input" and not result["samples"]


def test_exercise_task_allows_only_scene_hash_rebind(scene):
    alternative = deepcopy(scene)
    alternative["objects"][1] = compound(alternative["objects"][1])
    first = build_shared_exercise_task(scene, root_assistance=True)
    second = deepcopy(first)
    second["scene_sha256"] = canonical_hash(validate_shared_world(alternative))
    result = module.compare_shared_world_clearance(scene, alternative, exercise_tasks=(first, second))
    assert result["status"] == "complete", result["reasons"]
    second["control"]["root_assistance"] = "disabled"
    result = module.compare_shared_world_clearance(scene, alternative, exercise_tasks=(first, second))
    assert result["status"] == "unsupported_input" and "only" in result["reasons"][0]["message"]


@pytest.mark.parametrize("bound,value", [("MAX_SAMPLES", 2), ("MAX_DISTANCE_QUERIES", 1), ("MAX_REPORT_BYTES", 60000)])
def test_preflight_reserves_samples_queries_and_report_before_compile(scene, monkeypatch, bound, value):
    monkeypatch.setattr(module, bound, value)
    monkeypatch.setattr(module, "_Kinematics", lambda *a: pytest.fail("budget failure reached native code"))
    result = module.compare_shared_world_clearance(scene, scene)
    assert result["status"] == "unsupported_input"
    assert not any(row["status"] == "checked" for row in result["samples"])


def test_deadline_preserves_complete_schedule_and_already_checked_rows(scene, monkeypatch):
    measure, calls = module._Kinematics.measure, 0
    def stop(self, *args):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise TimeoutError("analytical deadline checkpoint")
        return measure(self, *args)
    monkeypatch.setattr(module._Kinematics, "measure", stop)
    result = module.compare_shared_world_clearance(scene, scene)
    assert result["status"] == "incomplete" and len(result["samples"]) == 183
    assert result["summary"]["status_counts"] == {"checked": 1, "not_checked": 182}
    assert result["samples"][0]["baseline_gaps_world"] is not None
    assert result["samples"][1]["baseline_gaps_world"] is None


def test_distance_failure_retains_unknown_gaps_instead_of_fabricating_clearance(scene, monkeypatch):
    monkeypatch.setattr(mujoco, "mj_geomDistance", lambda *args: 100.)
    result = module.compare_shared_world_clearance(scene, scene)
    assert result["status"] == "incomplete" and result["summary"]["status_counts"] == {"not_checked": 183}
    assert all(row["baseline_gaps_world"] is None for row in result["samples"])
    assert all(row["alternative_max_penetration_world"] is None for row in result["comparison_summary"])
    assert result["summary"]["distance_queries_attempted"] == 1


def test_partial_native_query_results_survive_failure(scene, monkeypatch):
    original, calls = mujoco.mj_geomDistance, 0
    def query(*args):
        nonlocal calls
        calls += 1
        return float("nan") if calls == 5 else original(*args)
    monkeypatch.setattr(mujoco, "mj_geomDistance", query)
    result = module.compare_shared_world_clearance(scene, scene)
    assert result["status"] == "incomplete"
    assert result["summary"]["status_counts"] == {"partially_checked": 1, "not_checked": 182}
    assert result["summary"]["distance_queries_attempted"] == 5
    assert result["summary"]["finite_gaps_retained"] == 4
    assert result["samples"][0]["alternative_gaps_world"] is None
    assert all(row["checked_samples"] == 0 for row in result["comparison_summary"])


def test_unsupported_large_reference_gap_is_not_silently_bridged(scene):
    scene["human_spec"]["reference"] = scene["human_spec"]["reference"][::30]
    scene["human_spec"]["root_reference"] = scene["human_spec"]["root_reference"][::30]
    result = module.compare_shared_world_clearance(scene, scene)
    assert result["status"] == "unsupported_input"
    assert "gap" in result["reasons"][0]["message"] and not result["samples"]


@pytest.mark.parametrize("limit", [True, 0, 61, float("nan"), float("inf")])
def test_invalid_wall_bounds_are_explicitly_unsupported(scene, limit):
    result = module.compare_shared_world_clearance(scene, scene, max_seconds=limit)
    assert result["status"] == "unsupported_input" and result["reasons"]


def test_source_binding_requires_actual_complete_sealed_input(tmp_path, scene):
    scene, _, _ = source_fixture(tmp_path, scene)
    result = module.compare_shared_world_clearance(scene, scene)
    assert result["status"] == "unsupported_input" and "source_run" in result["reasons"][0]["message"]
