"""Saved-reference inspection, including malformed joins and bounded reports."""

from collections import Counter
from copy import deepcopy
import json

import pytest

from weft import exercise_build_report as build_report
from weft import reference_equipment_report as inspection
from weft.io import canonical_hash
from test_exercise_build_report import fixture as build_fixture, put, run, seal


def fixture(count=3, source_sha="a" * 64):
    objects = [{"id": name, "role": role, "shape": "box", "size": [.2, .3, .4],
                "position": [index, 0., .2], "quaternion": [1., 0., 0., 0.]}
               for index, (name, role) in enumerate((("weight-a", "passive"), ("weight-b", "controlled"), ("bench", "static")))]
    poses = [{"time_s": .1 + i / 30, "joints": {f"joint-{j}": j / 100 for j in range(19)}} for i in range(count)]
    scene = {"profile": "shared-world-v2", "human_spec": {"source_sha256": source_sha, "reference": poses,
        "root_reference": [{"time_s": p["time_s"], "translation": [i*.005, 0., 0.]} for i, p in enumerate(poses)],
        "reference_evidence": [{"time_s": p["time_s"], "evidence_id": f"{source_sha}:frame:{2 + 4*i}"} for i, p in enumerate(poses)]},
        "source_binding": {"mode": "source_qualified", "source_sha256": source_sha}, "objects": objects,
        "action_contract": {"names": [f"action-{i}" for i in range(28)]},
        "coordinate_frame": {"axes": "x_right_y_depth_z_up", "scale_bridge": {"body_height_to_world_scale": 2., "origin_world": [1., 1., 0.]},
                             "units": {"length": "body_scaled_world_units"}}}
    frames = []
    for index, pose in enumerate(poses):
        frames.append({"profile": scene["profile"], "coordinate_frame": deepcopy(scene["coordinate_frame"]),
            "source_time_s": pose["time_s"], "time_s": pose["time_s"] - poses[0]["time_s"],
            "steps": index, "episode_steps": 0, "substeps": 0,
            "human": {"reference": True, "angles": deepcopy(pose["joints"]), "origin": [1., 1., 0.],
                      "mobility": {"translation": [index*.005, 0., 0.]}},
            "success": False, "terminated": False, "truncated": False,
            "objects": [{**deepcopy(obj), "virtual_force": [0., 0., 0.]} for obj in objects],
            "normalized_action": [0.] * 28, "virtual_force": [0., 0., 0.],
            "contact_details": [], "contact_count": 0, "contact_summary": dict.fromkeys(inspection.KINDS, 0)})
    reference = {"schema_version": 1, "profile": "shared-world-kinematic-reference-v1", "frames": frames,
                 "reference_poses_total": count, "reference_poses_included": count}
    return scene, reference


def contact(name="weight-a", *, distance=-.02, geometry="right_elbow_geom", kind="human_object"):
    return {"objects": sorted([name, geometry]), "kind": kind, "distance": distance,
            "position": [.1, .2, .3], "normal_force": 12345.}


def contacts(frame, values):
    frame["contact_details"] = values
    frame["contact_count"] = len(values)
    counts = Counter(row["kind"] for row in values)
    frame["contact_summary"] = {kind: counts[kind] for kind in inspection.KINDS}


def test_counts_unique_poses_separately_from_contacts_and_keeps_exact_source_gaps():
    scene, reference = fixture()
    contacts(reference["frames"][0], [contact(distance=0.)])
    contacts(reference["frames"][1], [contact(), contact(distance=-.01), contact("bench", kind="human_static")])
    contacts(reference["frames"][2], [contact("weight-b", distance=-.04), contact("mobile_floor", kind="human_floor")])
    before = deepcopy((scene, reference))
    value = inspection.summarize_reference_equipment(scene, reference)
    assert value["status"] == "overlap_reported" and value["poses_with_equipment_contacts"] == 3
    assert value["poses_with_equipment_overlap"] == 2 and value["negative_contact_records"] == 3
    assert value["contact_records_inspected"] == 6 and value["dynamic_objects"] == 2
    a, b = value["objects"]
    assert a["poses_with_overlap"] == a["poses_with_contacts"] - 1 == 1
    assert a["negative_contact_records"] == 2 and a["max_penetration_model_units"] == .01
    assert a["first_overlap"]["source_frame"] == 6 and b["first_overlap"]["source_frame"] == 10
    assert value["examples"][0]["evidence_id"] == "a" * 64 + ":frame:6"
    assert "normal_force" not in json.dumps(value) and "not interpreted" in value["scope"]
    assert (scene, reference) == before


def test_zero_overlap_is_not_certified_clearance_or_task_success():
    scene, reference = fixture()
    contacts(reference["frames"][0], [contact(distance=.0001)])
    value = inspection.summarize_reference_equipment(scene, reference)
    assert value["status"] == "no_overlap_in_saved_poses" and value["poses_with_equipment_overlap"] == 0
    assert value["physical_validation"] == "unverified" and "intervals between poses" in value["scope"]
    assert "success" not in value and value["objects"][0]["first_overlap"] is None


def test_analytical_source_has_no_invented_source_frames():
    scene, reference = fixture()
    scene["source_binding"] = {"mode": "analytic_fixture", "source_sha256": None}
    scene["human_spec"].update(source_sha256=None, reference_evidence=None)
    contacts(reference["frames"][0], [contact()])
    value = inspection.summarize_reference_equipment(scene, reference)
    assert value["source_sha256"] is None and value["examples"][0]["source_frame"] is None


@pytest.mark.parametrize("fault", ["missing_pose", "clock", "scene_frame", "position", "geometry", "action", "physics",
    "not_reference", "angles", "source_identity", "source_order", "contact_count", "contact_summary", "classification",
    "nan", "bool_distance", "unknown_object", "contact_bound", "unknown_role", "root", "origin", "root_clock"])
def test_contradictory_or_incomplete_metadata_is_rejected(fault):
    scene, reference = fixture()
    frame = reference["frames"][1]
    contacts(frame, [contact()])
    if fault == "missing_pose": reference["frames"].pop()
    elif fault == "clock": frame["source_time_s"] += .01
    elif fault == "scene_frame": frame["coordinate_frame"]["axes"] = "different"
    elif fault == "position": frame["objects"][0]["position"][0] += .01
    elif fault == "geometry": frame["objects"][0]["size"][0] += .01
    elif fault == "action": frame["normalized_action"][0] = .1
    elif fault == "physics": frame["substeps"] = 16
    elif fault == "not_reference": frame["human"]["reference"] = False
    elif fault == "angles": frame["human"]["angles"]["joint-0"] = .5
    elif fault == "source_identity": scene["human_spec"]["reference_evidence"][1]["evidence_id"] = "b" * 64 + ":frame:6"
    elif fault == "source_order": scene["human_spec"]["reference_evidence"][1]["evidence_id"] = "a" * 64 + ":frame:2"
    elif fault == "contact_count": frame["contact_count"] = 0
    elif fault == "contact_summary": frame["contact_summary"]["human_object"] = 0
    elif fault == "classification": contacts(frame, [contact(kind="object_object")])
    elif fault == "nan": frame["contact_details"][0]["distance"] = float("nan")
    elif fault == "bool_distance": frame["contact_details"][0]["distance"] = False
    elif fault == "unknown_object": frame["contact_details"][0]["objects"] = ["other", "right_elbow_geom"]
    elif fault == "contact_bound": contacts(frame, [contact()] * 257)
    elif fault == "unknown_role": scene["objects"][0]["role"] = "unknown"
    elif fault == "root": frame["human"]["mobility"]["translation"][0] += .01
    elif fault == "origin": frame["human"]["origin"][0] += .01
    elif fault == "root_clock": scene["human_spec"]["root_reference"][1]["time_s"] += .01
    with pytest.raises(ValueError):
        inspection.summarize_reference_equipment(scene, reference)


def test_full_reference_bound_and_compact_examples_preserve_complete_counts():
    scene, reference = fixture(900)
    for frame in reference["frames"]:
        contacts(frame, [contact()])
    checks = []
    value = inspection.summarize_reference_equipment(scene, reference, check=lambda: checks.append(1))
    assert value["reference_poses"] == value["poses_with_equipment_overlap"] == value["negative_contact_records"] == 900
    assert len(value["examples"]) == 12 and value["examples_omitted"] == 888 and len(checks) == 902
    assert len(json.dumps(value)) < 16384
    scene, reference = fixture(901)
    with pytest.raises(ValueError): inspection.summarize_reference_equipment(scene, reference)


def test_deadline_and_unsupported_profile_do_not_become_zero_error():
    scene, reference = fixture()
    def expired(): raise TimeoutError("Bound reached")
    with pytest.raises(TimeoutError): inspection.summarize_reference_equipment(scene, reference, check=expired)
    scene["profile"] = "shared-world-v6"
    value = inspection.summarize_reference_equipment(scene, reference)
    assert value["status"] == "unavailable" and "poses_with_equipment_overlap" not in value


def saved_build(root):
    source = build_fixture(root, ready=True)
    source_sha = json.loads((source / "body_source/source.json").read_text())["sha256"]
    scene, reference = fixture(source_sha=source_sha)
    contacts(reference["frames"][1], [contact()])
    prefix = "lab/run/"
    put(source, prefix + "scene.json", scene)
    put(source, prefix + "human-reference.json", reference)
    put(source, prefix + "exercise-lab.json", {"schema_version": 1, "profile": "shared-world", "scene_sha256": canonical_hash(scene)})
    result = json.loads((source / "result.json").read_text()); result["scene_sha256"] = canonical_hash(scene)
    put(source, "result.json", result); seal(source)
    return source


def test_public_report_exposes_exact_artifacts_without_changing_lab_or_task_status(tmp_path):
    source = saved_build(tmp_path / "source")
    before = {p.relative_to(source).as_posix(): p.read_bytes() for p in source.rglob("*") if p.is_file()}
    out = tmp_path / "report"; _, summary = run(source, out)
    diagnostic = summary["reference_equipment"]
    assert diagnostic["poses_with_equipment_overlap"] == 1 and summary["saved_lab_status"] == "ready"
    assert summary["task_success"] == "not_assessed_by_report" and all(value == 0 for value in summary["work"].values())
    copied = {row["source_name"]: row for row in summary["artifacts"]}
    for name, digest in diagnostic["selected_members_sha256"].items():
        assert copied[name]["sha256"] == digest and (out / copied[name]["path"]).read_bytes() == before[name]
    html = (out / "index.html").read_text()
    assert "Sandbox prepared; reference overlaps equipment" in html and "before any action is applied" in html
    assert "not verified native-video synchronization" in html and "Kinematic solver forces are not interpreted" in html
    assert {p.relative_to(source).as_posix(): p.read_bytes() for p in source.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("fault", ["missing_scene", "contract_identity", "build_identity", "video_identity", "changed_saved_bytes", "two_labs_disagree"])
def test_public_report_rejects_unbound_or_contradictory_reference_inputs(tmp_path, fault):
    source = saved_build(tmp_path / "source")
    if fault == "missing_scene": (source / "lab/run/scene.json").unlink()
    elif fault == "contract_identity": put(source, "lab/run/exercise-lab.json", {"schema_version": 1, "profile": "shared-world", "scene_sha256": "b" * 64})
    elif fault == "build_identity":
        value = json.loads((source / "result.json").read_text()); value["scene_sha256"] = "b" * 64; put(source, "result.json", value)
    elif fault == "video_identity":
        scene, reference = fixture(source_sha="b" * 64)
        put(source, "lab/run/scene.json", scene); put(source, "lab/run/human-reference.json", reference)
        put(source, "lab/run/exercise-lab.json", {"schema_version": 1, "profile": "shared-world", "scene_sha256": canonical_hash(scene)})
        value = json.loads((source / "result.json").read_text()); value["scene_sha256"] = canonical_hash(scene); put(source, "result.json", value)
    elif fault == "two_labs_disagree":
        for name in ("scene.json", "human-reference.json", "exercise-lab.json"):
            raw = (source / "lab/run" / name).read_bytes()
            put(source, "learning_workflow/lab/run/" + name, raw + b" ")
    if fault != "changed_saved_bytes": seal(source)
    else: (source / "lab/run/human-reference.json").write_bytes(b"{}\n")
    with pytest.raises(ValueError): run(source, tmp_path / "report")


def test_missing_or_oversized_reference_is_unavailable_not_a_partial_success(tmp_path, monkeypatch):
    source = saved_build(tmp_path / "source")
    ref_name = "lab/run/human-reference.json"
    value = json.loads((source / ref_name).read_text()); value["scope"] = "x" * 32768
    put(source, ref_name, value); seal(source)
    monkeypatch.setattr(build_report, "MAX_JSON_BYTES", 16384)
    _, summary = run(source, tmp_path / "large")
    assert summary["reference_equipment"]["status"] == "unavailable"
    assert "no poses were sampled or clipped" in summary["reference_equipment"]["reason"]
    assert ref_name not in {row["source_name"] for row in summary["artifacts"]}
    source = build_fixture(tmp_path / "legacy", ready=True)
    _, summary = run(source, tmp_path / "missing")
    assert summary["reference_equipment"]["status"] == "unavailable" and summary["saved_lab_status"] == "ready"


def test_remaining_metadata_budget_does_not_trigger_partial_reads(tmp_path):
    import time

    source = build_report._Source(saved_build(tmp_path / "source"), time.monotonic() + 10)
    try:
        source.consumed = build_report.MAX_METADATA_BYTES
        before = dict(source.pins)
        result = build_report._reference_equipment(source)
        assert result["status"] == "unavailable" and "remaining report metadata budget" in result["reason"]
        assert source.pins == before
    finally:
        import os
        os.close(source.fd)
