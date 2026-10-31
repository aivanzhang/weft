"""Analytical static-support receipts; report checks never construct a native world."""

import builtins
import json
import socket
import subprocess

import pytest

from test_exercise_build_report import Links, put, run, seal, sha
from weft.io import canonical_hash


@pytest.fixture(autouse=True)
def no_runtime(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"mujoco", "gymnasium", "cv2", "mediapipe", "torch", "PIL"}:
            pytest.fail("Offline reports must not import native simulation or media/inference runtimes")
        return original(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("Offline reports must not start a subprocess or network call")

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)


def support_build(root, *, status="ready", feedback="authored_unchanged"):
    root.mkdir()
    video = b"\x1aE\xdf\xa3analytical source; never decoded"
    source_sha = sha(video)
    human = {"source_sha256": source_sha, "reference": [{"time_s": t} for t in (0., .1, .2)]}
    obj = {"id": "bench", "label": "Authored bench", "role": "static", "shape": "box", "size": [.6, .4, .2],
        "position": [1., 1., .3], "provenance": {"mode": "authored_addition", "calibration_status": "estimated_prior",
        "physical_validation": "unverified", "source_contact_verified": False,
        "assumptions": ["Analytical support shape, not observed contact."]}}
    requirements = [{"object_id": "bench", "human_segment": "left_forearm", "direction_world": [0., 0., 1.],
        "minimum_load_body_weight_fraction": .05, "start_step": 0, "end_step": 6}]
    declared = {"schema_version": 1, "profile": "shared-static-support-priors-v1", "source_sha256": source_sha,
        "scale": 1., "root_assistance": False, "objects": [obj], "support_requirements": requirements}
    declared_raw = (json.dumps(declared, indent=1) + "  \n").encode()
    put(root, "support-priors.json", declared_raw)
    settings = {"scene_scope": "human_static_supports", "video": "/never-follow-source.webm",
        "static_supports": "/never-follow-priors.json", "support_priors_sha256": sha(declared_raw),
        "body_scale": 1., "root_assistance": False, "full_reference": True,
        "training": {"generations": 1, "population": 4, "evaluation_seed": 101}}
    put(root, "settings.json", settings)
    put(root, "body_source/source.webm", video)
    put(root, "body_source/source.json", {"artifact": "source.webm", "sha256": source_sha, "title": "Analytical support fixture"})
    put(root, "body_source/evidence.json", {"schema_version": 1, "profile": "exercise-pose-v1", "source_sha256": source_sha,
        "fps": 30, "frame_count": 10, "decoded_frames": 10, "quality": {"errors": []},
        "frames": [{"id": f"{source_sha}:frame:{frame}", "frame": frame, "time_s": frame / 30,
                    "decoder_timestamp_s": None, "status": "tracked"} for frame in (0, 3, 6)]})
    stages = [{"stage": "body_source", "status": "failed" if status == "preconstruction" else "ready", "elapsed_seconds": 1.}]
    result = {"schema_version": 1, "seal_scope": "complete_immutable_outputs", "status": "ready" if status == "ready" else "failed",
        "lab_status": "ready" if status == "ready" else "not_ready", "stages": stages, "physical_validation": "unverified",
        "reasons": [] if status == "ready" else [{"stage": "body_source" if status == "preconstruction" else "shared_world",
            "message": "Analytical saved rejection; no runtime was executed by this fixture."}]}
    if status != "preconstruction":
        put(root, "body_source/human.json", human)
        body_manifest = b'{"schema_version":1,"scope":"analytical body identity only"}\n'
        resolved = {**declared, "profile": "shared-static-support-construction-v1", "body_source_manifest_sha256": sha(body_manifest)}
        put(root, "support-config.json", resolved)
        put(root, "candidate/inputs/body-source-manifest.json", body_manifest)
        stages.extend([{"stage": "support_priors", "status": "ready", "elapsed_seconds": 2.},
            {"stage": "shared_world", "status": "failed" if status == "rejected" else "ready", "elapsed_seconds": 3.}])
        scene = {"schema_version": 1, "profile": "shared-world-v4", "objects": [obj], "human_spec": human,
            "controlled_id": "human", "target": None, "source_binding": {"mode": "source_qualified", "source_sha256": source_sha},
            "task": {"type": "human_support_v1", "object_goal": None, "horizon_steps": 6, "start_offset_s": 0.,
                "human_limits": {"max_floor_penetration_model_units": .025}}}
        task = {"profile": "shared-support-task-v1", "support_requirements": requirements,
            "control": {"root_assistance": "disabled", "object_forces": "forbidden"}}
        put(root, "candidate/scene.json", scene)
        put(root, "candidate/exercise-task.json", task)
        result.update(scene_sha256=canonical_hash(scene), task_sha256=canonical_hash(task))
        if status != "rejected":
            checks = [{"kind": kind, "time_s": stamp, "source_reference_index": index,
                "max_penetration_model_units": 0., "passes_floor_limit": True, "passes_coupled_preflight": True,
                "floor_contact_count": 0, "floor_contacts": [], "static_conflict_count": 0, "static_conflicts": []}
                for kind, stamp, index in [("source_pose", row["time_s"], i) for i, row in enumerate(human["reference"])]
                + [("control_grid", i / 30, None) for i in range(7)]]
            put(root, "candidate/reference-feasibility.json", {"schema_version": 1, "profile": "shared-world-feasibility-v1",
                "physical_validation": "unverified", "status": "feasible", "scene_sha256": result["scene_sha256"],
                "checks": checks, "initial_intersections": [], "summary": {"source_pose_checks": 3, "control_grid_checks": 7,
                    "checks": 10, "failed_source_poses": 0, "failed_control_grid_poses": 0, "max_penetration_model_units": 0.,
                    "static_conflict_poses": 0, "initial_cross_entity_intersections": 0, "physics_steps": 0}})
            training = {"policy_sha256": "c" * 64, "selected_feedback_kind": feedback,
                "control_steps": 12, "evaluated_episodes": 2, "completed_generations": 1}
            selection = {"policy_sha256": "c" * 64, "manifest_sha256": "d" * 64,
                "selection_status": "not_applicable", "performed": False, "evaluation_used": False,
                "directory": "/never-follow/training", "reason": "Use the optimizer-frozen policy directly."}
            stages.append({"stage": "learning_training", "status": "failed" if status == "training_failed" else "trained", "elapsed_seconds": 4.})
            evaluation = None
            if status == "ready":
                stages.extend([{"stage": "learning_training_selection", "status": "not_applicable", "performed": False, "elapsed_seconds": 5.},
                    {"stage": "learning_independent_evaluation", "status": "evaluated", "elapsed_seconds": 6.},
                    {"stage": "learning_interactive_lab", "status": "ready", "elapsed_seconds": 7.}])
                evaluation = {"trials_per_controller": 2, "comparison_complete": True,
                    "controllers": {name: {"successes": 0} for name in ("coast", "random", "authored", "learned")}}
                put(root, "learning_workflow/lab/run/scene.json", scene)
            put(root, "learning_workflow/result.json", {"profile": "shared-support-learning-workflow-v1",
                "status": "ready" if status == "ready" else "failed", "scene_sha256": result["scene_sha256"], "task_sha256": result["task_sha256"],
                "training": training if status == "ready" else None, "selection": selection if status == "ready" else None,
                "evaluation": evaluation})
    put(root, "result.json", result)
    put(root, "stages.json", {"stages": stages})
    seal(root)
    return root


@pytest.mark.parametrize("feedback", ["authored_unchanged", "searched_joint_residuals"])
def test_ready_support_report_preserves_priors_and_skips_only_secondary_selection(tmp_path, feedback):
    source, out = support_build(tmp_path / "build", feedback=feedback), tmp_path / "report"
    _, summary = run(source, out)
    states = {row["stage"]: row["state"] for row in summary["stages"]}
    assert "body_seed" not in states and "equipment_priors" not in states
    assert states["support_priors"] == states["learning_training"] == states["learning_independent_evaluation"] == "completed"
    assert states["learning_training_selection"] == "not_applicable"
    assert summary["equipment"]["status"] == "authored_static_supports"
    assert summary["equipment"]["body_manifest_identity"] == "matches_copied_body_manifest"
    assert summary["equipment"]["source_identity"] == "matches_preserved_source"
    assert summary["reference_feasibility"]["status"] == "feasible"
    assert summary["task_success"] == "not_assessed_by_report" and summary["physical_validation"] == "unverified"
    assert all(count == 0 for count in summary["work"].values())
    assert summary["learning"]["selection"]["performed"] is False
    assert all(row["successes"] == 0 for row in summary["learning"]["evaluation"]["controllers"].values())
    copied = {row["source_name"]: row for row in summary["artifacts"]}
    for name in ("support-priors.json", "support-config.json", "candidate/inputs/body-source-manifest.json"):
        assert (out / copied[name]["path"]).read_bytes() == (source / name).read_bytes()
    assert not any("selected/" in name for name in copied)
    html = (out / "index.html").read_text()
    assert "not a person-only scene or an equipment-absence assertion" in html
    assert "not applicable and not performed" in html and "no feasibility-selection pass is claimed" in html
    assert ("unchanged programmed feedback, not new learned corrections" if feedback == "authored_unchanged"
            else "phase-dependent joint corrections") in html
    assert "/never-follow-priors.json" not in html
    links = Links(); links.feed(html)
    assert "script" not in links.tags and all(link.startswith("#") or (out / link).is_file() for link in links.links)


@pytest.mark.parametrize("status", ["preconstruction", "rejected", "training_failed"])
def test_stopped_support_report_retains_failures_and_unrun_stages(tmp_path, status):
    out = tmp_path / "report"
    _, summary = run(support_build(tmp_path / "build", status=status), out)
    states = {row["stage"]: row["state"] for row in summary["stages"]}
    assert summary["saved_lab_status"] == "not_ready" and summary["reasons"]
    assert "body_seed" not in states and "equipment_priors" not in states
    assert states["learning_training_selection"] == states["learning_independent_evaluation"] == "not_run"
    assert states["learning_training"] == ("stopped" if status == "training_failed" else "not_run")
    if status == "preconstruction":
        assert states["support_priors"] == states["shared_world"] == "not_run"
        assert summary["equipment"]["status"] == "declared_static_supports"
        assert summary["equipment"]["resolved_config"] == {}
    elif status == "rejected":
        assert states["shared_world"] == "stopped"
    assert "training_feedback" not in summary["learning"] and "selection_scope" not in summary["learning"]
    html = (out / "index.html").read_text()
    assert "No interactive lab was produced" in html and "Analytical saved rejection" in html
    assert "not applicable and not performed" not in html


@pytest.mark.parametrize("name", ["support-priors.json", "support-config.json"])
def test_support_artifact_hashes_are_verified_before_publishing(tmp_path, name):
    source, out = support_build(tmp_path / "build"), tmp_path / "report"
    with (source / name).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="hash differs"):
        run(source, out)
    assert not out.exists()


@pytest.mark.parametrize("change", ["resolved_object", "source_pin", "body_pin", "person_only_scene", "performed_selection", "training_identity"])
def test_contradictory_support_metadata_cannot_be_relabelled_as_ready(tmp_path, change):
    source, out = support_build(tmp_path / "build"), tmp_path / "report"
    name = {"resolved_object": "support-config.json", "source_pin": "settings.json", "body_pin": "support-config.json",
            "person_only_scene": "candidate/scene.json", "performed_selection": "learning_workflow/result.json",
            "training_identity": "learning_workflow/result.json"}[change]
    document = json.loads((source / name).read_bytes())
    if change == "resolved_object": document["objects"][0]["position"][0] += .1
    elif change == "source_pin": document["support_priors_sha256"] = "b" * 64
    elif change == "body_pin": document["body_source_manifest_sha256"] = "b" * 64
    elif change == "person_only_scene": document["profile"] = "shared-world-v3"
    elif change == "performed_selection": document["selection"]["performed"] = True
    else: document["training"].pop("policy_sha256")
    put(source, name, document); seal(source)
    messages = {"resolved_object": "Resolved support config differs", "source_pin": "source/settings identities",
                "body_pin": "body-manifest identity differs", "person_only_scene": "Saved scene contradicts",
                "performed_selection": "secondary feasibility selection", "training_identity": "selected feedback identity"}
    with pytest.raises(ValueError, match=messages[change]):
        run(source, out)
    assert not out.exists()
